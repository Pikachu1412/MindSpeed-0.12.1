import json
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from functools import wraps

from .transport_host_profile import TransportHostMeter, host_region, host_timed, update_host_costs


@dataclass(eq=False)
class ModuleTransfer:
    chunk: object
    key: str
    action: str
    ordinal: int
    group_id: int = -1
    size: int = 0
    d2h_done: object = None
    h2d_done: object = None
    consumed: object = None
    backward_started: bool = False


class ByteCredits:
    def __init__(self, limit, meter=None):
        self.limit = int(limit)
        self.meter = meter
        if self.limit < 0:
            raise ValueError('Negative activation transfer budget')
        self.entries = deque()
        self.used = 0
        self.peak = 0
        self.waits = 0

    @host_timed
    def reap(self):
        remaining = deque()
        for owner, size, event in self.entries:
            if event is not None and event.query():
                self.used -= size
            else:
                remaining.append((owner, size, event))
        self.entries = remaining

    @host_timed
    def ensure(self, size, wait=False, reserve=0):
        if size < 0 or reserve < 0:
            raise ValueError('Negative activation transfer reservation')
        if size > self.limit:
            raise RuntimeError(f'Activation transfer {size} exceeds its profiled byte budget {self.limit}')
        if self.used + size + reserve > self.limit:
            self.reap()
        while self.used + size + reserve > self.limit:
            ready = next((entry for entry in self.entries if entry[2] is not None), None)
            if not wait or ready is None:
                return False
            with host_region(self, 'wait'):
                ready[2].synchronize()
            self.waits += 1
            self.reap()
        self.peak = max(self.peak, self.used + size)
        return True

    @host_timed
    def reserve(self, owner, size, event=None, wait=False, reserve=0):
        if not self.ensure(size, wait=wait, reserve=reserve):
            return False
        if size:
            self.entries.append((owner, size, event))
            self.used += size
            self.peak = max(self.peak, self.used)
        return True

    @host_timed
    def release_after(self, owner, event):
        self.entries = deque((entry_owner, size, event if entry_owner is owner else previous)
                             for entry_owner, size, previous in self.entries)


class ActivationTransferScheduler:
    def __init__(self):
        self.active = False
        self.managed = False
        self.mode = 'asynchronous'
        self.measure_transfers = True
        self.meter = None
        self.packets = deque()
        self.tickets = {}
        self.groups = {}

    def start(self, profiler, manager):
        self.flush()
        self.profiler = profiler
        self.manager = manager
        plan = getattr(profiler, '_plan', None)
        execution = getattr(plan, 'execution', {}) or {}
        self.managed = bool(profiler.is_optimization_applied() and execution.get('transport_version') == 1)
        self.active = self.managed and 'OFFLOAD' in plan.decisions.values()
        self.mode = execution.get('transport_mode', 'asynchronous')
        self.execution = execution
        self.counts = Counter()
        self.events = []
        self.audit = bool(os.environ.get('ADAPTIVE_MEM_PROFILE_JSON'))
        self.measure_transfers = self.audit or profiler._current_iter % 8 == 0
        self.meter = TransportHostMeter(self.active and self.measure_transfers)
        self.d2h = ByteCredits(math.ceil(execution.get('d2h_pending_mb', 0) * 2**20), self.meter)
        self.h2d = ByteCredits(math.ceil(execution.get('h2d_live_mb', 0) * 2**20), self.meter)
        self.tickets = {}
        self.groups = {}
        if not self.managed:
            return
        import torch
        from . import fine_grained_activation_offload as runtime
        self.torch = torch
        runtime.RELOAD_EVENT_SYNC_ENABLED = True
        runtime.H2D_PREFETCH_ENABLED = False
        runtime.CROSS_LAYER_PREFETCH_ENABLED = False
        if self.mode not in ('synchronous', 'asynchronous'):
            raise ValueError('Unknown activation transport mode')
        if self.mode == 'synchronous' and (execution['adjacent_prefetch'] > 1 or execution['d2h_slots'] != 1):
            raise ValueError('Synchronous transport requires depth zero/one and one source slot')
        if not self.active:
            return
        self.largest = int(math.ceil(execution['largest_offload_mb'] * 2**20))
        self.depth = execution['adjacent_prefetch']
        self.origin = self.event(torch.cuda.current_stream(), timing=True, optional=True) if self.audit else None
        if self.origin is not None:
            manager.d2h_stream.wait_event(self.origin)
            manager.h2d_stream.wait_event(self.origin)

    def event(self, stream, timing=False, optional=False):
        with host_region(self, 'optional_event' if optional else 'event'):
            event = self.torch.cuda.Event(enable_timing=timing)
            event.record(stream)
            return event

    def stats(self, key):
        return self.profiler._offload_group_stats.get(key)

    def trace(self, kind, ticket, start, end, source=None):
        if start is None or not self.measure_transfers:
            return
        if len(self.events) < 8192:
            self.events.append(({'kind': kind, 'module': ticket.key, 'action': ticket.action,
                                 'chunk': self.chunk_ids[ticket.chunk], 'group_id': ticket.group_id,
                                 'bytes': ticket.size, 'source': source}, start, end))
        else:
            self.counts['trace_events_dropped'] += 1

    @host_timed
    def begin(self, chunk, key, action):
        if not self.active:
            return None
        if self.mode == 'synchronous' and action != 'OFFLOAD':
            return None
        if chunk not in self.tickets:
            self.tickets[chunk] = []
            self.chunk_ids = {handler: index for index, handler in enumerate(self.tickets)}
        stats = self.stats(key)
        expected = int(math.ceil(stats.total_offload_bytes)) if stats is not None and action == 'OFFLOAD' and chunk.should_bulk_offload() else 0
        if expected:
            if not self.d2h.ensure(expected, wait=True):
                raise RuntimeError('No completed D2H source can satisfy the next module reservation')
        ticket = ModuleTransfer(chunk, key, action, len(self.tickets[chunk]))
        self.tickets[chunk].append(ticket)
        ticket.forward_start = self.event(self.torch.cuda.current_stream(), timing=True) if self.audit else None
        self.counts['forward_' + action] += 1
        if self.depth and self.mode == 'asynchronous':
            next_chunk = self.next_chunk(exclude=chunk)
            if next_chunk is not None:
                spec = self.profiler._auto_module_specs.get(key, {})
                lead = sum(self.stats(name).forward_compute_time_ms
                           for name, candidate in self.profiler._auto_module_specs.items()
                           if candidate['order'] >= spec.get('order', 0) and self.stats(name) is not None)
                boundary = stats.forward_compute_time_ms if stats is not None else 0.0
                self.prefetch(next_chunk, lead, 'pp_forward', chunk, next_boundary=boundary)
        return ticket

    @host_timed
    def prepare_offload(self, ticket, group_id):
        if ticket is None:
            return
        ticket.group_id = group_id
        self.groups[(ticket.chunk, group_id)] = ticket
        ticket.transfer_start = None
        if self.measure_transfers:
            with host_region(self, 'optional_dependency'):
                self.manager.d2h_stream.wait_stream(self.torch.cuda.current_stream())
            ticket.transfer_start = self.event(self.manager.d2h_stream, timing=True, optional=True)

    @host_timed
    def commit(self, ticket):
        if ticket is None:
            return
        ticket.size = sum(state[1].numel() * state[1].element_size()
                          for tag, state in ticket.chunk._tensor_tag_to_state.items()
                          if tag[0] == ticket.group_id and isinstance(state, tuple))
        if ticket.size:
            ticket.d2h_done = (self.event(self.manager.d2h_stream, timing=True) if self.measure_transfers
                               else ticket.chunk._offload_events_by_id[ticket.group_id])
            if not self.d2h.reserve(ticket, ticket.size, ticket.d2h_done, wait=True):
                raise RuntimeError('D2H transfer exceeded its safe in-flight source budget')
            self.trace('d2h', ticket, ticket.transfer_start, ticket.d2h_done)
            if self.mode == 'synchronous':
                self.torch.cuda.current_stream().wait_event(ticket.d2h_done)

    @host_timed
    def attach(self, ticket, result):
        if ticket is None:
            return result
        from torch.utils._pytree import tree_flatten
        if self.audit:
            end = self.event(self.torch.cuda.current_stream(), timing=True)
            self.trace('forward_region', ticket, ticket.forward_start, end)
        if self.mode == 'synchronous':
            return result
        leaves, _ = tree_flatten(result)
        seen = set()
        for tensor in leaves:
            if (isinstance(tensor, self.torch.Tensor) and not isinstance(tensor, self.torch.nn.Parameter)
                    and tensor.requires_grad and id(tensor) not in seen):
                seen.add(id(tensor))
                def begin_backward(gradient, transfer=ticket):
                    self.backward_begin(transfer)
                    return gradient
                tensor.register_hook(begin_backward)
        return result

    @host_timed
    def next_chunk(self, exclude=None):
        for chunk in self.manager._queue:
            if chunk is not exclude and chunk in self.tickets:
                if any(ticket.size and not ticket.backward_started for ticket in self.tickets[chunk]):
                    return chunk
        return None

    @host_timed
    def issue(self, ticket, source, demand=False):
        if not ticket.size or ticket.h2d_done is not None:
            return True
        active_demand = max((size for owner, size, event in self.h2d.entries
                             if owner.backward_started and owner.consumed is None), default=0)
        demand_reserve = 0 if demand else max(0, self.largest - active_demand)
        if not self.h2d.reserve(ticket, ticket.size, wait=demand, reserve=demand_reserve):
            return False
        if self.measure_transfers:
            self.manager.h2d_stream.wait_event(ticket.d2h_done)
        start = self.event(self.manager.h2d_stream, timing=True, optional=True) if self.measure_transfers else None
        with host_region(self, 'native_reload'):
            ticket.chunk.bulk_reload_group((ticket.group_id, ticket.key))
        ticket.h2d_done = (self.event(self.manager.h2d_stream, timing=True) if self.measure_transfers
                           else ticket.chunk._reload_events_by_id[ticket.group_id])
        self.trace('h2d', ticket, start, ticket.h2d_done, source)
        self.counts['h2d_' + source] += 1
        return True

    @host_timed
    def prefetch(self, chunk, lead, source, current_chunk, next_boundary=0.0):
        issued = 0
        for ticket in reversed(self.tickets.get(chunk, ())):
            if ticket.backward_started:
                continue
            stats = self.stats(ticket.key)
            if ticket.size and ticket.h2d_done is None:
                duration = stats.h2d_time_ms if stats is not None and stats.h2d_sample_count else 0.0
                if not duration:
                    bandwidth = getattr(self.profiler._pcie_stats, 'h2d_gbps', 0.0)
                    duration = ticket.size / 2**30 / bandwidth * 1000 if bandwidth > 0 else float('inf')
                if lead - next_boundary > duration * 1.25:
                    break
                if issued >= self.depth or not self.issue(ticket, source):
                    break
                issued += 1
                if chunk is not current_chunk:
                    self.counts['cross_chunk_prefetch'] += 1
            if stats is not None:
                lead += stats.backward_compute_time_ms
                if ticket.action == 'RECOMPUTE':
                    lead += stats.forward_compute_time_ms * 1.25

    @host_timed
    def backward_begin(self, ticket):
        if ticket.backward_started:
            return
        ticket.backward_started = True
        self.counts['backward_' + ticket.action] += 1
        if self.audit:
            start = self.event(self.torch.cuda.current_stream(), timing=True)
            self.trace('backward_begin', ticket, start, start)
        if ticket.size and not self.issue(ticket, 'demand', demand=True):
            raise RuntimeError('H2D live budget occupied by unconsumed activations; refusing unsafe reload')
        if self.depth:
            stats = self.stats(ticket.key)
            lead = stats.backward_compute_time_ms if stats is not None else 0.0
            if stats is not None and ticket.action == 'RECOMPUTE':
                lead += stats.forward_compute_time_ms * 1.25
            self.prefetch(ticket.chunk, lead, 'backward_' + ticket.action, ticket.chunk, next_boundary=lead)
            if not any(not candidate.backward_started for candidate in self.tickets[ticket.chunk]):
                next_chunk = self.next_chunk(exclude=ticket.chunk)
                if next_chunk is not None:
                    self.prefetch(next_chunk, lead, 'pp_backward', ticket.chunk, next_boundary=lead)

    @host_timed
    def consumed(self, chunk, group_id):
        ticket = self.groups.get((chunk, group_id))
        if ticket is not None and ticket.h2d_done is not None:
            ticket.consumed = self.event(self.torch.cuda.current_stream(), timing=self.audit)
            self.h2d.release_after(ticket, ticket.consumed)
            if self.audit:
                self.trace('backward_end', ticket, ticket.consumed, ticket.consumed)
                if hasattr(ticket, 'backward_compute_start'):
                    self.trace('backward_region', ticket, ticket.backward_compute_start, ticket.consumed)

    def finish(self):
        if not self.active:
            if self.managed and self.audit:
                prefix = os.environ['ADAPTIVE_MEM_PROFILE_JSON']
                rank = self.torch.distributed.get_rank() if self.torch.distributed.is_initialized() else 0
                summary = {'iteration': self.profiler._current_iter, 'rank': rank, 'execution': self.execution,
                           'fast_path': 'no_offload', 'counts': {}, 'events': [], 'ownership_released': True,
                           'd2h_peak_bytes': 0, 'h2d_peak_bytes': 0,
                           'd2h_backpressure_waits': 0, 'h2d_backpressure_waits': 0}
                with open(f'{prefix}.scheduler.rank{rank}.jsonl', 'a', encoding='utf-8') as output:
                    output.write(json.dumps(summary, sort_keys=True) + '\n')
            self.managed = False
            return
        leftovers = [ticket.key for tickets in self.tickets.values() for ticket in tickets
                     if ticket.size and ticket.consumed is None]
        if leftovers:
            raise RuntimeError(f'Activation transfer ownership was not released after backward: {leftovers}')
        if not self.measure_transfers:
            self.active = False
            self.managed = False
            return
        completion = self.event(self.torch.cuda.current_stream(), optional=True)
        rank = self.torch.distributed.get_rank() if self.torch.distributed.is_initialized() else 0
        prefix = os.environ.get('ADAPTIVE_MEM_PROFILE_JSON')
        summary = {'iteration': self.profiler._current_iter, 'rank': rank, 'execution': self.execution,
                   'counts': dict(self.counts), 'd2h_peak_bytes': self.d2h.peak,
                   'h2d_peak_bytes': self.h2d.peak, 'd2h_backpressure_waits': self.d2h.waits,
                   'h2d_backpressure_waits': self.h2d.waits, 'ownership_released': True}
        self.packets.append((completion, self.origin, self.events, summary,
                             f'{prefix}.scheduler.rank{rank}.jsonl' if prefix else None, self.profiler, self.meter))
        self.active = False
        self.managed = False
        self.flush(wait=len(self.packets) > 2)

    def flush(self, wait=False):
        while self.packets:
            completion, origin, records, summary, path, profiler, meter = self.packets[0]
            if wait:
                completion.synchronize()
            if not completion.query():
                break
            start_cpu = time.thread_time_ns()
            events = []
            for metadata, start, end in records:
                duration = start.elapsed_time(end)
                if path:
                    events.append(dict(metadata, start_ms=origin.elapsed_time(start), duration_ms=duration))
                if metadata['kind'] in ('d2h', 'h2d'):
                    stats = profiler._offload_group_stats.get(metadata['module'])
                    if stats is not None:
                        stats.update_transfer(metadata['kind'], duration)
            flush_cpu = (time.thread_time_ns() - start_cpu) / 1e6
            modules = sum(value for key, value in summary['counts'].items() if key.startswith('forward_'))
            offloads = sum(metadata['kind'] == 'd2h' for metadata, _, _ in records)
            update_host_costs(profiler, summary['execution'].get('transport_mode', 'asynchronous'),
                              meter, modules, offloads, flush_cpu_ms=flush_cpu / 8 if not path else flush_cpu,
                              sampling_divisor=8 if not path else 1,
                              profile_key=summary['execution'].get('host_cost_key',
                                  f"{summary['execution'].get('transport_mode', 'asynchronous')}:{summary['execution']['adjacent_prefetch']}:{summary['execution'].get('d2h_slots', 1)}"))
            summary['events'] = events
            summary['host_profile'] = meter.snapshot()
            summary['flush_cpu_ms'] = flush_cpu
            if path:
                with open(path, 'a', encoding='utf-8') as output:
                    output.write(json.dumps(summary, sort_keys=True) + '\n')
            self.packets.popleft()


_scheduler = ActivationTransferScheduler()


def get_scheduler():
    return _scheduler


def group_backward_begin_wrapper(original):
    @wraps(original)
    def wrapper(handler, name, group_id):
        ticket = _scheduler.groups.get((handler, group_id)) if _scheduler.active else None
        if ticket is not None and _scheduler.mode == 'synchronous':
            _scheduler.backward_begin(ticket)
        start = _scheduler.event(_scheduler.torch.cuda.current_stream(), timing=True) if ticket is not None and _scheduler.audit else None
        result = original(handler, name, group_id)
        if ticket is not None:
            if ticket.h2d_done is not None:
                _scheduler.torch.cuda.current_stream().wait_event(ticket.h2d_done)
            if _scheduler.audit:
                ticket.backward_compute_start = _scheduler.event(_scheduler.torch.cuda.current_stream(), timing=True)
                _scheduler.trace('reload_wait', ticket, start, ticket.backward_compute_start)
        return result
    return wrapper


def group_backward_complete_wrapper(original):
    @wraps(original)
    def wrapper(handler, name, group_id):
        result = original(handler, name, group_id)
        if _scheduler.active:
            _scheduler.consumed(handler, group_id)
        return result
    return wrapper


def legacy_prefetch_wrapper(original):
    @wraps(original)
    def wrapper(*args, **kwargs):
        if _scheduler.managed:
            return None
        return original(*args, **kwargs)
    return wrapper
