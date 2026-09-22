import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import wraps


class TransportHostMeter:
    def __init__(self, active=False):
        self.active = active
        self.stack = []
        self.metrics = defaultdict(lambda: {'calls': 0, 'wall_ms': 0.0, 'cpu_ms': 0.0})

    @contextmanager
    def region(self, name):
        frame = [0, 0]
        self.stack.append(frame)
        start_wall = time.perf_counter_ns()
        start_cpu = time.thread_time_ns()
        try:
            yield
        finally:
            elapsed_cpu = time.thread_time_ns() - start_cpu
            elapsed_wall = time.perf_counter_ns() - start_wall
            self.stack.pop()
            if self.stack:
                self.stack[-1][0] += elapsed_wall
                self.stack[-1][1] += elapsed_cpu
            metric = self.metrics[name]
            metric['calls'] += 1
            metric['wall_ms'] += max(0, elapsed_wall - frame[0]) / 1e6
            metric['cpu_ms'] += max(0, elapsed_cpu - frame[1]) / 1e6

    def snapshot(self):
        return {key: dict(value) for key, value in self.metrics.items()}


def host_region(owner, name):
    meter = getattr(owner, 'meter', None)
    return meter.region(name) if meter is not None and meter.active else nullcontext()


def host_timed(function):
    @wraps(function)
    def wrapper(self, *args, **kwargs):
        meter = getattr(self, 'meter', None)
        if meter is None or not meter.active:
            return function(self, *args, **kwargs)
        with meter.region(function.__name__):
            return function(self, *args, **kwargs)
    return wrapper


def estimate_host_costs(profiles, mode, depth, slots):
    key = f'{mode}:{depth}:{slots}'
    observed = {name: values for name, values in profiles.items() if values.get('sample_count', 0) > 0}
    if key in observed:
        return dict(observed[key]), 'observed_exclusive_cpu', [key]
    matching = sorted(name for name in observed if name.startswith(mode + ':'))
    if matching:
        costs = {field: max(observed[name].get(field, 0.0) for name in matching)
                 for field in ('module_cpu_ms', 'offload_cpu_ms')}
        source = 'same_mode_conservative_estimate'
    elif observed:
        matching = sorted(observed)
        costs = {
            'module_cpu_ms': max(values.get('offload_cpu_ms', 0.0) for values in observed.values())
            if mode == 'asynchronous' else 0.0,
            'offload_cpu_ms': max(values.get('module_cpu_ms', 0.0) + values.get('offload_cpu_ms', 0.0)
                                  for values in observed.values()),
        }
        source = 'cross_mode_conservative_estimate'
    else:
        costs = {'module_cpu_ms': 0.0, 'offload_cpu_ms': 0.0}
        source = 'unmeasured_lower_bound'
    costs['sample_count'] = 0
    return costs, source, matching


def update_host_costs(profiler, mode, meter, modules, offloads, flush_cpu_ms=0.0, sampling_divisor=1, profile_key=None):
    if not meter.active or not offloads:
        return
    module_names = {'begin', 'attach', 'backward_begin', 'prefetch', 'next_chunk'}
    module_cpu = 0.0
    offload_cpu = flush_cpu_ms
    for name, values in meter.metrics.items():
        if name.startswith('native') or name == 'wait':
            continue
        cpu_ms = values['cpu_ms'] / sampling_divisor if name.startswith('optional') else values['cpu_ms']
        if mode == 'asynchronous' and name in module_names:
            module_cpu += cpu_ms
        else:
            offload_cpu += cpu_ms
    observed = {'module_cpu_ms': module_cpu / max(1, modules),
                'offload_cpu_ms': offload_cpu / offloads}
    profile_key = profile_key or mode
    profiles = getattr(profiler, '_transport_host_costs', {})
    previous = profiles.get(profile_key, {})
    count = previous.get('sample_count', 0)
    weight = 0.25 if count else 1.0
    profiles[profile_key] = {key: previous.get(key, value) * (1 - weight) + value * weight
                      for key, value in observed.items()}
    profiles[profile_key]['sample_count'] = count + 1
    profiler._transport_host_costs = profiles
