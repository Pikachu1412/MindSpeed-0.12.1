import inspect
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as functional

from .dense_mlp_wrapper import _storage_key
from .unified_offload_optimizer import ChildRecomputeProfile


EXECUTOR = 'megatron_dense_activation_v1'
PROFILE_FIELDS = {'parent_group', 'executor', 'layout', 'sample_count', 'retained_bytes',
                  'workspace_bytes', 'output_bytes', 'recompute_ms'}


class _ActivationView:
    def __init__(self, module, projected, bias):
        self.config = module.config
        self.activation_func = module.activation_func
        self.projected = projected
        self.bias = bias

    def linear_fc1(self, hidden_states):
        return self.projected, self.bias

    def linear_fc2(self, activated):
        return activated, None


def _owned_output(output, inputs):
    """Allow a shape-only view of a fresh activation's entire storage, not input aliases."""
    if not isinstance(output, torch.Tensor) or not output.is_contiguous():
        return False
    if output.storage_offset() != 0:
        return False
    if output.untyped_storage().nbytes() != output.numel() * output.element_size():
        return False
    return _storage_key(output) not in {_storage_key(value) for value in inputs
                                       if isinstance(value, torch.Tensor)}


class DenseActivationBoundary:
    def __init__(self, module, core_forward):
        self.module = module
        self.core_forward = core_forward

    def activation(self, projected, bias, per_token_scale):
        output, output_bias = self.core_forward(
            _ActivationView(self.module, projected, bias), None, per_token_scale)
        if output_bias is not None:
            raise RuntimeError('Activation facade unexpectedly returned an output bias')
        return output

    def forward(self, hidden_states, per_token_scale=None, *, selective=False,
                observation=None, audit=None, child_key=None):
        projected, bias = self.module.linear_fc1(hidden_states)
        activation_inputs = (projected, bias, per_token_scale)
        if selective:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

            checkpoint = CheckpointWithoutOutput()
            called = False

            def activation(*inputs):
                nonlocal called
                if called and audit is not None:
                    audit.record_policy(child_key, 'REPLAY')
                called = True
                return self.activation(*inputs)

            activated = checkpoint.checkpoint(activation, *activation_inputs)
        elif observation is not None:
            activated = observation.activation(self.activation, activation_inputs)
        else:
            activated = self.activation(*activation_inputs)
        output, output_bias = self.module.linear_fc2(activated)
        if per_token_scale is not None and output_bias is not None:
            raise ValueError('Bias is not supported with per_token_scale')
        if selective:
            if not _owned_output(activated, activation_inputs + (hidden_states, output, output_bias)):
                raise RuntimeError('Activation checkpoint cannot discard aliased or non-owned output storage')
            if not output.requires_grad:
                raise RuntimeError('Activation checkpoint requires a backward hook on the FC2 output')
            if audit is not None:
                audit.record_policy(child_key, 'RECOMPUTE')
            checkpoint.discard_output_and_register_recompute(output)
        elif observation is not None:
            observation.probe_copy(activated, activation_inputs + (hidden_states, output, output_bias))
        return output, output_bias


def make_dense_activation_boundary(module):
    if type(module).__module__ != 'megatron.core.transformer.mlp' or type(module).__qualname__ != 'MLP':
        return None
    from megatron.core.transformer.mlp import MLP

    config = module.config
    if (type(module) is not MLP or config.activation_func not in (functional.silu, functional.gelu)
            or module.activation_func is not config.activation_func
            or getattr(config, 'fp8', None) or getattr(config, 'activation_func_fp8_input_store', False)
            or getattr(module, '_offload_dense_mlp', False)):
        return None
    forward = getattr(module.forward, '__func__', module.forward)
    core = inspect.unwrap(MLP.forward)
    allowed_wrapper = Path(__file__).with_name('dense_mlp_wrapper.py').resolve()
    while forward is not core:
        if not hasattr(forward, '__wrapped__') or Path(forward.__code__.co_filename).resolve() != allowed_wrapper:
            return None
        forward = forward.__wrapped__
    if (core.__name__ != 'forward' or core.__qualname__ != 'MLP.forward'
            or Path(core.__code__.co_filename).name != 'mlp.py'):
        return None
    return DenseActivationBoundary(module, core)


def _copy_probe_budget(profiler, output_bytes):
    free, capacity = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    try:
        stats = torch.cuda.memory_stats()
    except (AttributeError, RuntimeError):
        stats = {}
    fields = ('allocated_bytes.all.current', 'active_bytes.all.current',
              'reserved_bytes.all.current', 'inactive_split_bytes.all.current')
    valid = isinstance(stats, dict) and all(type(stats.get(name)) is int and stats[name] >= 0 for name in fields)
    if valid:
        valid = (stats['active_bytes.all.current'] >= stats['allocated_bytes.all.current']
                 and stats['reserved_bytes.all.current'] >= stats['active_bytes.all.current'] + stats['inactive_split_bytes.all.current'])
    pending = inactive = reusable = 0
    if valid:
        allocated = max(allocated, stats['allocated_bytes.all.current'])
        pending = stats['active_bytes.all.current'] - stats['allocated_bytes.all.current']
        inactive = stats['inactive_split_bytes.all.current']
        reusable = max(0, stats['reserved_bytes.all.current'] - allocated - pending - inactive)
    footprint = allocated + pending + inactive
    reserve = max(float(os.environ.get('ADAPTIVE_MEM_RESERVE_MB', '1024')) * 2**20,
                  float(os.environ.get('ADAPTIVE_MEM_RESERVE_FRACTION', '0.05')) * capacity)
    required = math.ceil(output_bytes * 1.1)
    limit = profiler._current_memory_limit_mb() * 2**20
    accepted = math.isfinite(limit) and free + reusable - reserve >= required and footprint + required <= limit
    return {'capacity_bytes': capacity, 'free_bytes': free, 'reusable_bytes': reusable, 'reserve_bytes': reserve,
            'allocator_allocated_bytes': stats[fields[0]] if valid else None,
            'allocator_active_bytes': stats[fields[1]] if valid else None,
            'allocator_reserved_bytes': stats[fields[2]] if valid else None,
            'required_bytes': required, 'allocated_bytes': allocated, 'pending_free_bytes': pending,
            'inactive_split_bytes': inactive, 'footprint_bytes': footprint, 'limit_bytes': limit,
            'allocator_stats_valid': valid, 'accepted': accepted,
            'reoptimization_count': int(getattr(profiler, '_reoptimize_count', 0))}


def _allocation_counters():
    try:
        stats = torch.cuda.memory_stats()
    except (AttributeError, RuntimeError):
        return None
    fields = ('allocated_bytes.all.allocated', 'allocated_bytes.all.freed', 'allocated_bytes.all.current')
    if not isinstance(stats, dict) or any(type(stats.get(name)) is not int or stats[name] < 0 for name in fields):
        return None
    return tuple(stats[name] for name in fields)


def _allocation_volume(before, after):
    if before is None or after is None:
        return None
    allocated = after[0] - before[0]
    freed = after[1] - before[1]
    if allocated < 0 or freed < 0 or after[2] - before[2] != allocated - freed:
        return None
    return allocated


class ChildBoundaryObservation:
    def __init__(self, profiler, parent_group):
        self.profiler = profiler
        self.parent_group = parent_group
        self.parameter_storages = set()
        self.outside_storages = {}
        self.input_storages = {}
        self.inside_activation = False
        self.activation_events = None
        self.copy_events = None
        self.copy_probe_budget = None
        self.activation_workspace_bytes = None
        self.copy_workspace_bytes = None
        self.activation_allocation_counters = None
        self.copy_allocation_counters = None
        self.output_key = None
        self.output_bytes = 0
        self.layout = None

    def record_saved(self, tensor):
        if not self.inside_activation:
            self.outside_storages[_storage_key(tensor)] = tensor.untyped_storage().nbytes()

    def activation(self, function, inputs):
        for value in inputs:
            if isinstance(value, torch.Tensor) and _storage_key(value) not in self.parameter_storages:
                self.input_storages[_storage_key(value)] = value.untyped_storage().nbytes()
        torch.cuda.current_stream().synchronize()
        before = _allocation_counters()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self.inside_activation = True
        try:
            output = function(*inputs)
        finally:
            self.inside_activation = False
        end.record()
        end.synchronize()
        self.activation_events = start, end
        after = _allocation_counters()
        self.activation_allocation_counters = before, after
        self.activation_workspace_bytes = _allocation_volume(before, after)
        return output

    def probe_copy(self, output, inputs):
        if not _owned_output(output, inputs):
            self.copy_probe_budget = {'accepted': False, 'reason': 'unsupported_output',
                                     'reoptimization_count': int(getattr(self.profiler, '_reoptimize_count', 0))}
            return
        output_bytes = output.untyped_storage().nbytes()
        if output_bytes <= 0:
            return
        torch.cuda.current_stream().synchronize()
        self.copy_probe_budget = _copy_probe_budget(self.profiler, output_bytes)
        if not self.copy_probe_budget['accepted']:
            return
        before = _allocation_counters()
        if before is None:
            self.copy_probe_budget.update(accepted=False, reason='missing_allocation_counters')
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            destination = torch.empty_like(output)
            start.record()
            destination.copy_(output)
            end.record()
            end.synchronize()
        after = _allocation_counters()
        self.copy_allocation_counters = before, after
        self.copy_workspace_bytes = _allocation_volume(before, after)
        self.copy_events = start, end
        self.output_key = _storage_key(output)
        self.output_bytes = output_bytes
        self.layout = [list(output.shape), str(output.dtype), list(output.stride()), output_bytes]
        del destination

    def complete(self):
        if (self.activation_events is None or self.copy_events is None
                or self.output_key not in self.outside_storages
                or self.activation_workspace_bytes is None or self.copy_workspace_bytes is None
                or min(self.activation_workspace_bytes, self.copy_workspace_bytes) < self.output_bytes):
            return None
        retained = {key: value for key, value in self.outside_storages.items() if key != self.output_key}
        for key, value in self.input_storages.items():
            retained[key] = max(retained.get(key, 0), value)
        duration = sum(start.elapsed_time(end) for start, end in (self.activation_events, self.copy_events))
        if not math.isfinite(duration) or duration <= 0:
            return None
        return {'parent_group': self.parent_group, 'executor': EXECUTOR, 'layout': self.layout,
                'sample_count': 1, 'retained_bytes': sum(retained.values()),
                'workspace_bytes': self.activation_workspace_bytes + self.copy_workspace_bytes,
                'output_bytes': self.output_bytes, 'recompute_ms': duration}


def merge_child_measurements(measurements, replicas=False):
    if not measurements or any(measurement is None for measurement in measurements):
        return None
    first = measurements[0]
    if any(any(measurement[key] != first[key] for key in ('parent_group', 'executor', 'layout'))
           for measurement in measurements):
        raise ValueError('Child activation profile changed layout or execution boundary')
    result = dict(first)
    for key in ('retained_bytes', 'workspace_bytes', 'output_bytes', 'recompute_ms'):
        result[key] = max(measurement[key] for measurement in measurements)
    counts = [measurement['sample_count'] for measurement in measurements]
    result['sample_count'] = min(counts) if replicas else sum(counts)
    return result


def validate_child_profiles(profiles, module_specs):
    if not isinstance(profiles, dict):
        raise ValueError('Child profiles must be a mapping')
    supported = {child: parent for parent, spec in module_specs.items()
                 for child in spec.get('checkpoint_children', ())}
    for name, value in profiles.items():
        if (name not in supported or not isinstance(value, dict) or set(value) != PROFILE_FIELDS
                or value['parent_group'] != supported[name] or value['executor'] != EXECUTOR):
            raise ValueError('Unknown or unbound child checkpoint profile')
        for field in ('sample_count', 'retained_bytes', 'workspace_bytes', 'output_bytes'):
            if type(value[field]) is not int or value[field] < (1 if field in ('sample_count', 'output_bytes') else 0):
                raise ValueError('Invalid child storage measurement')
        duration = value['recompute_ms']
        if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('Invalid child recompute duration')
        layout = value['layout']
        if (not isinstance(layout, list) or len(layout) != 4 or not isinstance(layout[0], list)
                or not isinstance(layout[1], str) or not isinstance(layout[2], list)
                or len(layout[0]) != len(layout[2]) or layout[3] != value['output_bytes']
                or any(type(size) is not int or size < 0 for size in layout[0] + layout[2])):
            raise ValueError('Invalid child activation layout')


def solver_child_profiles(profiles, module_specs, groups):
    validate_child_profiles(profiles, module_specs)
    return {name: ChildRecomputeProfile(value['parent_group'], value['recompute_ms'],
                                       value['retained_bytes'] / 2**20,
                                       value['workspace_bytes'] / 2**20)
            for name, value in profiles.items()
            if value['sample_count'] >= 2 and value['parent_group'] in groups}


def needs_child_profile(profiler, parent_group):
    measured = getattr(profiler, '_auto_child_profiles', {}).get(parent_group + '.activation', {}).get('sample_count', 0)
    pending = sum(observation.parent_group == parent_group
                  for observation in getattr(profiler, '_pending_child_profiles', ()))
    return measured + pending < 2


def finish_child_observations(profiler):
    measurements = getattr(profiler, '_auto_child_profiles', {})
    for observation in getattr(profiler, '_pending_child_profiles', ()):
        budget = getattr(observation, 'copy_probe_budget', None)
        if budget is not None:
            profiler.__dict__.setdefault('_auto_child_probe_budgets', {})[observation.parent_group] = budget
        value = observation.complete()
        if value is None:
            continue
        name = observation.parent_group + '.activation'
        previous = measurements.get(name)
        measurements[name] = merge_child_measurements([previous, value] if previous else [value])
    profiler._auto_child_profiles = measurements
    profiler._pending_child_profiles = []
