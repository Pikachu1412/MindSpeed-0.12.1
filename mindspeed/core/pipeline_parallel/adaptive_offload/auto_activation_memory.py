import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import sys
from functools import wraps
from pathlib import Path


AUTO_DEFAULTS = {
    'MEGATRON_ADAPTIVE_OFFLOAD': '1', 'MEGATRON_LAST_LAYER_NO_OFFLOAD': '0',
    'MEGATRON_H2D_PREFETCH': '0', 'MEGATRON_CROSS_LAYER_PREFETCH': '0',
    'MEGATRON_PREFETCH_DEPTH': '1', 'MEGATRON_LAYER_PREFETCH_DEPTH': '1',
    'MEGATRON_LAYER_PREFETCH_DELAY_GROUPS': '0', 'MEGATRON_OFFLOAD_EVENT_SYNC': '0',
    'MEGATRON_PINNED_MEMORY_POOL': '0', 'MEGATRON_OFFLOAD_CPU_AFFINITY': '',
    'ADAPTIVE_MEM_WARMUP_SKIP_ITERS': '2', 'ADAPTIVE_MEM_PROFILE_ITERS': '4',
    'ADAPTIVE_MEM_PER_PP_RANK': '1', 'ADAPTIVE_MEM_BUDGET_MB': 'auto',
    'ADAPTIVE_MEM_RESERVE_MB': '1024', 'ADAPTIVE_MEM_RESERVE_FRACTION': '0.05',
    'ADAPTIVE_MEM_ACTIVATION_MARGIN': '1.1', 'ADAPTIVE_MEM_GUARD_INTERVAL': '1',
    'ADAPTIVE_MEM_REOPTIMIZE_INTERVAL': '64', 'ADAPTIVE_RECOMPUTE_FACTOR': '1.25',
    'ADAPTIVE_TRANSFER_FACTOR': '1.1',
}


def enabled(args=None):
    return bool(getattr(args, 'auto_activation_memory', False)) or os.environ.get('AUTO_ACTIVATION_MEMORY') == '1'


def configure(args):
    if not enabled(args):
        return
    if getattr(args, 'recompute_granularity', None) is not None or getattr(args, 'recompute_activation_function', False):
        raise ValueError('Automatic activation memory owns checkpoint decisions; remove manual recompute options')
    if getattr(args, 'swap_optimizer', False) or getattr(args, 'optimizer_cpu_offload', False):
        raise ValueError('Automatic activation memory does not enable optimizer swap/CPU offload')
    if getattr(args, 'fp8', None) or getattr(args, 'enable_cuda_graph', False) or getattr(args, 'use_torch_compile', False):
        raise ValueError('Automatic activation memory currently supports eager FP32/BF16/FP16, not FP8 or captured/compiled graphs')
    args.auto_activation_memory = True
    args.fine_grained_activation_offloading = True
    args.offload_modules = []
    os.environ['AUTO_ACTIVATION_MEMORY'] = '1'
    os.environ.update(AUTO_DEFAULTS)
    os.environ.setdefault('PYTORCH_NPU_ALLOC_CONF', 'expandable_segments:True')


def _tensor_layout(args, kwargs):
    import torch
    from torch.utils._pytree import tree_flatten

    leaves, structure = tree_flatten((args, kwargs))
    positions = tuple(index for index, value in enumerate(leaves) if isinstance(value, torch.Tensor))
    tensors = tuple(leaves[index] for index in positions)
    template = tuple(None if isinstance(value, torch.Tensor) else value for value in leaves)
    return tensors, positions, template, structure


def _restore(tensors, positions, template, structure):
    from torch.utils._pytree import tree_unflatten

    leaves = list(template)
    for position, tensor in zip(positions, tensors):
        leaves[position] = tensor
    return tree_unflatten(leaves, structure)


def _checkpoint(original, inputs, positions, template, structure, profiler, layer_number, key, is_moe, owner, boundary=None):
    import torch
    from torch.utils._pytree import tree_flatten, tree_unflatten
    from megatron.core import tensor_parallel

    output_layout = {}
    called = False

    def replay(*tensor_inputs):
        nonlocal called
        replaying = called
        called = True
        if replaying and enabled():
            from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import PipelineOffloadManager
            PipelineOffloadManager.get_instance().transport_audit.record_policy(key, 'REPLAY')
        restored_args, restored_kwargs = _restore(tensor_inputs, positions, template, structure)
        context = profiler.profile_module(layer_number, key, is_moe, tensor_inputs, recompute=replaying) if profiler.is_profiling_active() else contextlib.nullcontext()
        child_observation = None
        if boundary is not None and replaying and profiler.is_profiling_active() and torch.cuda.is_available():
            from .dense_activation_boundary import ChildBoundaryObservation, needs_child_profile
            if needs_child_profile(profiler, key):
                child_observation = ChildBoundaryObservation(profiler, key)
        observation = observe_saved_activations(owner, profiler, key, child_observation) if replaying and profiler.is_profiling_active() else contextlib.nullcontext()
        with context, observation:
            result = (boundary.forward(*restored_args, observation=child_observation, **restored_kwargs)
                      if child_observation is not None else original(*restored_args, **restored_kwargs))
        if replaying and profiler.is_profiling_active() and torch.cuda.is_available():
            observe_backward(result, tensor_inputs, profiler, key)
        leaves, output_structure = tree_flatten(result)
        output_positions = tuple(index for index, value in enumerate(leaves)
                                 if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter))
        if output_layout and (output_layout['positions'] != output_positions or output_layout['structure'] != output_structure):
            raise RuntimeError(f'Checkpoint output structure changed for {key}')
        output_layout.update(positions=output_positions, structure=output_structure,
                             template=tuple(None if index in output_positions else value for index, value in enumerate(leaves)))
        return tuple(leaves[index] for index in output_positions)

    outputs = tensor_parallel.checkpoint(replay, False, *inputs)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    leaves = list(output_layout['template'])
    for position, tensor in zip(output_layout['positions'], outputs):
        leaves[position] = tensor
    return tree_unflatten(leaves, output_layout['structure'])


def wrap_module(module, layer_number, kind, order, is_moe, profiler):
    import torch
    from torch.utils._pytree import tree_flatten, tree_unflatten
    from mindspeed.core.pipeline_parallel.adaptive_offload import fine_grained_activation_offload as runtime
    from mindspeed.core.pipeline_parallel.adaptive_offload.dense_mlp_wrapper import _activation_saved_tensor_hooks

    key = f'layer{layer_number}.{kind}'
    profiler._auto_module_specs[key] = {'layer': layer_number, 'kind': kind, 'order': order, 'is_moe': is_moe,
                                      'class': type(module).__module__ + '.' + type(module).__qualname__}
    original = module.forward
    boundary = None
    child_key = key + '.activation'
    if kind == 'dense_mlp' and not is_moe:
        from .dense_activation_boundary import make_dense_activation_boundary
        boundary = make_dense_activation_boundary(module)
        if boundary is not None:
            profiler._auto_module_specs[key]['checkpoint_children'] = [child_key]
    observed_shapes = None

    @wraps(original)
    def forward(*args, **kwargs):
        nonlocal observed_shapes
        if not torch.is_grad_enabled():
            return original(*args, **kwargs)
        inputs, positions, template, structure = _tensor_layout(args, kwargs)
        if not inputs:
            return original(*args, **kwargs)
        shapes = tuple((tuple(tensor.shape), str(tensor.dtype), str(tensor.device)) for tensor in inputs)
        signatures = getattr(profiler, '_auto_input_signatures', {})
        encoded_shapes = [[list(shape), dtype, device] for shape, dtype, device in shapes]
        cached_shapes = signatures.get(key)
        if (observed_shapes is not None and shapes != observed_shapes) or (cached_shapes is not None and cached_shapes != encoded_shapes):
            raise RuntimeError(f'Activation input signature changed for {key}; refusing a stale memory plan. Restart profiling for the new shape')
        observed_shapes = shapes
        signatures[key] = encoded_shapes
        profiler._auto_input_signatures = signatures
        active_plan = profiler._plan if profiler.is_optimization_applied() else None
        action = active_plan.decisions.get(key, 'KEEP') if active_plan is not None else 'RECOMPUTE'
        child_action = active_plan.decisions.get(child_key, 'KEEP') if active_plan is not None else 'KEEP'
        if child_action != 'KEEP' and (child_action != 'RECOMPUTE' or action != 'KEEP' or boundary is None):
            raise RuntimeError(f'Unbound or conflicting child activation policy for {key}')
        from .activation_transfer_scheduler import get_scheduler
        scheduler = get_scheduler()
        ticket = scheduler.begin(runtime.PipelineOffloadManager.get_instance().cur_forward_chunk(), key, action) if scheduler.active else None
        if enabled():
            runtime.PipelineOffloadManager.get_instance().transport_audit.record_policy(key, action)
        if action == 'RECOMPUTE':
            return scheduler.attach(ticket, _checkpoint(original, inputs, positions, template, structure, profiler, layer_number, key, is_moe, module, boundary))
        context = profiler.profile_module(layer_number, key, is_moe, inputs) if profiler.is_profiling_active() else contextlib.nullcontext()
        with context:
            if action == 'KEEP':
                child_observation = None
                if boundary is not None and child_action == 'KEEP' and profiler.is_profiling_active() and torch.cuda.is_available():
                    from .dense_activation_boundary import ChildBoundaryObservation, needs_child_profile
                    if needs_child_profile(profiler, key):
                        child_observation = ChildBoundaryObservation(profiler, key)
                observation = observe_saved_activations(module, profiler, key, child_observation) if profiler.is_profiling_active() and child_action == 'KEEP' else contextlib.nullcontext()
                with observation:
                    if child_action == 'RECOMPUTE':
                        result = boundary.forward(*args, selective=True, child_key=child_key,
                                                  audit=runtime.PipelineOffloadManager.get_instance().transport_audit,
                                                  **kwargs)
                    elif child_observation is not None:
                        result = boundary.forward(*args, observation=child_observation, **kwargs)
                    else:
                        result = original(*args, **kwargs)
                if profiler.is_profiling_active() and torch.cuda.is_available():
                    observe_backward(result, inputs, profiler, key)
                return scheduler.attach(ticket, result)
            manager = runtime.PipelineOffloadManager.get_instance()
            changed_inputs = (runtime.fine_grained_offloading_group_start(inputs[0], name=key), *inputs[1:])
            restored_args, restored_kwargs = _restore(changed_inputs, positions, template, structure)
            with runtime.get_fine_grained_offloading_context(True), _activation_saved_tensor_hooks(module, manager):
                result = original(*restored_args, **restored_kwargs)
            leaves, output_structure = tree_flatten(result)
            output_positions = tuple(index for index, value in enumerate(leaves)
                                     if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter))
            if ticket is not None:
                scheduler.prepare_offload(ticket, ticket.chunk._groups_to_offload[-1][0])
            outputs = runtime.fine_grained_offloading_group_commit(*(leaves[index] for index in output_positions), name=key)
            if ticket is not None:
                scheduler.commit(ticket)
            elif enabled():
                torch.cuda.current_stream().wait_stream(manager.d2h_stream)
            for position, output in zip(output_positions, outputs):
                leaves[position] = output
            return scheduler.attach(ticket, tree_unflatten(leaves, output_structure))

    module.forward = forward
    return key


@contextlib.contextmanager
def observe_saved_activations(module, profiler, key, child_observation=None):
    import torch
    from mindspeed.core.pipeline_parallel.adaptive_offload.dense_mlp_wrapper import _storage_key
    from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import PipelineOffloadManager
    from mindspeed.core.pipeline_parallel.adaptive_offload.saved_activation_identity import SavedActivationIdentity

    storages = {_storage_key(parameter) for parameter in module.parameters()}
    if child_observation is not None:
        child_observation.parameter_storages = storages
    checker = PipelineOffloadManager.get_instance().cur_forward_chunk().tensor_need_offloading_checker
    observed_bytes = 0
    resident_bytes = 0
    identities = SavedActivationIdentity()
    resident_identities = SavedActivationIdentity()
    resident_count = 0
    saved_sizes = {}

    def pack(tensor):
        nonlocal observed_bytes, resident_bytes, resident_count
        if isinstance(tensor, torch.nn.Parameter):
            return tensor
        detached = tensor.detach()
        if _storage_key(tensor) not in storages:
            if child_observation is not None:
                child_observation.record_saved(tensor)
            if checker(tensor):
                tensor_bytes = tensor.numel() * tensor.element_size()
                observed_bytes += tensor_bytes
                tag = len(saved_sizes)
                saved_sizes[tag] = tensor_bytes
                identities.record(tag, tensor, witness=detached)
            else:
                resident_bytes += tensor.numel() * tensor.element_size()
                resident_identities.record(resident_count, tensor, witness=detached)
                resident_count += 1
        return detached

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        yield
    physical_bytes = observed_bytes - sum(saved_sizes[tag] for tag in identities.aliases())
    profiler.record_group_offload_bytes(key, physical_bytes, logical_bytes=observed_bytes)
    stats = profiler._get_or_create_offload_group(key)
    stats.non_offloadable_bytes = max(stats.non_offloadable_bytes, resident_bytes)
    eligible = identities.storage_footprint(len(saved_sizes))
    resident = resident_identities.storage_footprint(resident_count)
    materialized = identities.contiguous_copy_bytes(len(saved_sizes))
    incomplete = eligible is None or resident is None or materialized is None
    stats.storage_footprint_incomplete = getattr(stats, "storage_footprint_incomplete", False) or incomplete
    if resident is not None:
        stats.resident_storage_bytes = max(getattr(stats, "resident_storage_bytes", 0), sum(resident.values()))
    if not incomplete:
        combined = dict(resident)
        for storage, nbytes in eligible.items():
            combined[storage] = max(combined.get(storage, 0), nbytes)
        resident_storage_bytes = sum(resident.values())
        keep_storage_bytes = sum(combined.values()) - resident_storage_bytes
        stats.keep_storage_bytes = max(getattr(stats, "keep_storage_bytes", 0), keep_storage_bytes)
        stats.storage_footprint_sample_count = getattr(stats, "storage_footprint_sample_count", 0) + 1
        if physical_bytes:
            stats.d2h_storage_ratio = max(getattr(stats, "d2h_storage_ratio", 1.0),
                                          (keep_storage_bytes + materialized) / physical_bytes)
    if child_observation is not None:
        profiler.__dict__.setdefault('_pending_child_profiles', []).append(child_observation)


def observe_backward(output, inputs, profiler, key):
    import torch
    from torch.utils._pytree import tree_flatten

    source = next((tensor for tensor in inputs if tensor.requires_grad), None)
    targets = [tensor for tensor in tree_flatten(output)[0]
               if isinstance(tensor, torch.Tensor) and tensor.requires_grad and not isinstance(tensor, torch.nn.Parameter)]
    if source is None or not targets:
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    started = False

    def begin(gradient):
        nonlocal started
        if not started:
            start.record()
            started = True
        return gradient

    def finish(gradient):
        if started:
            end.record()
            profiler.enqueue_backward_compute_events(key, start, end)
        return gradient

    for target in targets:
        target.register_hook(begin)
    source.register_hook(finish)


def transformer_layer_init_wrapper(original_init):
    @wraps(original_init)
    def wrapper(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        from megatron.core.transformer.identity_op import IdentityOp
        from megatron.core.transformer.moe.moe_layer import BaseMoELayer
        from mindspeed.core.pipeline_parallel.adaptive_offload.adaptive_memory_profiler import AdaptiveMemoryProfiler

        profiler = AdaptiveMemoryProfiler.get_instance()
        is_moe = isinstance(self.mlp, BaseMoELayer)
        self._is_moe_layer = is_moe
        for order, (attribute, kind) in enumerate((('input_layernorm', 'attn_norm'), ('self_attention', 'attention'),
                                                 ('pre_cross_attn_layernorm', 'cross_norm'), ('cross_attention', 'cross_attention'),
                                                 ('pre_mlp_layernorm', 'mlp_norm'), ('mlp', 'moe_mlp' if is_moe else 'dense_mlp'))):
            module = getattr(self, attribute, None)
            if module is not None and not isinstance(module, IdentityOp):
                wrap_module(module, self.layer_number, kind, order, is_moe, profiler)
    return wrapper


def optimizer_capacity_preflight_wrapper(original):
    @wraps(original)
    def wrapper(config, model_chunks, *args, **kwargs):
        import torch
        from megatron.core import parallel_state

        if not enabled():
            return original(config, model_chunks, *args, **kwargs)
        if getattr(config, 'optimizer', 'adam') != 'adam' or getattr(config, 'optimizer_cpu_offload', False):
            raise ValueError('Automatic activation-memory preflight currently requires device-resident Adam')
        parameters = {id(parameter): parameter for model in model_chunks for parameter in model.parameters() if parameter.requires_grad}
        distributed_optimizer = getattr(config, 'use_distributed_optimizer', False)
        dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True) if distributed_optimizer else 1
        expert_dp_size = parallel_state.get_expert_data_parallel_world_size() if distributed_optimizer else 1
        parameter_count = sum(parameter.numel() for parameter in parameters.values())
        for field in ('main_params_dtype', 'main_grads_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype'):
            if str(getattr(config, field, torch.float32)) not in ('torch.float32', 'float32'):
                raise ValueError(f'Unmodeled optimizer storage dtype for {field}; the activation-only preflight requires FP32 states')
        projected_new_bytes = sum(
            parameter.numel() * (8 + (4 if parameter.dtype in (torch.float16, torch.bfloat16) else 0))
            // max(dp_size if getattr(parameter, 'allreduce', True) else expert_dp_size, 1)
            for parameter in parameters.values())
        free, total = torch.cuda.mem_get_info()
        capacity = min(total, free + torch.cuda.memory_reserved())
        reserve = max(1024 ** 3, int(capacity * 0.05))
        resident_lower_bound = torch.cuda.memory_allocated() + projected_new_bytes
        record = {'rank': torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                  'resident_lower_bound_bytes': resident_lower_bound, 'capacity_bytes': capacity,
                  'reserve_bytes': reserve, 'parameter_count': parameter_count, 'dp_size': dp_size,
                  'expert_dp_size': expert_dp_size}
        records = [record]
        if torch.distributed.is_initialized():
            records = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(records, record)
        failures = [item for item in records if item['resident_lower_bound_bytes'] > item['capacity_bytes'] - item['reserve_bytes']]
        print('[AutoActivationMemory][CAPACITY] ' + json.dumps(record, sort_keys=True), flush=True)
        if failures:
            raise RuntimeError('Activation-only optimization cannot fit resident parameters/gradients/Adam state; rejected before optimizer allocation: ' + json.dumps(failures, sort_keys=True))
        return original(config, model_chunks, *args, **kwargs)
    return wrapper


def report_failure(profiler, exception):
    import torch

    directory = Path.cwd() / '.mindspeed/activation_profiles'
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    record = {'rank': rank, 'iteration': profiler._current_iter,
              'phase': 'optimized_training' if profiler.is_optimization_applied() else 'profiling',
              'error_type': type(exception).__name__, 'error': str(exception),
              'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
              'policy': profiler._plan.to_dict() if profiler._plan else None,
              'handling': 'Abort this distributed attempt; do not skip optimizer updates or resume a partially failed step'}
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'failure.rank{rank}.json').write_text(json.dumps(record, indent=2) + '\n')
    except OSError as error:
        print(f'[AutoActivationMemory][FAILURE_REPORT_WARNING] {error}', flush=True)


def _profile_runtime_sources():
    dependencies = {}
    for name in ('mindspeed.core.megatron_basic.requirements_basic',
                 'mindspeed.features_manager.megatron_basic.requirements_basic'):
        module = sys.modules.get(name)
        dependencies[name] = (hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest()
                              if module is not None else None)
    return dependencies


def profile_fingerprint(model, args):
    import torch

    parameters = sorted((name, tuple(parameter.shape), str(parameter.dtype)) for chunk in model for name, parameter in chunk.named_parameters())
    fields = ('micro_batch_size', 'global_batch_size', 'seq_length', 'tensor_model_parallel_size',
              'pipeline_model_parallel_size', 'data_parallel_size', 'context_parallel_size', 'expert_model_parallel_size',
              'swiglu', 'bias_swiglu_fusion', 'bias_gelu_fusion', 'use_flash_attn', 'attention_dropout',
              'hidden_dropout', 'normalization', 'qk_layernorm', 'position_embedding_type', 'rotary_percent',
              'optimizer', 'use_distributed_optimizer', 'reuse_fp32_param', 'main_params_dtype', 'main_grads_dtype',
              'exp_avg_dtype', 'exp_avg_sq_dtype', 'overlap_grad_reduce', 'overlap_param_gather',
              'sequence_parallel', 'moe_grouped_gemm', 'moe_router_load_balancing_type',
              'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads', 'num_query_groups',
              'kv_channels', 'num_experts', 'moe_ffn_hidden_size', 'moe_router_topk', 'moe_layer_freq',
              'moe_expert_capacity_factor', 'moe_pad_expert_input_to_capacity', 'moe_token_dispatcher_type',
              'rotary_base', 'use_fused_rotary_pos_emb', 'use_fused_rmsnorm')
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob('*.py')}
    module_types = [(name, type(module).__module__, type(module).__qualname__) for chunk in model for name, module in chunk.named_modules()]
    module_sources = {}
    for module_type in {type(module) for chunk in model for module in chunk.modules()}:
        try:
            path = inspect.getsourcefile(module_type)
            if path and Path(path).is_file() and path not in module_sources:
                module_sources[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except (TypeError, OSError):
            continue
    runtime_versions = {'cuda': torch.version.cuda, 'ascend_path': os.environ.get('ASCEND_HOME_PATH')}
    for package in ('torch-npu', 'transformer-engine', 'apex', 'megatron-core'):
        try:
            runtime_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime_versions[package] = None
    ascend_path = os.environ.get('ASCEND_HOME_PATH')
    if ascend_path:
        for relative in ('version.info', 'compiler/version.info', 'runtime/version.info'):
            path = Path(ascend_path) / relative
            if path.is_file():
                runtime_versions[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    signature = {'schema': 1, 'parameters': parameters, 'module_types': module_types,
                 'configuration': {field: getattr(args, field, None) for field in fields}, 'source': sources,
                 'module_source': module_sources, 'runtime_versions': runtime_versions,
                 'runtime_source': _profile_runtime_sources(),
                 'torch_version': str(torch.__version__), 'allocator': os.environ.get('PYTORCH_NPU_ALLOC_CONF', ''),
                 'device': str(torch.cuda.get_device_properties(torch.cuda.current_device())) if torch.cuda.is_available() else 'cpu'}
    return hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()


def _decode_profile_cache(candidate, signature, profiler):
    from mindspeed.core.pipeline_parallel.adaptive_offload.adaptive_memory_profiler import LayerStats, LayerModuleStats, OffloadGroupStats, PCIeBandwidthStats

    if candidate['schema'] != 2 or candidate['signature'] != signature or candidate['module_specs'] != profiler._auto_module_specs:
        raise ValueError('Cache schema, signature or module boundaries do not match')
    payload = candidate['payload']
    memory = payload['memory']
    if not set(profiler._memory_telemetry).issubset(memory) or memory['samples'] < 1 or memory['baseline_peak_bytes'] <= 0:
        raise ValueError('Incomplete cache memory telemetry')
    profile = payload['profile']
    from .dense_activation_boundary import validate_child_profiles
    validate_child_profiles(profile.get('children', {}), profiler._auto_module_specs)
    transport_host = profile.get('transport_host', {})
    host_keys = {f'asynchronous:{depth}:{slots}' for depth in (0, 1, 2) for slots in (1, 2)}
    host_keys.update(f'synchronous:{depth}:1' for depth in (0, 1))
    if not isinstance(transport_host, dict) or set(transport_host) - host_keys:
        raise ValueError('Invalid cached transport host modes')
    reference = candidate['reference_validation']
    if not reference or len(reference['samples']) < 2:
        raise ValueError('Cache lacks the two uninstrumented baseline measurements')

    def validate_numbers(value):
        if isinstance(value, dict):
            for child in value.values():
                validate_numbers(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                validate_numbers(child)
        elif isinstance(value, (float, int)) and (not math.isfinite(value) or value < 0):
            raise ValueError('Cache contains an invalid measurement')

    validate_numbers(memory)
    for mode, measurement in transport_host.items():
        if not isinstance(measurement, dict) or set(measurement) != {'module_cpu_ms', 'offload_cpu_ms', 'sample_count'}:
            raise ValueError('Invalid cached transport host measurements')
        validate_numbers(measurement)
        if measurement['sample_count'] < 1:
            raise ValueError('Cached transport host costs lack real samples')
    groups = {key: OffloadGroupStats(**data) for key, data in profile['groups'].items()}
    if set(groups) != set(profiler._auto_module_specs):
        raise ValueError('Cache lacks measurements for registered modules')
    layers = {int(number): LayerStats(layer_number=int(number), is_moe=data['is_moe'],
              modules={key: LayerModuleStats(**module) for key, module in data['modules'].items()})
              for number, data in profile['layers'].items()}
    pcie = PCIeBandwidthStats(**profile['pcie']) if profile['pcie'] else None
    for key, stats in groups.items():
        if key not in profiler._auto_module_specs:
            raise ValueError('Cache contains an unknown module')
        for field, value in vars(stats).items():
            if field != 'layer_number':
                validate_numbers(value)
        if stats.total_offload_bytes > 0 or stats.non_offloadable_bytes > 0:
            spec = profiler._auto_module_specs[key]
            measured = layers[spec['layer']].modules[key]
            validate_numbers(vars(measured))
            if measured.sample_count < 2 or measured.recompute_sample_count < 2 or measured.input_bytes <= 0:
                raise ValueError('Cache lacks module/replay measurements')
    if pcie is not None:
        validate_numbers(vars(pcie))
    signatures = candidate['input_signatures']
    if set(signatures) != set(profiler._auto_module_specs):
        raise ValueError('Cache lacks actual module input signatures')
    return payload, groups, layers, pcie, signatures, reference


def initialize_profile_cache(profiler, model, args):
    import torch

    if not enabled(args):
        return
    profiler._init_pp_rank_profiling()
    training_end = getattr(args, 'train_iters', 0)
    profiler._auto_training_end_iteration = training_end if type(training_end) is int and training_end > 0 else 0
    distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    signature = profile_fingerprint(model, args)
    path = Path.cwd() / '.mindspeed/activation_profiles' / f'{signature}.rank{rank}.json'
    profiler._auto_cache_path = path
    profiler._auto_cache_signature = signature
    decoded = None
    error = None
    try:
        if path.exists():
            candidate = json.loads(path.read_text())
            decoded = _decode_profile_cache(candidate, signature, profiler)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exception:
        error = str(exception)
    availability = torch.tensor([int(decoded is not None)], device='cuda', dtype=torch.int32) if distributed else None
    if distributed:
        torch.distributed.all_reduce(availability, op=torch.distributed.ReduceOp.MIN)
    if decoded is None or (distributed and not availability.item()):
        profiler._warmup_skip_iters = int(getattr(args, 'iteration', 0)) + 2
        profiler._auto_calibration_iteration = profiler._warmup_skip_iters
        print(f'[AutoActivationMemory][CACHE_MISS] rank={rank} signature={signature} reason={error or "new_or_incomplete_signature"}', flush=True)
        return
    value, groups, layers, pcie, signatures, reference = decoded
    profiler._auto_calibration_iteration = None
    profiler._offload_group_stats = groups
    profiler._layer_stats = layers
    profiler._pcie_stats = pcie
    profiler._transport_host_costs = value['profile'].get('transport_host', {})
    profiler._auto_child_profiles = value['profile'].get('children', {})
    profiler._pcie_measured = pcie is not None
    profiler._auto_input_signatures = signatures
    profiler._auto_reference_validation = reference
    profiler._memory_telemetry = dict(value['memory'])
    profiler._memory_telemetry['capacity_bytes'] = 0
    profiler._sample_memory_capacity()
    profiler._profiling_done = True
    profiler._optimization_applied = False
    profiler._auto_cache_loaded = True
    profiler._auto_baseline_validated = True
    print(f'[AutoActivationMemory][CACHE_HIT] rank={rank} signature={signature}; resolving against current capacity, not reusing a stored plan', flush=True)


def save_profile_cache(profiler):
    path = getattr(profiler, '_auto_cache_path', None)
    if path is None or getattr(profiler, '_auto_cache_saved', False):
        return
    value = {'schema': 2, 'signature': profiler._auto_cache_signature,
             'module_specs': profiler._auto_module_specs, 'payload': profiler._profile_payload(),
             'input_signatures': getattr(profiler, '_auto_input_signatures', {}),
             'reference_validation': getattr(profiler, '_auto_reference_validation', None)}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f'.{os.getpid()}.writing')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)
        profiler._auto_cache_saved = True
    except OSError as exception:
        print(f'[AutoActivationMemory][CACHE_WRITE_WARNING] {exception}', flush=True)


def advance_reference_validation(profiler):
    if not enabled() or getattr(profiler, '_auto_baseline_validated', False) or not profiler.is_profiling_done():
        return
    if not hasattr(profiler, '_auto_reference_validation'):
        profiler._auto_reference_validation = {'start_iteration': profiler._current_iter, 'samples': [],
                                               'instrumented_peak_bytes': profiler._memory_telemetry['baseline_peak_bytes']}
        return
    reference = profiler._auto_reference_validation
    if profiler._current_iter <= reference['start_iteration']:
        return
    if reference['samples'] and reference['samples'][-1]['iteration'] == profiler._current_iter:
        return
    reference['samples'].append({'iteration': profiler._current_iter, 'peak_bytes': profiler._iteration_peak_bytes,
                                 'activation_peak_bytes': getattr(profiler, '_activation_phase_peak_bytes', 0) or profiler._iteration_peak_bytes})
    if len(reference['samples']) >= 2:
        peak = max(sample['peak_bytes'] for sample in reference['samples'])
        if peak <= 0:
            raise RuntimeError('Missing uninstrumented recompute-reference memory measurements')
        profiler._memory_telemetry['baseline_peak_bytes'] = peak
        profiler._memory_telemetry['activation_baseline_peak_bytes'] = max(sample['activation_peak_bytes'] for sample in reference['samples'])
        profiler._auto_baseline_validated = True
        print(f'[AutoActivationMemory][BASELINE_VERIFIED] pp_rank={profiler._pp_rank} ' + json.dumps(reference, sort_keys=True), flush=True)
