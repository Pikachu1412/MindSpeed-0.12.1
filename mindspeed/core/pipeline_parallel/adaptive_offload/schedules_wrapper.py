# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from functools import wraps
from inspect import signature


def _offload_schedule_wrapper(original_func):
    original_signature = signature(original_func)

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        arguments = dict(kwargs)
        if args:
            arguments.update(original_signature.bind_partial(*args, **kwargs).arguments)
        forward_only = arguments.get('forward_only', False)
        config = arguments.get('config')
        if config is None:
            model = arguments.get('model')
            if isinstance(model, (list, tuple)):
                model = model[0] if model else None
            if model is not None:
                from megatron.core.utils import get_model_config
                config = get_model_config(model)
        active = not forward_only and config is not None and getattr(config, 'fine_grained_activation_offloading', False)
        if active:
            from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
                fine_grained_offloading_reset,
            )
            fine_grained_offloading_reset()
            from .auto_activation_memory import enabled
            if enabled():
                from .activation_transfer_scheduler import get_scheduler
                from .adaptive_memory_profiler import AdaptiveMemoryProfiler
                from .fine_grained_activation_offload import PipelineOffloadManager
                get_scheduler().start(AdaptiveMemoryProfiler.get_instance(), PipelineOffloadManager.get_instance())
        result = original_func(*args, **kwargs)
        if active:
            from .fine_grained_activation_offload import PipelineOffloadManager
            from .auto_activation_memory import enabled
            if enabled():
                from .activation_transfer_scheduler import get_scheduler
                get_scheduler().finish()
            PipelineOffloadManager.get_instance().transport_audit.finish()
            from .auto_activation_memory import enabled
            if enabled():
                from .adaptive_memory_profiler import AdaptiveMemoryProfiler
                AdaptiveMemoryProfiler.get_instance().end_activation_phase()
        return result
    return wrapper


def forward_backward_no_pipelining_wrapper(original_func):
    return _offload_schedule_wrapper(original_func)


def forward_backward_pipelining_with_interleaving_wrapper(original_func):
    return _offload_schedule_wrapper(original_func)


def forward_backward_pipelining_without_interleaving_wrapper(original_func):
    return _offload_schedule_wrapper(original_func)
