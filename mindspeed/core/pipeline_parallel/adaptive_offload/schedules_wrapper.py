# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from functools import wraps


def forward_backward_no_pipelining_wrapper(original_func):
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        forward_only = kwargs.get('forward_only', False)
        config = kwargs.get('config', None)
        if not forward_only and config is not None and getattr(config, 'fine_grained_activation_offloading', False):
            from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
                fine_grained_offloading_reset,
            )
            fine_grained_offloading_reset()
        return original_func(*args, **kwargs)
    return wrapper


def forward_backward_pipelining_with_interleaving_wrapper(original_func):
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        forward_only = kwargs.get('forward_only', False)
        config = kwargs.get('config', None)
        if not forward_only and config is not None and getattr(config, 'fine_grained_activation_offloading', False):
            from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
                fine_grained_offloading_reset,
            )
            fine_grained_offloading_reset()
        return original_func(*args, **kwargs)
    return wrapper


def forward_backward_pipelining_without_interleaving_wrapper(original_func):
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        forward_only = kwargs.get('forward_only', False)
        config = kwargs.get('config', None)
        if not forward_only and config is not None and getattr(config, 'fine_grained_activation_offloading', False):
            from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
                fine_grained_offloading_reset,
            )
            fine_grained_offloading_reset()
        return original_func(*args, **kwargs)
    return wrapper
