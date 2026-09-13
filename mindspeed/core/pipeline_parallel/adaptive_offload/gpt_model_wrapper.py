# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Wrappers for GPTModel.forward and core_transformer_config_from_args
# to inject fine-grained activation offloading initialization.

from functools import wraps


def gpt_model_forward_wrapper(original_forward):
    """Wrap GPTModel.forward to call _preprocess_for_fine_grained_offloading
    before the original forward, mirroring Megatron-0.12.1's gpt_model.py:275-276."""

    @wraps(original_forward)
    def wrapper(self, *args, **kwargs):
        if getattr(self.config, 'fine_grained_activation_offloading', False):
            from megatron.core import parallel_state
            from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
                fine_grained_offloading_init_chunk_handler,
            )
            vp_stage = parallel_state.get_virtual_pipeline_model_parallel_rank()
            fine_grained_offloading_init_chunk_handler(
                vp_size=self.config.virtual_pipeline_model_parallel_size,
                vp_stage=vp_stage,
                min_offloaded_tensor_size=self.config.min_offloaded_tensor_size,
            )
        return original_forward(self, *args, **kwargs)

    return wrapper


def core_transformer_config_from_args_wrapper(original_func):
    """Wrap core_transformer_config_from_args so that fine-grained activation
    offloading args are propagated to the resulting TransformerConfig instance.

    The original function uses dataclasses.fields() to filter args, which only
    keeps fields declared on TransformerConfig. Since MindSpeed adds the offload
    fields as plain class attributes (not dataclass fields), they would be
    silently dropped. We post-process the config to set the attributes from args.
    """

    @wraps(original_func)
    def wrapper(args, *func_args, **func_kwargs):
        config = original_func(args, *func_args, **func_kwargs)

        if hasattr(args, 'fine_grained_activation_offloading'):
            config.fine_grained_activation_offloading = (
                args.fine_grained_activation_offloading
            )
        if hasattr(args, 'offload_modules'):
            config.offload_modules = args.offload_modules
        if hasattr(args, 'min_offloaded_tensor_size'):
            config.min_offloaded_tensor_size = args.min_offloaded_tensor_size

        return config

    return wrapper
