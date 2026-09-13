# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Wrappers for megatron.core.transformer.transformer_block.TransformerBlock and
# megatron.core.transformer.transformer_layer.TransformerLayer to inject the
# per-layer ``fine_grained_offloading_set_last_layer`` call that the
# LAST_LAYER_NO_OFFLOAD strategy depends on.
#
# Strategy:
#   1. ``TransformerBlock.__init__`` wrapper marks the final layer of every
#      block (one block == one PP / VPP chunk) with ``_fgao_is_last_layer = True``.
#   2. ``TransformerLayer.forward`` wrapper reads that flag and notifies the
#      PipelineOffloadManager via ``fine_grained_offloading_set_last_layer``
#      before delegating to the original forward.
#
# This faithfully mirrors Megatron-0.12.1's transformer_block.py loop
# (lines 522-548) without having to reproduce the entire forward body.

from functools import wraps


def transformer_block_init_wrapper(original_init):
    """After TransformerBlock builds ``self.layers``, mark the last one."""

    @wraps(original_init)
    def wrapper(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if not getattr(self.config, "fine_grained_activation_offloading", False):
            return

        layers = getattr(self, "layers", None)
        if layers is None or len(layers) == 0:
            return

        for layer in layers:
            layer._fgao_is_last_layer = False
        layers[-1]._fgao_is_last_layer = True

    return wrapper


def transformer_layer_forward_wrapper(original_forward):
    """Set the ``is_last_layer`` flag on the chunk handler before forward.

    ``set_last_layer`` requires that a chunk handler exists for the current
    forward pass.  ``GPTModel.forward`` (wrapped elsewhere) creates the chunk
    handler before invoking the model body, so by the time this wrapper runs
    the handler is guaranteed to be in place.
    """

    @wraps(original_forward)
    def wrapper(self, *args, **kwargs):
        if getattr(self.config, "fine_grained_activation_offloading", False):
            from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
                fine_grained_offloading_set_last_layer,
            )
            fine_grained_offloading_set_last_layer(
                getattr(self, "_fgao_is_last_layer", False)
            )
        return original_forward(self, *args, **kwargs)

    return wrapper
