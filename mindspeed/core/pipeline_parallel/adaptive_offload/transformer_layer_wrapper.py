# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Wrappers for megatron.core.transformer.transformer_layer.TransformerLayer
# to inject fine-grained activation offload group logic and dynamic
# recompute / KEEP decisions from the adaptive memory profiler.

import contextlib
from functools import wraps

from megatron.core import tensor_parallel


def transformer_layer_init_wrapper(original_init):
    """Wrap TransformerLayer.__init__ to set offload flags for attn_norm / mlp_norm."""

    @wraps(original_init)
    def wrapper(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        # Set offload flags based on config — mirrors Megatron transformer_layer.py:383-392
        from megatron.core.transformer.identity_op import IdentityOp

        self.offload_attn_norm = (
            self.config.fine_grained_activation_offloading
            and "attn_norm" in (self.config.offload_modules or [])
            and not isinstance(self.input_layernorm, IdentityOp)
        )
        self.offload_mlp_norm = (
            self.config.fine_grained_activation_offloading
            and "mlp_norm" in (self.config.offload_modules or [])
            and not isinstance(self.pre_mlp_layernorm, IdentityOp)
        )

    return wrapper


def forward_attention_wrapper(original_forward_attention):
    """Wrap TransformerLayer._forward_attention to inject attn_norm offload group
    and dynamic recompute/KEEP decisions for attention sub-groups."""

    @wraps(original_forward_attention)
    def wrapper(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
            PipelineOffloadManager,
            fine_grained_offloading_group_commit,
            fine_grained_offloading_group_start,
            get_fine_grained_offloading_context,
        )
        from mindspeed.core.pipeline_parallel.adaptive_offload.adaptive_memory_profiler import (
            AdaptiveMemoryProfiler,
        )

        # Handle deprecated inference_params
        from megatron.core.transformer.utils import deprecate_inference_params
        inference_context = deprecate_inference_params(
            inference_context, inference_params
        )

        _profiler = AdaptiveMemoryProfiler.get_instance()
        _is_moe = getattr(self, "_is_moe_layer", False)

        # Determine dynamic recompute/KEEP groups from the optimizer.
        _offload_mgr = PipelineOffloadManager.get_instance()
        _dyn_recompute_groups = (
            _offload_mgr.recompute_groups if _offload_mgr is not None else set()
        )
        _dyn_skip_groups = (
            _offload_mgr.skip_offload_groups if _offload_mgr is not None else set()
        )

        # attn_norm recompute logic
        _recompute_attn_norm = self.recompute_input_layernorm or (
            "attn_norm" in _dyn_recompute_groups
        )
        _dyn_recompute_attn_norm = (not self.recompute_input_layernorm) and (
            "attn_norm" in _dyn_recompute_groups
        )

        # attention sub-groups coupled decision
        _ATTN_SUBGROUPS = {"qkv_linear", "core_attn", "attn_proj"}
        _recompute_attention = bool(_ATTN_SUBGROUPS & _dyn_recompute_groups)
        _keep_attention = bool(_ATTN_SUBGROUPS & _dyn_skip_groups)

        # Residual connection
        residual = hidden_states

        # Determine whether to skip offload for attn_norm
        _skip_attn_norm = _dyn_recompute_attn_norm or ("attn_norm" in _dyn_skip_groups)
        _do_offload_attn_norm = getattr(self, 'offload_attn_norm', False) and not _skip_attn_norm

        if _do_offload_attn_norm:
            hidden_states = fine_grained_offloading_group_start(
                hidden_states, name="attn_norm"
            )

        # Optional Input Layer norm with profiling context
        with (
            _profiler.profile_module(self.layer_number, "attn_norm", _is_moe)
            if _profiler.is_profiling_active()
            else contextlib.nullcontext()
        ):
            if _recompute_attn_norm:
                self.input_layernorm_checkpoint = (
                    tensor_parallel.CheckpointWithoutOutput()
                )
                with get_fine_grained_offloading_context(_do_offload_attn_norm):
                    input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                        self.input_layernorm, hidden_states
                    )
            else:
                with get_fine_grained_offloading_context(_do_offload_attn_norm):
                    input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention with profiling context
        with (
            _profiler.profile_module(self.layer_number, "attention", _is_moe)
            if _profiler.is_profiling_active()
            else contextlib.nullcontext()
        ):
            if _recompute_attention:
                _attn_kwargs = dict(
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )

                def _run_self_attention(hidden):
                    return self.self_attention(hidden, **_attn_kwargs)

                attention_output_with_bias = tensor_parallel.checkpoint(
                    _run_self_attention,
                    False,
                    input_layernorm_output,
                )
            else:
                attention_output_with_bias = self.self_attention(
                    input_layernorm_output,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )

        if _recompute_attn_norm:
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        if _do_offload_attn_norm:
            (hidden_states,) = fine_grained_offloading_group_commit(
                hidden_states, name="attn_norm", forced_released_tensors=[residual]
            )

        # Residual connection
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if (
            isinstance(attention_output_with_bias, dict)
            and "context" in attention_output_with_bias
        ):
            context = attention_output_with_bias["context"]

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        # Residual connection
        residual = hidden_states

        # mlp_norm recompute logic
        _recompute_mlp_norm = self.recompute_pre_mlp_layernorm or (
            "mlp_norm" in _dyn_recompute_groups
        )
        _dyn_recompute_mlp_norm = (not self.recompute_pre_mlp_layernorm) and (
            "mlp_norm" in _dyn_recompute_groups
        )
        _skip_mlp_norm = _dyn_recompute_mlp_norm or ("mlp_norm" in _dyn_skip_groups)
        _do_offload_mlp_norm = getattr(self, 'offload_mlp_norm', False) and not _skip_mlp_norm

        if _do_offload_mlp_norm:
            hidden_states = fine_grained_offloading_group_start(
                hidden_states, name="mlp_norm"
            )

        with (
            _profiler.profile_module(self.layer_number, "mlp_norm", _is_moe)
            if _profiler.is_profiling_active()
            else contextlib.nullcontext()
        ):
            if _recompute_mlp_norm:
                self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
                with get_fine_grained_offloading_context(_do_offload_mlp_norm):
                    pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                        self.pre_mlp_layernorm, hidden_states
                    )
            else:
                with get_fine_grained_offloading_context(_do_offload_mlp_norm):
                    pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        return pre_mlp_layernorm_output, residual, context

    return wrapper


def forward_mlp_wrapper(original_forward_mlp):
    """Wrap TransformerLayer._forward_mlp to inject mlp_norm group_commit
    and dynamic recompute decisions for MoE sub-groups."""

    @wraps(original_forward_mlp)
    def wrapper(self, pre_mlp_layernorm_output, residual):
        from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
            PipelineOffloadManager,
            fine_grained_offloading_group_commit,
        )
        from mindspeed.core.pipeline_parallel.adaptive_offload.adaptive_memory_profiler import (
            AdaptiveMemoryProfiler,
        )
        from megatron.core.utils import make_viewless_tensor

        _profiler = AdaptiveMemoryProfiler.get_instance()
        _is_moe = getattr(self, "_is_moe_layer", False)

        _offload_mgr = PipelineOffloadManager.get_instance()
        _dyn_recompute_groups = (
            _offload_mgr.recompute_groups if _offload_mgr is not None else set()
        )
        _dyn_skip_groups = (
            _offload_mgr.skip_offload_groups if _offload_mgr is not None else set()
        )

        # mlp_norm discard logic
        _recompute_mlp_norm = self.recompute_pre_mlp_layernorm or (
            "mlp_norm" in _dyn_recompute_groups
        )
        _dyn_recompute_mlp_norm = (not self.recompute_pre_mlp_layernorm) and (
            "mlp_norm" in _dyn_recompute_groups
        )
        _skip_mlp_norm = _dyn_recompute_mlp_norm or ("mlp_norm" in _dyn_skip_groups)
        _do_offload_mlp_norm = getattr(self, 'offload_mlp_norm', False) and not _skip_mlp_norm

        # MoE sub-groups
        _MOE_SUBGROUPS = {"moe_act", "expert_fc1"}
        _dyn_recompute_mlp = bool(_MOE_SUBGROUPS & _dyn_recompute_groups)

        # MLP with profiling context
        with (
            _profiler.profile_module(self.layer_number, "mlp", _is_moe)
            if _profiler.is_profiling_active()
            else contextlib.nullcontext()
        ):
            if self.recompute_mlp or _dyn_recompute_mlp:
                mlp_output_with_bias = tensor_parallel.checkpoint(
                    self.mlp, False, pre_mlp_layernorm_output
                )
            else:
                mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        if _recompute_mlp_norm:
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(
                self.training, self.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, self.hidden_dropout)

        if _do_offload_mlp_norm:
            (hidden_states,) = fine_grained_offloading_group_commit(
                hidden_states, name="mlp_norm", forced_released_tensors=[residual]
            )

        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )

        return output

    return wrapper
