# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Wrappers for megatron.core.transformer.moe.experts.TEGroupedMLP
# to inject expert_fc1 / moe_act fine-grained offload groups
# and dynamic recompute / KEEP decisions from the adaptive memory profiler.

from functools import wraps

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel


def te_grouped_mlp_init_wrapper(original_init):
    """Wrap TEGroupedMLP.__init__ to set offload flags and configure save_original_input."""

    @wraps(original_init)
    def wrapper(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        # Set offload flags — mirrors Megatron experts.py:782-801
        self.offload_expert_fc1 = (
            self.config.fine_grained_activation_offloading
            and "expert_fc1" in (self.config.offload_modules or [])
        )
        self.offload_moe_act = (
            self.config.fine_grained_activation_offloading
            and "moe_act" in (self.config.offload_modules or [])
        )

        # When offloading expert_fc1, set save_original_input so that linear_fc1
        # saves the original input tensor (not split views) via save_for_backward.
        if self.offload_expert_fc1 and not self.config.fp8:
            if hasattr(self.linear_fc1, 'save_original_input'):
                self.linear_fc1.save_original_input = True

    return wrapper


def te_grouped_mlp_forward_wrapper(original_forward):
    """Wrap TEGroupedMLP.forward to inject expert_fc1 / moe_act
    offload group_start / group_commit calls with dynamic skip logic."""

    @wraps(original_forward)
    def wrapper(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        _has_offload = (
            getattr(self, 'offload_expert_fc1', False)
            or getattr(self, 'offload_moe_act', False)
        )

        if not _has_offload:
            return original_forward(
                self, permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )

        from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
            PipelineOffloadManager,
            fine_grained_offloading_group_commit,
            fine_grained_offloading_group_start,
            get_fine_grained_offloading_context,
        )

        # Dynamically check recompute/KEEP decisions.
        _offload_mgr = PipelineOffloadManager.get_instance()
        _dyn_recompute_groups = (
            _offload_mgr.recompute_groups if _offload_mgr is not None else set()
        )
        _dyn_skip_groups = (
            _offload_mgr.skip_offload_groups if _offload_mgr is not None else set()
        )
        _MOE_SUBGROUPS = {"moe_act", "expert_fc1"}
        _recompute_mlp = bool(_MOE_SUBGROUPS & _dyn_recompute_groups)
        _keep_mlp = bool(_MOE_SUBGROUPS & _dyn_skip_groups)
        _skip_offload = _recompute_mlp or _keep_mlp

        if _skip_offload:
            return original_forward(
                self, permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )

        _do_offload_expert_fc1 = getattr(self, 'offload_expert_fc1', False)
        _do_offload_moe_act = getattr(self, 'offload_moe_act', False)

        # --- Replicate the forward logic with offload calls injected ---
        # Mirrors Megatron experts.py:848-993

        tokens_per_expert = tokens_per_expert.tolist()
        if self.config.fp8:
            actual_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.fp8_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.fp8_padding(
                permuted_probs.unsqueeze(-1), actual_tokens_per_expert
            )
        else:
            permuted_probs = permuted_probs.unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert self.config.moe_router_topk == 1, (
                "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            )
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(
                original_dtype
            )
            permuted_probs = torch.ones_like(permuted_probs)

        # expert_fc1 offload group
        if _do_offload_expert_fc1:
            permuted_local_hidden_states = fine_grained_offloading_group_start(
                permuted_local_hidden_states, name="expert_fc1"
            )
        with get_fine_grained_offloading_context(_do_offload_expert_fc1):
            fc1_output, bias_parallel = self.linear_fc1(
                permuted_local_hidden_states, tokens_per_expert
            )
        if _do_offload_expert_fc1:
            if bias_parallel is not None:
                fc1_output, bias_parallel = fine_grained_offloading_group_commit(
                    fc1_output, bias_parallel, name="expert_fc1",
                    forced_released_tensors=[permuted_local_hidden_states],
                )
            else:
                (fc1_output,) = fine_grained_offloading_group_commit(
                    fc1_output, name="expert_fc1",
                    forced_released_tensors=[permuted_local_hidden_states],
                )

        # bias_act_func — same as Megatron experts.py:925-964
        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
            if self.config.bias_activation_fusion:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    from megatron.core.transformer.moe.experts import (
                        weighted_bias_swiglu_impl,
                    )
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of swiglu in TEGroupedMLP.")
            else:
                if bias_parallel is not None:
                    shape = intermediate_parallel.shape
                    intermediate_parallel = torch.cat(
                        [
                            t + b
                            for t, b in zip(
                                torch.split(
                                    intermediate_parallel.view(-1, shape[-1]),
                                    tokens_per_expert,
                                ),
                                bias_parallel,
                            )
                        ]
                    ).view(shape)
                if self.config.gated_linear_unit:
                    def glu(x):
                        x = torch.chunk(x, 2, dim=-1)
                        return self.config.activation_func(x[0]) * x[1]
                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
            return intermediate_parallel

        # moe_act offload group
        if _do_offload_moe_act:
            fc1_output = fine_grained_offloading_group_start(
                fc1_output, name="moe_act"
            )

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with get_fine_grained_offloading_context(_do_offload_moe_act):
                bias_act_output = self.activation_checkpoint.checkpoint(
                    bias_act_func, fc1_output, bias_parallel, permuted_probs
                )
            output, output_bias = self.linear_fc2(bias_act_output, tokens_per_expert)
            self.activation_checkpoint.discard_output_and_register_recompute(output)
        else:
            with get_fine_grained_offloading_context(_do_offload_moe_act):
                bias_act_output = bias_act_func(
                    fc1_output, bias_parallel, permuted_probs
                )
            output, output_bias = self.linear_fc2(bias_act_output, tokens_per_expert)

        if _do_offload_moe_act:
            (output,) = fine_grained_offloading_group_commit(
                output, name="moe_act", forced_released_tensors=[fc1_output]
            )

        # FP8 unpadding
        if self.config.fp8:
            output = self.fp8_unpadding(output, actual_tokens_per_expert)

        return output, output_bias

    return wrapper
