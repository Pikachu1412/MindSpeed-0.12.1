# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Wrappers for megatron.core.transformer.attention.Attention
# to inject qkv_linear / core_attn / attn_proj fine-grained offload groups
# and dynamic recompute / KEEP decisions from the adaptive memory profiler.

from functools import wraps


def attention_init_wrapper(original_init):
    """Wrap Attention.__init__ to set offload flags for qkv_linear / core_attn / attn_proj."""

    @wraps(original_init)
    def wrapper(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        # Set offload flags — mirrors Megatron attention.py:148-161
        self.offload_qkv_linear = (
            self.config.fine_grained_activation_offloading
            and "qkv_linear" in (self.config.offload_modules or [])
        )
        self.offload_core_attention = (
            self.config.fine_grained_activation_offloading
            and "core_attn" in (self.config.offload_modules or [])
        )
        self.offload_attn_proj = (
            self.config.fine_grained_activation_offloading
            and "attn_proj" in (self.config.offload_modules or [])
        )

    return wrapper


def attention_forward_wrapper(original_forward):
    """Wrap Attention.forward to inject qkv_linear / core_attn / attn_proj
    offload group_start / group_commit calls with dynamic skip logic."""

    @wraps(original_forward)
    def wrapper(self, *args, **kwargs):
        from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
            PipelineOffloadManager,
            fine_grained_offloading_group_commit,
            fine_grained_offloading_group_start,
            get_fine_grained_offloading_context,
        )

        # Check if offload flags exist (set by attention_init_wrapper).
        _has_offload = (
            getattr(self, 'offload_qkv_linear', False)
            or getattr(self, 'offload_core_attention', False)
            or getattr(self, 'offload_attn_proj', False)
        )

        if not _has_offload:
            # No offload configured — call original directly.
            return original_forward(self, *args, **kwargs)

        # Dynamically check recompute/KEEP decisions.
        _offload_mgr = PipelineOffloadManager.get_instance()
        _dyn_recompute_groups = (
            _offload_mgr.recompute_groups if _offload_mgr is not None else set()
        )
        _dyn_skip_groups = (
            _offload_mgr.skip_offload_groups if _offload_mgr is not None else set()
        )
        _ATTN_SUBGROUPS = {"qkv_linear", "core_attn", "attn_proj"}
        _recompute_attention = bool(_ATTN_SUBGROUPS & _dyn_recompute_groups)
        _keep_attention = all(
            not getattr(self, flag, False) or group in _dyn_skip_groups
            for group, flag in (("qkv_linear", "offload_qkv_linear"),
                                ("core_attn", "offload_core_attention"),
                                ("attn_proj", "offload_attn_proj"))
        )
        _skip_offload = _recompute_attention or _keep_attention

        if _skip_offload:
            # All attention offload groups skipped — call original directly.
            return original_forward(self, *args, **kwargs)

        # Determine per-group offload decisions.
        _do_offload_qkv_linear = getattr(self, 'offload_qkv_linear', False) and "qkv_linear" not in _dyn_skip_groups
        _do_offload_core_attn = getattr(self, 'offload_core_attention', False) and "core_attn" not in _dyn_skip_groups
        _do_offload_attn_proj = getattr(self, 'offload_attn_proj', False) and "attn_proj" not in _dyn_skip_groups

        # We need to intercept the forward to insert group_start/commit around
        # the three computation phases. Since the original forward has complex
        # logic (inference paths, flash decode, RoPE, etc.), we use a strategy
        # of wrapping the key methods that produce the offload-worthy tensors.
        #
        # However, the Megatron forward is monolithic — the offload calls are
        # inlined. For MindSpeed, we replicate the full forward with offload
        # calls injected, matching Megatron attention.py:588-800.

        # Extract args in the same order as Attention.forward signature
        hidden_states = args[0] if args else kwargs.get('hidden_states')
        attention_mask = kwargs.get('attention_mask', args[1] if len(args) > 1 else None)

        # Parse remaining kwargs
        key_value_states = kwargs.get('key_value_states', None)
        inference_context = kwargs.get('inference_context', None)
        rotary_pos_emb = kwargs.get('rotary_pos_emb', None)
        rotary_pos_cos = kwargs.get('rotary_pos_cos', None)
        rotary_pos_sin = kwargs.get('rotary_pos_sin', None)
        attention_bias = kwargs.get('attention_bias', None)
        packed_seq_params = kwargs.get('packed_seq_params', None)
        sequence_len_offset = kwargs.get('sequence_len_offset', None)

        # Handle deprecated inference_params
        inference_params = kwargs.get('inference_params', None)
        if inference_params is not None and inference_context is None:
            inference_context = inference_params

        # Handle rotary_pos_emb conversion
        if rotary_pos_cos is not None and rotary_pos_sin is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value — with qkv_linear offload group
        # =====================
        if _do_offload_qkv_linear:
            hidden_states = fine_grained_offloading_group_start(
                hidden_states, name="qkv_linear"
            )
        with get_fine_grained_offloading_context(_do_offload_qkv_linear):
            query, key, value = self.get_query_key_value_tensors(
                hidden_states, key_value_states
            )
        if _do_offload_qkv_linear:
            query, key, value = fine_grained_offloading_group_commit(
                query, key, value, name="qkv_linear", forced_released_tensors=[]
            )

        # Flash decode early return path
        if (
            self.config.flash_decode
            and inference_context is not None
            and inference_context.is_decode_only()
            and not self.training
            and rotary_pos_cos is not None
        ):
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = (
                inference_context.key_value_memory_dict[self.layer_number]
            )
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        # Adjust key/value for inference
        query, key, value, rotary_pos_emb, attn_mask_type = (
            self._adjust_key_value_for_inference(
                inference_context,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            )
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # Rotary positional embedding
        if rotary_pos_emb is not None and not self.config.flash_decode:
            from megatron.core.models.common.embeddings.rotary_pos_embedding import (
                apply_rotary_pos_emb,
            )
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = (
                    packed_seq_params.cu_seqlens_q_padded
                    if packed_seq_params.cu_seqlens_q_padded is not None
                    else packed_seq_params.cu_seqlens_q
                )
                cu_seqlens_kv = (
                    packed_seq_params.cu_seqlens_kv_padded
                    if packed_seq_params.cu_seqlens_kv_padded is not None
                    else packed_seq_params.cu_seqlens_kv
                )
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb(
                        query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(
                        query, q_pos_emb, self.config, cu_seqlens_q
                    )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv
                )

        # ==================================
        # Core attention — with core_attn offload group
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if _do_offload_core_attn and self.training:
                query = fine_grained_offloading_group_start(query, name="core_attn")
            if inference_context is None or inference_context.is_static_batching():
                with get_fine_grained_offloading_context(_do_offload_core_attn):
                    core_attn_out = self.core_attention(
                        query, key, value, attention_mask,
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params,
                    )
            else:
                from einops import rearrange
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()
                core_attn_out = self.flash_decode_and_prefill(
                    q, k, v, max_seqlen_q, max_seqlen_k,
                    cu_query_lengths, cu_kv_lengths,
                )
                core_attn_out = core_attn_out.squeeze(0).unsqueeze(1)
                core_attn_out = rearrange(core_attn_out, "s b h d -> s b (h d)")
            if _do_offload_core_attn and self.training:
                (core_attn_out,) = fine_grained_offloading_group_commit(
                    core_attn_out, name="core_attn",
                    forced_released_tensors=[] if "qkv_linear" in _dyn_skip_groups else [query, key, value],
                )

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output — with attn_proj offload group
        # =================
        if _do_offload_attn_proj:
            core_attn_out = fine_grained_offloading_group_start(
                core_attn_out, name="attn_proj"
            )
        with get_fine_grained_offloading_context(_do_offload_attn_proj):
            output, bias = self.linear_proj(core_attn_out)
        if _do_offload_attn_proj:
            if bias is not None:
                output, bias = fine_grained_offloading_group_commit(
                    output, bias, name="attn_proj",
                    forced_released_tensors=[] if "core_attn" in _dyn_skip_groups else [core_attn_out],
                )
            else:
                (output,) = fine_grained_offloading_group_commit(
                    output, name="attn_proj",
                    forced_released_tensors=[] if "core_attn" in _dyn_skip_groups else [core_attn_out],
                )

        return output, bias

    return wrapper
