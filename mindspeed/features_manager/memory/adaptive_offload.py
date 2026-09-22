import os
from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class AdaptiveOffloadFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('fine-grained-activation-offloading', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='Fine-grained activation offloading')
        group.add_argument('--auto-activation-memory', action='store_true',
                           default=os.environ.get('AUTO_ACTIVATION_MEMORY') == '1',
                           help='Automatically profile and choose per-module KEEP/OFFLOAD/RECOMPUTE under a device-memory limit.')
        group.add_argument(
            '--fine-grained-activation-offloading',
            action='store_true',
            default=False,
            help='Enable fine-grained activation offloading to CPU.',
        )
        group.add_argument(
            '--offload-modules',
            nargs='*',
            type=str,
            default=[],
            help='The submodules to offload. '
                 'Choices: "attn_norm", "qkv_linear", "core_attn", '
                 '"attn_proj", "mlp_norm", "expert_fc1", "moe_act", "dense_mlp".',
        )
        group.add_argument(
            '--min-offloaded-tensor-size',
            type=int,
            default=1024 * 1024,
            help='Minimum tensor size (in elements) to offload.',
        )

    def is_need_apply(self, args):
        from mindspeed.core.pipeline_parallel.adaptive_offload.auto_activation_memory import configure
        configure(args)
        return super().is_need_apply(args)

    def pre_validate_args(self, args):
        from mindspeed.core.pipeline_parallel.adaptive_offload.auto_activation_memory import configure
        configure(args)

    def pre_register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace):
        """Patch TransformerConfig with default attributes so that any
        `self.config.fine_grained_activation_offloading` access works even
        before the core_transformer_config_from_args wrapper sets them."""
        if not getattr(args, 'fine_grained_activation_offloading', False):
            return

        from megatron.core.transformer.transformer_config import TransformerConfig
        if not hasattr(TransformerConfig, 'fine_grained_activation_offloading'):
            TransformerConfig.fine_grained_activation_offloading = False
        if not hasattr(TransformerConfig, 'offload_modules'):
            TransformerConfig.offload_modules = None
        if not hasattr(TransformerConfig, 'min_offloaded_tensor_size'):
            TransformerConfig.min_offloaded_tensor_size = 1024 * 1024

    def register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace):
        if not getattr(args, 'fine_grained_activation_offloading', False):
            return

        # --- Args -> Config bridge: propagate offload fields onto TransformerConfig ---
        from mindspeed.core.pipeline_parallel.adaptive_offload.gpt_model_wrapper import (
            core_transformer_config_from_args_wrapper,
            gpt_model_forward_wrapper,
        )
        patch_manager.register_patch(
            'megatron.training.arguments.core_transformer_config_from_args',
            core_transformer_config_from_args_wrapper,
        )

        # --- GPTModel.forward wrapper: init chunk handler before forward ---
        patch_manager.register_patch(
            'megatron.core.models.gpt.gpt_model.GPTModel.forward',
            gpt_model_forward_wrapper,
        )

        # --- Schedules wrappers: inject fine_grained_offloading_reset ---
        from mindspeed.core.pipeline_parallel.adaptive_offload.schedules_wrapper import (
            forward_backward_no_pipelining_wrapper,
            forward_backward_pipelining_with_interleaving_wrapper,
            forward_backward_pipelining_without_interleaving_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining',
            forward_backward_no_pipelining_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving',
            forward_backward_pipelining_with_interleaving_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
            forward_backward_pipelining_without_interleaving_wrapper,
        )

        # --- Training wrapper: inject adaptive profiler integration ---
        from mindspeed.core.pipeline_parallel.adaptive_offload.training_wrapper import (
            train_wrapper,
        )
        patch_manager.register_patch(
            'megatron.training.training.train',
            train_wrapper,
        )

        if getattr(args, 'auto_activation_memory', False):
            from mindspeed.core.pipeline_parallel.adaptive_offload.auto_activation_memory import (
                transformer_layer_init_wrapper as auto_layer_init, optimizer_capacity_preflight_wrapper,
            )
            from mindspeed.core.pipeline_parallel.adaptive_offload.transformer_block_wrapper import (
                transformer_block_init_wrapper, transformer_layer_forward_wrapper,
            )
            patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer.__init__', auto_layer_init)
            patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.__init__', transformer_block_init_wrapper)
            patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer.forward', transformer_layer_forward_wrapper)
            patch_manager.register_patch('megatron.training.training.get_megatron_optimizer', optimizer_capacity_preflight_wrapper)
            from mindspeed.core.pipeline_parallel.adaptive_offload.activation_transfer_scheduler import (
                group_backward_begin_wrapper, group_backward_complete_wrapper, legacy_prefetch_wrapper,
            )
            handler = 'mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload.ChunkOffloadHandler'
            patch_manager.register_patch(handler + '.on_group_commit_backward', group_backward_begin_wrapper)
            patch_manager.register_patch(handler + '.on_group_start_backward', group_backward_complete_wrapper)
            patch_manager.register_patch(handler + '.bulk_reload_next', legacy_prefetch_wrapper)
            patch_manager.register_patch(handler + '.pre_reload_last_layer', legacy_prefetch_wrapper)
            return

        # --- TransformerLayer wrappers: inject offload group logic ---
        from mindspeed.core.pipeline_parallel.adaptive_offload.transformer_layer_wrapper import (
            transformer_layer_init_wrapper,
            forward_attention_wrapper,
            forward_mlp_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.transformer_layer.TransformerLayer.__init__',
            transformer_layer_init_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention',
            forward_attention_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp',
            forward_mlp_wrapper,
        )

        # --- TransformerBlock / TransformerLayer wrappers: set is_last_layer ---
        # Required for LAST_LAYER_NO_OFFLOAD to actually take effect.  Without
        # these, the chunk handler's ``is_last_layer`` flag is never set True.
        from mindspeed.core.pipeline_parallel.adaptive_offload.transformer_block_wrapper import (
            transformer_block_init_wrapper,
            transformer_layer_forward_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.transformer_block.TransformerBlock.__init__',
            transformer_block_init_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.transformer_layer.TransformerLayer.forward',
            transformer_layer_forward_wrapper,
        )

        # --- Attention wrappers: inject qkv/core_attn/attn_proj offload groups ---
        from mindspeed.core.pipeline_parallel.adaptive_offload.attention_wrapper import (
            attention_init_wrapper,
            attention_forward_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.attention.Attention.__init__',
            attention_init_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.attention.Attention.forward',
            attention_forward_wrapper,
        )

        from mindspeed.core.pipeline_parallel.adaptive_offload.dense_mlp_wrapper import (
            dense_mlp_forward_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.mlp.MLP.forward',
            dense_mlp_forward_wrapper,
        )

        # --- Experts wrappers: inject expert_fc1/moe_act offload groups ---
        from mindspeed.core.pipeline_parallel.adaptive_offload.experts_wrapper import (
            te_grouped_mlp_init_wrapper,
            te_grouped_mlp_forward_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.moe.experts.TEGroupedMLP.__init__',
            te_grouped_mlp_init_wrapper,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.moe.experts.TEGroupedMLP.forward',
            te_grouped_mlp_forward_wrapper,
        )

    def validate_args(self, args):
        if getattr(args, 'fine_grained_activation_offloading', False):
            if not getattr(args, 'transformer_impl', None) == 'transformer_engine':
                raise AssertionError(
                    'Fine-grained activation offloading requires --transformer-impl transformer_engine'
                )
            if 'dense_mlp' in (getattr(args, 'offload_modules', None) or []):
                if os.environ.get('MEGATRON_ADAPTIVE_OFFLOAD', '0') == '1':
                    raise ValueError('dense_mlp offload currently requires MEGATRON_ADAPTIVE_OFFLOAD=0')
                if getattr(args, 'recompute_granularity', None) is not None or getattr(args, 'recompute_activation_function', False):
                    raise ValueError('dense_mlp offload does not yet support activation checkpointing')
