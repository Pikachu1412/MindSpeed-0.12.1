# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser
from functools import partial

from mindspeed.features_manager.feature import MindSpeedFeature


class StreamedVocabParallelCrossEntropyFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("use-streamed-vocab-parallel-cross-entropy")

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--use-streamed-vocab-parallel-cross-entropy", action="store_true",
            help="Compute the vocabulary projection and cross entropy in token chunks."
        )
        group.add_argument(
            "--streamed-vocab-parallel-cross-entropy-chunk-size", type=int, default=4096,
            help="Deprecated compatibility option; the fused operator streams internally."
        )

    def validate_args(self, args):
        if not getattr(args, "use_streamed_vocab_parallel_cross_entropy", False):
            return
        if args.streamed_vocab_parallel_cross_entropy_chunk_size <= 0:
            raise AssertionError("streamed vocab-parallel cross entropy chunk size must be positive")
        if not (getattr(args, "fp16", False) or getattr(args, "bf16", False)):
            raise AssertionError(
                "streamed vocab-parallel cross entropy requires --fp16 or --bf16"
            )
        if getattr(args, "deterministic_mode", False) or getattr(args, "npu_deterministic", False):
            raise AssertionError(
                "streamed vocab-parallel cross entropy does not support deterministic mode"
            )
        if getattr(args, "label_smoothing", 0.0) != 0.0:
            raise AssertionError(
                "streamed vocab-parallel cross entropy supports label_smoothing=0 only"
            )
        if getattr(args, "config_logger_dir", ""):
            raise AssertionError(
                "streamed vocab-parallel cross entropy does not support config logging"
            )
        if getattr(args, "unaligned_linear", False):
            raise AssertionError(
                "streamed vocab-parallel cross entropy does not support unaligned linear"
            )
        if getattr(args, "cross_entropy_loss_fusion", False):
            raise AssertionError(
                "streamed vocab-parallel cross entropy and cross-entropy fusion are incompatible"
            )
        if getattr(args, "gradient_accumulation_fusion", False):
            raise AssertionError(
                "streamed vocab-parallel cross entropy does not support gradient accumulation fusion"
            )
        if getattr(args, "defer_embedding_wgrad_compute", False):
            raise AssertionError(
                "streamed vocab-parallel cross entropy does not support deferred embedding weight gradients"
            )
        if getattr(args, "mtp_num_layers", None):
            raise AssertionError("streamed vocab-parallel cross entropy does not support MTP")

    def register_patches(self, patch_manager, args):
        from mindspeed.core.tensor_parallel.streamed_vocab_parallel_cross_entropy import (
            compute_language_model_loss_wrapper,
            gpt_forward_wrapper,
            output_layer_forward_wrapper,
            validate_npu_operator_availability,
        )
        validate_npu_operator_availability()
        wrapper = partial(
            gpt_forward_wrapper,
            chunk_size=args.streamed_vocab_parallel_cross_entropy_chunk_size,
        )
        wrapper.__name__ = "streamed_vocab_parallel_cross_entropy_wrapper"
        patch_manager.register_patch(
            "megatron.core.models.gpt.gpt_model.GPTModel.forward", wrapper
        )
        patch_manager.register_patch(
            "megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward",
            output_layer_forward_wrapper,
        )
        patch_manager.register_patch(
            "megatron.core.models.common.language_module.language_module."
            "LanguageModule.compute_language_model_loss",
            compute_language_model_loss_wrapper,
        )
