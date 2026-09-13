# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from mindspeed import megatron_adaptor
from mindspeed.core.tensor_parallel import streamed_vocab_parallel_cross_entropy as streamed_ce
from mindspeed.core.tensor_parallel.streamed_vocab_parallel_cross_entropy import (
    streamed_vocab_parallel_cross_entropy,
)
from mindspeed.features_manager.tensor_parallel.streamed_vocab_parallel_cross_entropy import (
    StreamedVocabParallelCrossEntropyFeature,
)
from mindspeed.core.tensor_parallel.streamed_vocab_parallel_cross_entropy import (
    _STREAMED_LOSS_CONTEXT,
    _StreamedLossContext,
    _StreamedLossSentinel,
    compute_language_model_loss_wrapper,
    gpt_forward_wrapper,
    output_layer_forward_wrapper,
    validate_npu_operator_availability,
)
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer import TransformerConfig
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)
from tests_extend.commons import initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest


class TestStreamedVocabParallelCrossEntropy(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("chunk_size", [1, 7, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_vocab_parallel_cross_entropy(self, chunk_size, dtype):
        initialize_model_parallel(2, 1)
        torch.manual_seed(1234)
        sequence_length, batch_size, hidden_size = 9, 3, 32
        vocab_size = 256
        labels = torch.randint(0, vocab_size, (batch_size, sequence_length), device="npu")
        hidden = 0.1 * torch.randn(
            sequence_length, batch_size, hidden_size, device="npu", dtype=dtype
        )
        weight = 0.1 * torch.randn(
            vocab_size // 2, hidden_size, device="npu", dtype=dtype
        )
        grad_output = torch.randn(batch_size, sequence_length, device="npu")

        streamed_hidden = hidden.detach().clone().requires_grad_()
        streamed_weight = weight.detach().clone().requires_grad_()
        streamed_loss = streamed_vocab_parallel_cross_entropy(
            streamed_hidden, streamed_weight, labels, chunk_size
        )
        (streamed_loss * grad_output).sum().backward()

        reference_hidden = hidden.detach().clone().requires_grad_()
        reference_weight = weight.detach().clone().requires_grad_()
        logits = F.linear(
            copy_to_tensor_model_parallel_region(reference_hidden), reference_weight
        )
        reference_loss = vocab_parallel_cross_entropy(
            logits, labels.transpose(0, 1).contiguous()
        ).transpose(0, 1).contiguous()
        (reference_loss * grad_output).sum().backward()

        assert torch.allclose(streamed_loss, reference_loss, atol=2e-5, rtol=2e-5)
        tolerance = 2e-3 if dtype == torch.float16 else 2e-2
        assert torch.allclose(
            streamed_hidden.grad, reference_hidden.grad, atol=tolerance, rtol=tolerance
        )
        assert torch.allclose(
            streamed_weight.grad, reference_weight.grad, atol=tolerance, rtol=tolerance
        )
        parallel_state.destroy_model_parallel()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_tiled_backward_matches_reference(self, dtype):
        initialize_model_parallel(2, 1)
        torch.manual_seed(2031)
        sequence_length, batch_size, hidden_size = 9, 3, 32
        vocab_size = 16384
        labels = torch.randint(0, vocab_size, (batch_size, sequence_length), device="npu")
        hidden = 0.1 * torch.randn(
            sequence_length, batch_size, hidden_size, device="npu", dtype=dtype
        )
        weight = 0.1 * torch.randn(
            vocab_size // 2, hidden_size, device="npu", dtype=dtype
        )
        grad_output = torch.randn(batch_size, sequence_length, device="npu")

        original_threshold = streamed_ce._TILED_BACKWARD_MIN_TOKENS
        streamed_ce._TILED_BACKWARD_MIN_TOKENS = 1
        try:
            streamed_hidden = hidden.detach().clone().requires_grad_()
            streamed_weight = weight.detach().clone().requires_grad_()
            streamed_loss = streamed_vocab_parallel_cross_entropy(
                streamed_hidden, streamed_weight, labels, 7
            )
            (streamed_loss * grad_output).sum().backward()
        finally:
            streamed_ce._TILED_BACKWARD_MIN_TOKENS = original_threshold

        reference_hidden = hidden.detach().clone().requires_grad_()
        reference_weight = weight.detach().clone().requires_grad_()
        logits = F.linear(
            copy_to_tensor_model_parallel_region(reference_hidden), reference_weight
        )
        reference_loss = vocab_parallel_cross_entropy(
            logits, labels.transpose(0, 1).contiguous()
        ).transpose(0, 1).contiguous()
        (reference_loss * grad_output).sum().backward()

        tolerance = 2e-3 if dtype == torch.float16 else 2e-2
        assert torch.allclose(streamed_loss, reference_loss, atol=2e-5, rtol=2e-5)
        assert torch.allclose(
            streamed_hidden.grad, reference_hidden.grad, atol=tolerance, rtol=tolerance
        )
        assert torch.allclose(
            streamed_weight.grad, reference_weight.grad, atol=tolerance, rtol=tolerance
        )
        parallel_state.destroy_model_parallel()

    def test_backward_reuses_forward_statistics(self):
        initialize_model_parallel(2, 1)
        torch.manual_seed(2027)
        sequence_length, batch_size, hidden_size = 9, 3, 32
        vocab_size = 256
        labels = torch.randint(0, vocab_size, (batch_size, sequence_length), device="npu")
        hidden = 0.1 * torch.randn(
            sequence_length, batch_size, hidden_size,
            device="npu", dtype=torch.bfloat16, requires_grad=True
        )
        weight = 0.1 * torch.randn(
            vocab_size // 2, hidden_size,
            device="npu", dtype=torch.bfloat16, requires_grad=True
        )
        grad_output = torch.randn(batch_size, sequence_length, device="npu")

        original_batched_statistics = streamed_ce._batched_statistics
        call_count = 0

        def counted_batched_statistics(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_batched_statistics(*args, **kwargs)

        streamed_ce._batched_statistics = counted_batched_statistics
        try:
            loss = streamed_vocab_parallel_cross_entropy(hidden, weight, labels, 7)
            assert call_count == 1
            (loss * grad_output).sum().backward()
            assert call_count == 1
        finally:
            streamed_ce._batched_statistics = original_batched_statistics
            parallel_state.destroy_model_parallel()

    def test_aligned_chunks_use_one_forward_kernel(self):
        initialize_model_parallel(2, 1)
        torch.manual_seed(2029)
        sequence_length, batch_size, hidden_size = 9, 3, 32
        vocab_size = 256
        labels = torch.randint(0, vocab_size, (batch_size, sequence_length), device="npu")
        hidden = 0.1 * torch.randn(
            sequence_length, batch_size, hidden_size,
            device="npu", dtype=torch.bfloat16, requires_grad=True
        )
        weight = 0.1 * torch.randn(
            vocab_size // 2, hidden_size,
            device="npu", dtype=torch.bfloat16, requires_grad=True
        )

        original_forward = streamed_ce.torch_npu.fused_linear_online_max_sum
        call_count = 0

        def counted_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_forward(*args, **kwargs)

        streamed_ce.torch_npu.fused_linear_online_max_sum = counted_forward
        try:
            streamed_vocab_parallel_cross_entropy(hidden, weight, labels, 8)
            assert call_count == 1
        finally:
            streamed_ce.torch_npu.fused_linear_online_max_sum = original_forward
            parallel_state.destroy_model_parallel()

    def test_small_batches_use_one_fused_backward_kernel(self):
        initialize_model_parallel(2, 1)
        torch.manual_seed(2028)
        sequence_length, batch_size, hidden_size = 9, 3, 32
        vocab_size = 256
        labels = torch.randint(0, vocab_size, (batch_size, sequence_length), device="npu")
        hidden = 0.1 * torch.randn(
            sequence_length, batch_size, hidden_size,
            device="npu", dtype=torch.bfloat16, requires_grad=True
        )
        weight = 0.1 * torch.randn(
            vocab_size // 2, hidden_size,
            device="npu", dtype=torch.bfloat16, requires_grad=True
        )
        grad_output = torch.randn(batch_size, sequence_length, device="npu")

        original_backward = streamed_ce.torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad
        call_count = 0

        def counted_backward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_backward(*args, **kwargs)

        streamed_ce.torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad = counted_backward
        try:
            loss = streamed_vocab_parallel_cross_entropy(hidden, weight, labels, 8)
            (loss * grad_output).sum().backward()
            assert call_count == 1
        finally:
            streamed_ce.torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad = original_backward
            parallel_state.destroy_model_parallel()

    def test_large_batches_use_tiled_backward(self):
        initialize_model_parallel(2, 1)
        torch.manual_seed(2030)
        token_count, hidden_size = streamed_ce._TILED_BACKWARD_MIN_TOKENS, 32
        vocab_size = 256
        labels = torch.randint(0, vocab_size, (1, token_count), device="npu")
        hidden = 0.1 * torch.randn(
            token_count, 1, hidden_size,
            device="npu", dtype=torch.bfloat16, requires_grad=True
        )
        weight = 0.1 * torch.randn(
            vocab_size // 2, hidden_size,
            device="npu", dtype=torch.bfloat16, requires_grad=True
        )

        original_backward = streamed_ce.torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad
        original_min_vocab = streamed_ce._TILED_BACKWARD_MIN_LOCAL_VOCAB
        streamed_ce._TILED_BACKWARD_MIN_LOCAL_VOCAB = 1
        call_count = 0

        def counted_backward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_backward(*args, **kwargs)

        streamed_ce.torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad = counted_backward
        try:
            loss = streamed_vocab_parallel_cross_entropy(hidden, weight, labels, 8)
            loss.sum().backward()
            assert call_count == 0
        finally:
            streamed_ce.torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad = original_backward
            streamed_ce._TILED_BACKWARD_MIN_LOCAL_VOCAB = original_min_vocab
            parallel_state.destroy_model_parallel()

    def test_sequence_parallel_matches_reference(self):
        initialize_model_parallel(2, 1)
        torch.manual_seed(4321)
        local_sequence_length, batch_size, hidden_size = 5, 2, 32
        vocab_size = 256
        hidden = 0.1 * torch.randn(
            local_sequence_length, batch_size, hidden_size,
            device="npu", dtype=torch.bfloat16
        )
        labels = torch.randint(
            0, vocab_size, (batch_size, local_sequence_length * 2), device="npu"
        )
        weight = 0.1 * torch.randn(
            vocab_size // 2, hidden_size, device="npu", dtype=torch.bfloat16
        )
        grad_output = torch.randn_like(labels, dtype=torch.float32)

        streamed_hidden = hidden.detach().clone().requires_grad_()
        streamed_weight = weight.detach().clone().requires_grad_()
        streamed_loss = streamed_vocab_parallel_cross_entropy(
            streamed_hidden, streamed_weight, labels, 3, sequence_parallel=True
        )
        (streamed_loss * grad_output).sum().backward()

        reference_hidden = hidden.detach().clone().requires_grad_()
        reference_weight = weight.detach().clone().requires_grad_()
        gathered_hidden = gather_from_sequence_parallel_region(
            reference_hidden, tensor_parallel_output_grad=True
        )
        logits = F.linear(gathered_hidden, reference_weight)
        reference_loss = vocab_parallel_cross_entropy(
            logits, labels.transpose(0, 1).contiguous()
        ).transpose(0, 1).contiguous()
        (reference_loss * grad_output).sum().backward()

        assert torch.allclose(streamed_loss, reference_loss, atol=2e-5, rtol=2e-5)
        assert torch.allclose(
            streamed_hidden.grad, reference_hidden.grad, atol=2e-2, rtol=2e-2
        )
        assert torch.allclose(
            streamed_weight.grad, reference_weight.grad, atol=2e-2, rtol=2e-2
        )
        parallel_state.destroy_model_parallel()



class TestStreamedVocabParallelCrossEntropyTp2Cp2(DistributedTest):
    world_size = 4

    def test_tp2_cp2_local_loss_matches_reference(self):
        initialize_model_parallel(2, 1, context_parallel_size=2)
        assert parallel_state.get_tensor_model_parallel_world_size() == 2
        assert parallel_state.get_context_parallel_world_size() == 2
        torch.manual_seed(2026 + parallel_state.get_context_parallel_rank())
        sequence_length, batch_size, hidden_size = 7, 2, 32
        vocab_size = 256
        hidden = 0.1 * torch.randn(
            sequence_length, batch_size, hidden_size,
            device="npu", dtype=torch.bfloat16
        )
        labels = torch.randint(0, vocab_size, (batch_size, sequence_length), device="npu")
        weight = 0.1 * torch.randn(
            vocab_size // 2, hidden_size, device="npu", dtype=torch.bfloat16
        )
        grad_output = torch.randn(batch_size, sequence_length, device="npu")

        streamed_hidden = hidden.detach().clone().requires_grad_()
        streamed_weight = weight.detach().clone().requires_grad_()
        streamed_loss = streamed_vocab_parallel_cross_entropy(
            streamed_hidden, streamed_weight, labels, 3
        )
        (streamed_loss * grad_output).sum().backward()

        reference_hidden = hidden.detach().clone().requires_grad_()
        reference_weight = weight.detach().clone().requires_grad_()
        logits = F.linear(
            copy_to_tensor_model_parallel_region(reference_hidden), reference_weight
        )
        reference_loss = vocab_parallel_cross_entropy(
            logits, labels.transpose(0, 1).contiguous()
        ).transpose(0, 1).contiguous()
        (reference_loss * grad_output).sum().backward()

        assert torch.allclose(streamed_loss, reference_loss, atol=2e-5, rtol=2e-5)
        assert torch.allclose(
            streamed_hidden.grad, reference_hidden.grad, atol=2e-2, rtol=2e-2
        )
        assert torch.allclose(
            streamed_weight.grad, reference_weight.grad, atol=2e-2, rtol=2e-2
        )
        parallel_state.destroy_model_parallel()


def test_feature_validation():
    feature = StreamedVocabParallelCrossEntropyFeature()
    args = type("Args", (), {
        "use_streamed_vocab_parallel_cross_entropy": True,
        "streamed_vocab_parallel_cross_entropy_chunk_size": 0,
        "fp16": True,
        "bf16": False,
        "deterministic_mode": False,
        "npu_deterministic": False,
        "cross_entropy_loss_fusion": False,
        "gradient_accumulation_fusion": False,
        "defer_embedding_wgrad_compute": False,
        "mtp_num_layers": None,
    })()
    with pytest.raises(AssertionError, match="chunk size must be positive"):
        feature.validate_args(args)


def test_feature_validation_rejects_unaligned_linear():
    feature = StreamedVocabParallelCrossEntropyFeature()
    args = type("Args", (), {
        "use_streamed_vocab_parallel_cross_entropy": True,
        "streamed_vocab_parallel_cross_entropy_chunk_size": 128,
        "fp16": False,
        "bf16": True,
        "deterministic_mode": False,
        "npu_deterministic": False,
        "label_smoothing": 0.0,
        "config_logger_dir": "",
        "unaligned_linear": True,
        "cross_entropy_loss_fusion": False,
        "gradient_accumulation_fusion": False,
        "defer_embedding_wgrad_compute": False,
        "mtp_num_layers": None,
    })()
    with pytest.raises(AssertionError, match="does not support unaligned linear"):
        feature.validate_args(args)


def test_feature_validation_rejects_fp32():
    feature = StreamedVocabParallelCrossEntropyFeature()
    args = type("Args", (), {
        "use_streamed_vocab_parallel_cross_entropy": True,
        "streamed_vocab_parallel_cross_entropy_chunk_size": 128,
        "fp16": False,
        "bf16": False,
        "deterministic_mode": False,
        "npu_deterministic": False,
        "cross_entropy_loss_fusion": False,
        "gradient_accumulation_fusion": False,
        "defer_embedding_wgrad_compute": False,
        "mtp_num_layers": None,
    })()
    with pytest.raises(AssertionError, match="requires --fp16 or --bf16"):
        feature.validate_args(args)


def test_feature_validation_rejects_deterministic_mode():
    feature = StreamedVocabParallelCrossEntropyFeature()
    args = type("Args", (), {
        "use_streamed_vocab_parallel_cross_entropy": True,
        "streamed_vocab_parallel_cross_entropy_chunk_size": 128,
        "fp16": True,
        "bf16": False,
        "deterministic_mode": True,
        "npu_deterministic": False,
        "cross_entropy_loss_fusion": False,
        "gradient_accumulation_fusion": False,
        "defer_embedding_wgrad_compute": False,
        "mtp_num_layers": None,
    })()
    with pytest.raises(AssertionError, match="deterministic mode"):
        feature.validate_args(args)


class _FakeOutputLayer:
    gather_output = False
    bias = None
    sequence_parallel = False
    weight = object()


def test_output_layer_interception_is_scoped():
    layer = _FakeOutputLayer()
    labels = torch.ones(2, 3, dtype=torch.long)
    hidden = torch.ones(3, 2, 4)
    calls = []

    def original(self, input_, weight=None, runtime_gather_output=None):
        calls.append(input_)
        return "normal", None

    wrapped = output_layer_forward_wrapper(original)
    assert wrapped(layer, hidden) == ("normal", None)
    token = _STREAMED_LOSS_CONTEXT.set(_StreamedLossContext(layer, labels, 7))
    try:
        output, bias = wrapped(layer, hidden)
    finally:
        _STREAMED_LOSS_CONTEXT.reset(token)
    assert isinstance(output, _StreamedLossSentinel)
    assert output.hidden_states is hidden
    assert output.labels is labels
    assert output.chunk_size == 7
    assert bias is None
    assert len(calls) == 1


def test_gpt_wrapper_restores_context_without_model_mutation():
    labels = torch.ones(2, 3, dtype=torch.long)
    model = type("Model", (), {
        "post_process": True,
        "training": True,
        "parallel_output": True,
        "mtp_process": False,
        "output_layer": type("Layer", (), {
            "bias": None,
            "gradient_accumulation_fusion": False,
        })(),
        "config": type("Config", (), {"defer_embedding_wgrad_compute": False})(),
    })()

    def original(self, *args, **kwargs):
        context = _STREAMED_LOSS_CONTEXT.get()
        assert context.output_layer is self.output_layer
        assert self.post_process is True
        return "result"

    wrapped = gpt_forward_wrapper(original, 5)
    assert wrapped(model, None, None, None, None, labels) == "result"
    assert model.post_process is True
    assert _STREAMED_LOSS_CONTEXT.get() is None



def test_gpt_wrapper_rejects_config_logger():
    labels = torch.ones(2, 3, dtype=torch.long)
    model = type("Model", (), {
        "post_process": True, "training": True, "parallel_output": True,
        "mtp_process": False,
        "output_layer": type("Layer", (), {
            "bias": None, "gradient_accumulation_fusion": False,
        })(),
        "config": type("Config", (), {
            "config_logger_dir": "/tmp/config-log",
            "defer_embedding_wgrad_compute": False,
        })(),
    })()
    wrapped = gpt_forward_wrapper(lambda *args, **kwargs: None, 5)
    with pytest.raises(RuntimeError, match="config logging"):
        wrapped(model, None, None, None, None, labels)


def test_gpt_wrapper_rejects_labeled_inference():
    model = type("Model", (), {
        "post_process": True,
        "training": False,
    })()
    wrapped = gpt_forward_wrapper(lambda *args, **kwargs: None, 5)
    with pytest.raises(RuntimeError, match="training-only"):
        wrapped(model, None, None, None, None, torch.ones(2, 3, dtype=torch.long))


def test_language_loss_wrapper_delegates_non_sentinel():
    calls = []
    wrapped = compute_language_model_loss_wrapper(
        lambda self, labels, logits: calls.append((labels, logits)) or "normal"
    )
    labels = object()
    logits = object()
    assert wrapped(object(), labels, logits) == "normal"
    assert calls == [(labels, logits)]


def test_required_torch_npu_operators_are_available():
    validate_npu_operator_availability()


def test_output_layer_interception_rejects_bias():
    layer = _FakeOutputLayer()
    layer.bias = object()
    labels = torch.ones(2, 3, dtype=torch.long)
    hidden = torch.ones(3, 2, 4)
    wrapped = output_layer_forward_wrapper(lambda *args, **kwargs: ("normal", None))
    token = _STREAMED_LOSS_CONTEXT.set(_StreamedLossContext(layer, labels, 7))
    try:
        with pytest.raises(RuntimeError, match="does not support output bias"):
            wrapped(layer, hidden)
    finally:
        _STREAMED_LOSS_CONTEXT.reset(token)


def test_feature_validation_rejects_label_smoothing():
    feature = StreamedVocabParallelCrossEntropyFeature()
    args = type("Args", (), {
        "use_streamed_vocab_parallel_cross_entropy": True,
        "streamed_vocab_parallel_cross_entropy_chunk_size": 128,
        "fp16": False, "bf16": True, "deterministic_mode": False,
        "npu_deterministic": False, "label_smoothing": 0.1,
    })()
    with pytest.raises(AssertionError, match="label_smoothing=0 only"):
        feature.validate_args(args)


def test_feature_validation_rejects_config_logger():
    feature = StreamedVocabParallelCrossEntropyFeature()
    args = type("Args", (), {
        "use_streamed_vocab_parallel_cross_entropy": True,
        "streamed_vocab_parallel_cross_entropy_chunk_size": 128,
        "fp16": False, "bf16": True, "deterministic_mode": False,
        "npu_deterministic": False, "label_smoothing": 0.0,
        "config_logger_dir": "/tmp/config-log",
    })()
    with pytest.raises(AssertionError, match="config logging"):
        feature.validate_args(args)


def test_public_api_rejects_label_smoothing():
    with pytest.raises(ValueError, match="label_smoothing=0 only"):
        streamed_vocab_parallel_cross_entropy(
            torch.ones(1, 1, 4), torch.ones(128, 4),
            torch.zeros(1, 1, dtype=torch.long), 1, label_smoothing=0.1
        )


class TestEnabledStreamedVocabParallelCrossEntropy(DistributedTest):
    world_size = 1

    def test_gpt_forward_backward_uses_registered_wrappers(self):
        initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(1234)
        from mindspeed.patch_utils import MindSpeedPatchesManager

        args = type("Args", (), {
            "streamed_vocab_parallel_cross_entropy_chunk_size": 3,
        })()
        feature = StreamedVocabParallelCrossEntropyFeature()
        feature.register_patches(MindSpeedPatchesManager, args)
        MindSpeedPatchesManager.apply_patches()

        global_args = parse_args(None, True)
        global_args.use_flash_attn = True
        global_args.seq_length = 8
        global_args.max_position_embeddings = 8
        global_args.micro_batch_size = 2
        set_args(global_args)
        config = TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4,
            use_cpu_initialization=True, bf16=True,
            gradient_accumulation_fusion=False,
        )
        model = GPTModel(
            config=config, transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=256, max_sequence_length=8, parallel_output=True,
        ).bfloat16().npu()
        model.train()
        input_ids = torch.randint(0, 256, (2, 8), device="npu")
        position_ids = torch.arange(8, device="npu").unsqueeze(0).expand(2, -1)
        labels = torch.randint(0, 256, (2, 8), device="npu")
        attention_mask = torch.triu(
            torch.ones(1, 1, 8, 8, device="npu", dtype=torch.bool), diagonal=1
        )
        loss = model(input_ids, position_ids, attention_mask, labels=labels)
        assert loss.shape == labels.shape
        assert loss.requires_grad
        loss.mean().backward()
        assert model.output_layer.weight.grad is not None
        assert _STREAMED_LOSS_CONTEXT.get() is None
        parallel_state.destroy_model_parallel()
