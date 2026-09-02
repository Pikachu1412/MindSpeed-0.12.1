# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

import torch
import torch_npu

from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import VocabUtility


@dataclass(frozen=True)
class _StreamedLossContext:
    output_layer: object
    labels: torch.Tensor
    chunk_size: int


@dataclass(frozen=True)
class _StreamedLossSentinel:
    hidden_states: torch.Tensor
    weight: torch.Tensor
    labels: torch.Tensor
    chunk_size: int
    sequence_parallel: bool


_STREAMED_LOSS_CONTEXT = ContextVar("streamed_vocab_parallel_loss", default=None)
_REQUIRED_NPU_OPERATORS = {
    "fused_linear_online_max_sum": (
        "Tensor input", "Tensor weight", "Tensor target",
        "int vocab_start_index", "int vocab_end_index",
    ),
    "fused_cross_entropy_loss_with_max_sum": (
        "Tensor logits_max", "Tensor sum_exp_logits", "Tensor predicted_logits",
    ),
    "fused_linear_cross_entropy_loss_with_max_sum_grad": (
        "Tensor grad", "Tensor input", "Tensor weight",
        "Tensor target_mask", "Tensor masked_target",
    ),
}


def validate_npu_operator_availability():
    """Fail early when the installed torch_npu lacks the required public bindings."""
    missing = [
        name for name in _REQUIRED_NPU_OPERATORS
        if not callable(getattr(torch_npu, name, None))
    ]
    if missing:
        raise RuntimeError(
            "streamed vocab-parallel cross entropy requires torch_npu operators: "
            + ", ".join(missing)
        )
    for name, required_arguments in _REQUIRED_NPU_OPERATORS.items():
        try:
            schema = str(getattr(torch.ops.npu, name).default._schema)
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(f"cannot inspect torch_npu operator schema for {name}") from error
        if any(argument not in schema for argument in required_arguments):
            raise RuntimeError(
                f"unsupported torch_npu operator contract for {name}: {schema}"
            )


def _all_reduce(tensor, op):
    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        torch.distributed.all_reduce(
            tensor, op=op, group=parallel_state.get_tensor_model_parallel_group()
        )


def _batched_statistics(input_, weight, target, chunk_size, vocab_start_index, vocab_end_index):
    """Project chunks locally, then batch collectives over their small statistics."""
    local_max_chunks = []
    local_sum_chunks = []
    predicted_chunks = []
    target_mask_chunks = []
    masked_target_chunks = []
    for start in range(0, target.numel(), chunk_size):
        end = min(start + chunk_size, target.numel())
        statistics = torch_npu.fused_linear_online_max_sum(
            input_[start:end], weight, target[start:end],
            vocab_start_index, vocab_end_index, False
        )
        local_max_chunks.append(statistics[0])
        local_sum_chunks.append(statistics[1])
        predicted_chunks.append(statistics[2])
        target_mask_chunks.append(statistics[3])
        masked_target_chunks.append(statistics[4])

    local_logits_max = torch.cat(local_max_chunks)
    global_logits_max = local_logits_max.clone()
    _all_reduce(global_logits_max, torch.distributed.ReduceOp.MAX)

    # The operator returns statistics relative to each rank's local maximum.
    # Rebase both quantities to the global maximum before reducing. The sum and
    # target-logit reductions are packed together to avoid one collective.
    sum_exp_logits = torch.cat(local_sum_chunks) * torch.exp(
        local_logits_max - global_logits_max
    )
    target_in_partition = (target >= vocab_start_index) & (target < vocab_end_index)
    predicted_logits = (
        (torch.cat(predicted_chunks) + local_logits_max - global_logits_max)
        * target_in_partition
    )
    reduced = torch.stack((sum_exp_logits, predicted_logits))
    _all_reduce(reduced, torch.distributed.ReduceOp.SUM)
    return (
        global_logits_max,
        reduced[0],
        reduced[1],
        target_mask_chunks,
        masked_target_chunks,
    )


class _StreamedVocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, target, chunk_size):
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("streamed vocab-parallel cross entropy supports fp16 and bf16 only")
        if weight.size(0) < 128:
            raise ValueError("streamed vocab-parallel cross entropy requires at least 128 local vocabulary entries")

        world_size = parallel_state.get_tensor_model_parallel_world_size()
        rank = parallel_state.get_tensor_model_parallel_rank()
        vocab_start_index, vocab_end_index = (
            VocabUtility.vocab_range_from_per_partition_vocab_size(
                weight.size(0), rank, world_size
            )
        )

        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        flat_target = target.reshape(-1)
        statistics = _batched_statistics(
            flat_hidden, weight, flat_target, chunk_size,
            vocab_start_index, vocab_end_index
        )
        losses, _ = torch_npu.fused_cross_entropy_loss_with_max_sum(
            statistics[0], statistics[1], statistics[2]
        )

        ctx.save_for_backward(hidden_states, weight, target)
        ctx.chunk_size = chunk_size
        ctx.vocab_start_index = vocab_start_index
        ctx.vocab_end_index = vocab_end_index
        return losses.view_as(target)

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight, target = ctx.saved_tensors
        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        flat_target = target.reshape(-1)
        flat_grad_output = grad_output.reshape(-1).float()
        grad_input = torch.empty_like(flat_hidden)
        grad_weight = torch.zeros_like(weight)

        statistics = _batched_statistics(
            flat_hidden, weight, flat_target, ctx.chunk_size,
            ctx.vocab_start_index, ctx.vocab_end_index
        )
        for index, start in enumerate(range(0, flat_target.numel(), ctx.chunk_size)):
            end = min(start + ctx.chunk_size, flat_target.numel())
            chunk_grad_input, chunk_grad_weight = (
                torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad(
                    flat_grad_output[start:end], flat_hidden[start:end], weight,
                    statistics[3][index], statistics[4][index], 0.0,
                    statistics[0][start:end], statistics[1][start:end], None
                )
            )
            grad_input[start:end].copy_(chunk_grad_input)
            grad_weight.add_(chunk_grad_weight)

        return (
            grad_input.view_as(hidden_states),
            grad_weight,
            None,
            None,
        )


def streamed_vocab_parallel_cross_entropy(
    hidden_states, weight, labels, chunk_size, sequence_parallel=False, label_smoothing=0.0
):
    """Compute projection and unsmoothed per-token TP cross entropy without logits.

    Label smoothing is accepted only to make the unsupported scope explicit; the
    private fused backward operator is used exclusively with zero smoothing.
    """
    if label_smoothing != 0.0:
        raise ValueError(
            "streamed vocab-parallel cross entropy supports label_smoothing=0 only"
        )
    if sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=True
        )
    else:
        hidden_states = copy_to_tensor_model_parallel_region(hidden_states)

    target = labels.transpose(0, 1).contiguous()
    if hidden_states.shape[:-1] != target.shape:
        raise RuntimeError(
            f"hidden states and labels have incompatible shapes: {hidden_states.shape} and {labels.shape}"
        )
    loss = _StreamedVocabParallelCrossEntropy.apply(
        hidden_states, weight, target, chunk_size
    )
    return loss.transpose(0, 1).contiguous()



def output_layer_forward_wrapper(fn):
    """Intercept only the active GPT model's output projection."""
    @wraps(fn)
    def wrapper(self, input_, weight=None, runtime_gather_output=None):
        context = _STREAMED_LOSS_CONTEXT.get()
        if context is None or self is not context.output_layer:
            return fn(
                self, input_, weight=weight,
                runtime_gather_output=runtime_gather_output
            )
        if runtime_gather_output:
            raise RuntimeError(
                "streamed vocab-parallel cross entropy requires partitioned output"
            )
        if self.gather_output:
            raise RuntimeError(
                "streamed vocab-parallel cross entropy requires gather_output=False"
            )
        if self.bias is not None:
            raise RuntimeError(
                "streamed vocab-parallel cross entropy does not support output bias"
            )
        output_weight = self.weight if weight is None else weight
        if output_weight is None:
            raise RuntimeError("streamed vocab-parallel cross entropy requires output weight")
        sentinel = _StreamedLossSentinel(
            hidden_states=input_,
            weight=output_weight,
            labels=context.labels,
            chunk_size=context.chunk_size,
            sequence_parallel=self.sequence_parallel,
        )
        return sentinel, None

    return wrapper


def compute_language_model_loss_wrapper(fn):
    """Consume the projection sentinel at Megatron's normal loss boundary."""
    @wraps(fn)
    def wrapper(self, labels, logits):
        if not isinstance(logits, _StreamedLossSentinel):
            return fn(self, labels, logits)
        if labels is not logits.labels:
            raise RuntimeError("streamed loss labels changed before loss computation")
        return streamed_vocab_parallel_cross_entropy(
            logits.hidden_states,
            logits.weight,
            labels,
            logits.chunk_size,
            logits.sequence_parallel,
        )

    return wrapper


def gpt_forward_wrapper(fn, chunk_size):
    """Scope output interception to one labeled GPT call without mutating model state."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        labels = kwargs.get("labels")
        if labels is None and len(args) >= 5:
            labels = args[4]
        if labels is None or not self.post_process:
            return fn(self, *args, **kwargs)
        if not self.training:
            raise RuntimeError(
                "streamed vocab-parallel cross entropy is training-only; "
                "remove labels to run inference"
            )
        runtime_gather_output = kwargs.get("runtime_gather_output")
        if runtime_gather_output is None and len(args) >= 9:
            runtime_gather_output = args[8]
        if runtime_gather_output:
            raise RuntimeError(
                "streamed vocab-parallel cross entropy requires partitioned output"
            )
        if not self.parallel_output:
            raise RuntimeError(
                "streamed vocab-parallel cross entropy requires parallel_output=True"
            )
        if self.output_layer.bias is not None:
            raise RuntimeError(
                "streamed vocab-parallel cross entropy does not support output bias"
            )
        if self.output_layer.gradient_accumulation_fusion:
            raise RuntimeError(
                "streamed vocab-parallel cross entropy does not support gradient accumulation fusion"
            )
        if self.mtp_process:
            raise RuntimeError("streamed vocab-parallel cross entropy does not support MTP")
        if getattr(self.config, "config_logger_dir", ""):
            raise RuntimeError(
                "streamed vocab-parallel cross entropy does not support config logging"
            )
        if self.config.defer_embedding_wgrad_compute:
            raise RuntimeError(
                "streamed vocab-parallel cross entropy does not support deferred embedding weight gradients"
            )

        context = _StreamedLossContext(self.output_layer, labels, chunk_size)
        token = _STREAMED_LOSS_CONTEXT.set(context)
        try:
            return fn(self, *args, **kwargs)
        finally:
            _STREAMED_LOSS_CONTEXT.reset(token)

    return wrapper
