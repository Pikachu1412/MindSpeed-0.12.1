#!/usr/bin/env python3
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Benchmark baseline and streamed projection plus cross entropy on NPU."""

import argparse
import gc
import os
import statistics
import tempfile
import time

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from mindspeed.core.tensor_parallel.streamed_vocab_parallel_cross_entropy import (
    streamed_vocab_parallel_cross_entropy,
    validate_npu_operator_availability,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def run_once(hidden, weight, labels, grad_output, chunk_size):
    hidden = hidden.detach().requires_grad_(True)
    weight = weight.detach().requires_grad_(True)
    if chunk_size is None:
        logits = F.linear(copy_to_tensor_model_parallel_region(hidden), weight)
        loss = vocab_parallel_cross_entropy(logits, labels)
    else:
        loss = streamed_vocab_parallel_cross_entropy(
            hidden, weight, labels.transpose(0, 1).contiguous(), chunk_size
        ).transpose(0, 1)
    (loss * grad_output).sum().backward()
    return loss.detach(), hidden.grad.detach(), weight.grad.detach()


def verify(hidden, weight, labels, grad_output, chunk_sizes):
    reference = run_once(hidden, weight, labels, grad_output, None)
    for chunk_size in chunk_sizes:
        actual = run_once(hidden, weight, labels, grad_output, chunk_size)
        if not torch.allclose(actual[0], reference[0], atol=2e-5, rtol=2e-5):
            raise RuntimeError(f"loss mismatch for chunk size {chunk_size}")
        if not torch.allclose(actual[1], reference[1], atol=2e-2, rtol=2e-2):
            raise RuntimeError(f"hidden gradient mismatch for chunk size {chunk_size}")
        if not torch.allclose(actual[2], reference[2], atol=2e-2, rtol=2e-2):
            raise RuntimeError(f"weight gradient mismatch for chunk size {chunk_size}")


def distributed_max(value, device):
    result = torch.tensor(value, device=device, dtype=torch.float32)
    torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.MAX)
    return result.item()


def measure(hidden, weight, labels, grad_output, chunk_size, warmup, iterations):
    for _ in range(warmup):
        run_once(hidden, weight, labels, grad_output, chunk_size)
    torch.npu.synchronize()
    latencies = []
    peaks = []
    for _ in range(iterations):
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        allocated_before = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        started = time.perf_counter()
        run_once(hidden, weight, labels, grad_output, chunk_size)
        torch.npu.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        peaks.append((
            torch.npu.max_memory_allocated() / (1024 ** 2),
            (torch.npu.max_memory_allocated() - allocated_before) / (1024 ** 2),
        ))
    device = hidden.device
    return (
        distributed_max(statistics.median(latencies), device),
        distributed_max(max(item[0] for item in peaks), device),
        distributed_max(max(item[1] for item in peaks), device),
    )


def initialize_distributed(device):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        torch.distributed.init_process_group(backend="hccl", init_method="env://")
    else:
        rendezvous = tempfile.NamedTemporaryFile(delete=False)
        rendezvous.close()
        os.unlink(rendezvous.name)
        torch.distributed.init_process_group(
            backend="hccl", rank=rank, world_size=world_size,
            init_method=f"file://{rendezvous.name}",
        )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    return rank, world_size


def main():
    args = parse_args()
    device_index = int(os.environ.get("LOCAL_RANK", args.device))
    torch.npu.set_device(device_index)
    validate_npu_operator_availability()
    rank, world_size = initialize_distributed(device_index)
    if args.vocab_size % world_size:
        raise ValueError("vocab size must be divisible by tensor parallel size")

    torch.manual_seed(1234)
    hidden = 0.02 * torch.randn(
        args.tokens, 1, args.hidden_size, device="npu", dtype=torch.bfloat16
    )
    labels = torch.randint(0, args.vocab_size, (args.tokens, 1), device="npu")
    grad_output = torch.randn(args.tokens, 1, device="npu")
    torch.manual_seed(1234 + rank)
    weight = 0.02 * torch.randn(
        args.vocab_size // world_size, args.hidden_size, device="npu", dtype=torch.bfloat16
    )

    verify(hidden, weight, labels, grad_output, args.chunk_sizes)
    torch.npu.synchronize()
    if rank == 0:
        print(f"# tensor_parallel_size={world_size}; correctness=verified")
        print("mode,chunk_size,latency_ms,peak_allocated_mib,incremental_peak_mib")
    latency, peak, incremental = measure(
        hidden, weight, labels, grad_output, None, args.warmup, args.iterations
    )
    if rank == 0:
        print(f"baseline,all,{latency:.3f},{peak:.1f},{incremental:.1f}")
    for chunk_size in args.chunk_sizes:
        latency, peak, incremental = measure(
            hidden, weight, labels, grad_output, chunk_size, args.warmup, args.iterations
        )
        if rank == 0:
            print(f"streamed,{chunk_size},{latency:.3f},{peak:.1f},{incremental:.1f}")
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
