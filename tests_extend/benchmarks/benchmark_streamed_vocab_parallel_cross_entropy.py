#!/usr/bin/env python3
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Benchmark TP=1 baseline and streamed projection plus cross entropy on one NPU."""

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


def run_once(hidden, weight, labels, chunk_size):
    hidden = hidden.detach().requires_grad_(True)
    weight = weight.detach().requires_grad_(True)
    if chunk_size is None:
        logits = F.linear(hidden, weight)
        loss = vocab_parallel_cross_entropy(logits, labels)
    else:
        loss = streamed_vocab_parallel_cross_entropy(
            hidden, weight, labels.transpose(0, 1).contiguous(), chunk_size
        ).transpose(0, 1)
    loss.mean().backward()
    return loss.detach(), hidden.grad.detach(), weight.grad.detach()


def verify(hidden, weight, labels, chunk_sizes):
    reference = run_once(hidden, weight, labels, None)
    for chunk_size in chunk_sizes:
        actual = run_once(hidden, weight, labels, chunk_size)
        if not torch.allclose(actual[0], reference[0], atol=2e-5, rtol=2e-5):
            raise RuntimeError(f"loss mismatch for chunk size {chunk_size}")
        if not torch.allclose(actual[1], reference[1], atol=2e-2, rtol=2e-2):
            raise RuntimeError(f"hidden gradient mismatch for chunk size {chunk_size}")
        if not torch.allclose(actual[2], reference[2], atol=2e-2, rtol=2e-2):
            raise RuntimeError(f"weight gradient mismatch for chunk size {chunk_size}")


def measure(hidden, weight, labels, chunk_size, warmup, iterations):
    for _ in range(warmup):
        run_once(hidden, weight, labels, chunk_size)
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
        run_once(hidden, weight, labels, chunk_size)
        torch.npu.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        peaks.append((
            torch.npu.max_memory_allocated() / (1024 ** 2),
            (torch.npu.max_memory_allocated() - allocated_before) / (1024 ** 2),
        ))
    return statistics.median(latencies), max(item[0] for item in peaks), max(
        item[1] for item in peaks
    )


def main():
    args = parse_args()
    torch.npu.set_device(args.device)
    validate_npu_operator_availability()
    rendezvous = tempfile.NamedTemporaryFile(delete=False)
    rendezvous.close()
    os.unlink(rendezvous.name)
    torch.distributed.init_process_group(
        backend="hccl", rank=0, world_size=1,
        init_method=f"file://{rendezvous.name}",
    )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    torch.manual_seed(1234)
    hidden = 0.02 * torch.randn(
        args.tokens, 1, args.hidden_size, device="npu", dtype=torch.bfloat16
    )
    weight = 0.02 * torch.randn(
        args.vocab_size, args.hidden_size, device="npu", dtype=torch.bfloat16
    )
    labels = torch.randint(0, args.vocab_size, (args.tokens, 1), device="npu")

    verify(hidden, weight, labels, args.chunk_sizes)
    torch.npu.synchronize()
    print("# tensor_parallel_size=1; correctness=verified")
    print("mode,chunk_size,latency_ms,peak_allocated_mib,incremental_peak_mib")
    latency, peak, incremental = measure(
        hidden, weight, labels, None, args.warmup, args.iterations
    )
    print(f"baseline,all,{latency:.3f},{peak:.1f},{incremental:.1f}")
    for chunk_size in args.chunk_sizes:
        latency, peak, incremental = measure(
            hidden, weight, labels, chunk_size, args.warmup, args.iterations
        )
        print(f"streamed,{chunk_size},{latency:.3f},{peak:.1f},{incremental:.1f}")
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
