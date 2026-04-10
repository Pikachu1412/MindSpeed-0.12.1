# Copyright (c) 2025. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# Activation Offloading Profiler for fine-grained performance analysis.

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class ModuleOffloadStats:
    """Statistics for a single module's offload/reload operations."""

    module_name: str
    # Activation tensor info
    tensor_sizes: List[int] = field(default_factory=list)
    tensor_bytes: List[int] = field(default_factory=list)

    # Timing metrics (in milliseconds)
    offload_times: List[float] = field(default_factory=list)  # D2H transfer time
    reload_times: List[float] = field(default_factory=list)   # H2D transfer time

    # Stream synchronization overhead
    d2h_wait_times: List[float] = field(default_factory=list)
    h2d_wait_times: List[float] = field(default_factory=list)
    compute_wait_times: List[float] = field(default_factory=list)

    def add_offload(self, tensor_numel: int, tensor_bytes: int,
                    offload_time: float, wait_time: float = 0.0):
        """Record an offload operation."""
        self.tensor_sizes.append(tensor_numel)
        self.tensor_bytes.append(tensor_bytes)
        self.offload_times.append(offload_time)
        self.d2h_wait_times.append(wait_time)

    def add_reload(self, reload_time: float, wait_time: float = 0.0,
                   compute_wait: float = 0.0):
        """Record a reload operation."""
        self.reload_times.append(reload_time)
        self.h2d_wait_times.append(wait_time)
        self.compute_wait_times.append(compute_wait)

    @property
    def total_bytes(self) -> int:
        return sum(self.tensor_bytes)

    @property
    def avg_offload_time(self) -> float:
        return sum(self.offload_times) / max(len(self.offload_times), 1)

    @property
    def avg_reload_time(self) -> float:
        return sum(self.reload_times) / max(len(self.reload_times), 1)

    @property
    def total_transfer_time(self) -> float:
        return sum(self.offload_times) + sum(self.reload_times)

    @property
    def effective_bandwidth_gbps(self) -> float:
        """Calculate effective PCIe bandwidth in GB/s."""
        total_bytes = self.total_bytes * 2  # Both directions
        total_time_s = self.total_transfer_time / 1000.0
        if total_time_s == 0:
            return 0.0
        return (total_bytes / 1e9) / total_time_s


class OffloadProfiler:
    """
    Global profiler for activation offloading statistics.

    Usage:
        # Enable profiling
        export MEGATRON_OFFLOAD_PROFILING=1

        # In training loop
        profiler = OffloadProfiler.get_instance()
        profiler.start_iteration()
        # ... training step ...
        profiler.end_iteration()

        # Print summary
        profiler.print_summary()
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "OffloadProfiler":
        if cls._instance is None:
            cls._instance = OffloadProfiler()
        return cls._instance

    @classmethod
    def is_enabled(cls) -> bool:
        return os.environ.get('MEGATRON_OFFLOAD_PROFILING', '0') == '1'

    def __init__(self):
        self._enabled = self.is_enabled()
        self._rank = 0
        self._profile_rank = int(os.environ.get('MEGATRON_OFFLOAD_PROFILE_RANK', '0'))
        self._module_stats: Dict[str, ModuleOffloadStats] = {}
        self._iteration_count = 0
        self._iteration_start_time = 0.0
        self._iteration_times: List[float] = []

        # Hardware parameters for analysis
        self._pcie_bandwidth_gbps = float(os.environ.get('PCIE_BANDWIDTH_GBPS', '50'))
        self._gpu_tflops = float(os.environ.get('GPU_TFLOPS', '100'))

        # Per-iteration stats
        self._current_iter_offload_bytes = 0
        self._current_iter_offload_time = 0.0
        self._current_iter_reload_time = 0.0
        self._current_iter_compute_wait = 0.0

        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()

    @property
    def should_log(self) -> bool:
        return self._enabled and self._rank == self._profile_rank

    def start_iteration(self):
        """Call at the beginning of each training iteration."""
        if not self._enabled:
            return
        self._iteration_start_time = time.time()
        self._current_iter_offload_bytes = 0
        self._current_iter_offload_time = 0.0
        self._current_iter_reload_time = 0.0
        self._current_iter_compute_wait = 0.0

    def end_iteration(self):
        """Call at the end of each training iteration."""
        if not self._enabled:
            return
        iter_time = (time.time() - self._iteration_start_time) * 1000
        self._iteration_times.append(iter_time)
        self._iteration_count += 1

        if self.should_log and self._iteration_count % 10 == 0:
            self._print_iteration_summary()

    def record_offload(self, module_name: str, tensor: torch.Tensor,
                       offload_time_ms: float, wait_time_ms: float = 0.0):
        """Record an offload operation for a specific module."""
        if not self._enabled:
            return

        if module_name not in self._module_stats:
            self._module_stats[module_name] = ModuleOffloadStats(module_name)

        tensor_bytes = tensor.numel() * tensor.element_size()
        self._module_stats[module_name].add_offload(
            tensor.numel(), tensor_bytes, offload_time_ms, wait_time_ms
        )
        self._current_iter_offload_bytes += tensor_bytes
        self._current_iter_offload_time += offload_time_ms

    def record_reload(self, module_name: str, reload_time_ms: float,
                      wait_time_ms: float = 0.0, compute_wait_ms: float = 0.0):
        """Record a reload operation for a specific module."""
        if not self._enabled:
            return

        if module_name not in self._module_stats:
            self._module_stats[module_name] = ModuleOffloadStats(module_name)

        self._module_stats[module_name].add_reload(
            reload_time_ms, wait_time_ms, compute_wait_ms
        )
        self._current_iter_reload_time += reload_time_ms
        self._current_iter_compute_wait += compute_wait_ms

    def _print_iteration_summary(self):
        """Print summary for current iteration."""
        if not self.should_log:
            return

        print(f"\n{'='*80}")
        print(f"[OFFLOAD PROFILER] Iteration {self._iteration_count} Summary (Rank {self._rank})")
        print(f"{'='*80}")

        total_offload_mb = self._current_iter_offload_bytes / 1e6
        print(f"  Total offloaded: {total_offload_mb:.2f} MB")
        print(f"  Offload time: {self._current_iter_offload_time:.2f} ms")
        print(f"  Reload time: {self._current_iter_reload_time:.2f} ms")
        print(f"  Compute wait (stall): {self._current_iter_compute_wait:.2f} ms")

        # Calculate effective bandwidth
        total_bytes = self._current_iter_offload_bytes * 2
        total_time_s = (self._current_iter_offload_time + self._current_iter_reload_time) / 1000
        if total_time_s > 0:
            eff_bw = (total_bytes / 1e9) / total_time_s
            utilization = (eff_bw / self._pcie_bandwidth_gbps) * 100
            print(f"  Effective bandwidth: {eff_bw:.2f} GB/s ({utilization:.1f}% utilization)")

        # Warning if compute is waiting
        if self._current_iter_compute_wait > 1.0:
            print(f"  WARNING: Compute stalled for {self._current_iter_compute_wait:.2f} ms!")
            print(f"      Consider reducing offload modules or checking PCIe bandwidth.")

    def print_summary(self):
        """Print comprehensive profiling summary."""
        if not self.should_log:
            return

        print(f"\n{'='*80}")
        print(f"[OFFLOAD PROFILER] Final Summary")
        print(f"{'='*80}")
        print(f"Hardware Configuration:")
        print(f"  - GPU TFLOPs: {self._gpu_tflops}")
        print(f"  - PCIe Bandwidth: {self._pcie_bandwidth_gbps} GB/s")
        print(f"\nTotal Iterations: {self._iteration_count}")

        if not self._module_stats:
            print("No offload data collected.")
            return

        print(f"\n{'Module':<15} {'Total MB':<12} {'Offload ms':<12} {'Reload ms':<12} "
              f"{'Eff BW GB/s':<12} {'Stall ms':<12}")
        print("-" * 80)

        total_bytes = 0
        total_offload = 0.0
        total_reload = 0.0
        total_stall = 0.0

        for name, stats in sorted(self._module_stats.items()):
            mb = stats.total_bytes / 1e6
            offload = sum(stats.offload_times)
            reload = sum(stats.reload_times)
            stall = sum(stats.compute_wait_times)
            eff_bw = stats.effective_bandwidth_gbps

            print(f"{name:<15} {mb:<12.2f} {offload:<12.2f} {reload:<12.2f} "
                  f"{eff_bw:<12.2f} {stall:<12.2f}")

            total_bytes += stats.total_bytes
            total_offload += offload
            total_reload += reload
            total_stall += stall

        print("-" * 80)
        print(f"{'TOTAL':<15} {total_bytes/1e6:<12.2f} {total_offload:<12.2f} {total_reload:<12.2f}")

        # Performance analysis
        print(f"\n{'='*80}")
        print("Performance Analysis:")
        print(f"{'='*80}")

        total_transfer_time = total_offload + total_reload
        if self._iteration_times:
            avg_iter_time = sum(self._iteration_times) / len(self._iteration_times)
            transfer_overhead_pct = (total_transfer_time / self._iteration_count) / avg_iter_time * 100
            stall_overhead_pct = (total_stall / self._iteration_count) / avg_iter_time * 100

            print(f"  Average iteration time: {avg_iter_time:.2f} ms")
            print(f"  Transfer overhead: {transfer_overhead_pct:.2f}%")
            print(f"  Compute stall overhead: {stall_overhead_pct:.2f}%")

            if stall_overhead_pct > 5.0:
                print(f"\n  RECOMMENDATION: High compute stall detected!")
                print(f"      Consider the following optimizations:")
                self._print_recommendations()

    def _print_recommendations(self):
        """Print optimization recommendations based on collected data."""
        # Identify problematic modules
        high_stall_modules = []
        for name, stats in self._module_stats.items():
            if sum(stats.compute_wait_times) > 10:  # > 10ms total stall
                high_stall_modules.append((name, sum(stats.compute_wait_times)))

        if high_stall_modules:
            print(f"\n      High-stall modules (consider recomputation instead):")
            for name, stall in sorted(high_stall_modules, key=lambda x: -x[1]):
                print(f"        - {name}: {stall:.2f} ms stall")

        # Check bandwidth utilization
        for name, stats in self._module_stats.items():
            if stats.effective_bandwidth_gbps < self._pcie_bandwidth_gbps * 0.5:
                print(f"\n      Low bandwidth utilization for '{name}':")
                print(f"        - Consider batching smaller tensors")

    def get_optimal_offload_modules(self) -> List[str]:
        """
        Based on collected data, return the optimal set of modules to offload.

        Returns modules where offloading is more efficient than recomputation.
        """
        if not self._module_stats:
            return []

        optimal = []
        for name, stats in self._module_stats.items():
            avg_stall = sum(stats.compute_wait_times) / max(len(stats.compute_wait_times), 1)
            # If average stall is low (< 0.5ms per operation), offloading is beneficial
            if avg_stall < 0.5:
                optimal.append(name)

        return optimal

    def export_to_json(self, filepath: str):
        """Export profiling data to JSON for external analysis."""
        import json

        data = {
            "hardware": {
                "pcie_bandwidth_gbps": self._pcie_bandwidth_gbps,
                "gpu_tflops": self._gpu_tflops,
            },
            "iterations": self._iteration_count,
            "avg_iteration_time_ms": sum(self._iteration_times) / max(len(self._iteration_times), 1),
            "modules": {}
        }

        for name, stats in self._module_stats.items():
            data["modules"][name] = {
                "total_bytes": stats.total_bytes,
                "total_offload_time_ms": sum(stats.offload_times),
                "total_reload_time_ms": sum(stats.reload_times),
                "total_stall_time_ms": sum(stats.compute_wait_times),
                "effective_bandwidth_gbps": stats.effective_bandwidth_gbps,
                "num_operations": len(stats.offload_times),
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def calculate_optimal_strategy(
    modules: List[str],
    activation_sizes_mb: Dict[str, float],
    compute_flops: Dict[str, float],
    pcie_bandwidth_gbps: float = 50.0,
    gpu_tflops: float = 100.0,
) -> Dict[str, str]:
    """
    Calculate the optimal offload/recompute strategy for each module.

    Args:
        modules: List of module names
        activation_sizes_mb: Activation size in MB for each module
        compute_flops: Compute FLOPs for recomputation for each module
        pcie_bandwidth_gbps: PCIe bandwidth in GB/s
        gpu_tflops: GPU compute capability in TFLOPs

    Returns:
        Dictionary mapping module name to strategy ("offload", "recompute", or "keep")
    """
    strategy = {}

    for module in modules:
        size_mb = activation_sizes_mb.get(module, 0)
        flops = compute_flops.get(module, 0)

        # Calculate offload time (round-trip)
        # Time = Size / Bandwidth * 2 (D2H + H2D)
        offload_time_ms = (size_mb / 1000) / pcie_bandwidth_gbps * 2 * 1000

        # Calculate recompute time
        recompute_time_ms = (flops / 1e12) / gpu_tflops * 1000

        # Decision logic
        if offload_time_ms < recompute_time_ms * 0.8:
            # Offload is significantly faster
            strategy[module] = "offload"
        elif recompute_time_ms < offload_time_ms * 0.8:
            # Recompute is significantly faster
            strategy[module] = "recompute"
        elif size_mb > 100:
            # Large activation - prefer offload to save memory
            strategy[module] = "offload"
        else:
            # Similar cost - prefer keeping in memory if small
            strategy[module] = "keep" if size_mb < 10 else "offload"

    return strategy


# Example usage for Qwen3-30B-A3B model
QWEN3_30B_MODULE_SPECS = {
    # Module: (activation_size_mb_per_layer, compute_gflops_per_layer)
    "attn_norm": (0.5, 0.001),
    "qkv_linear": (1.5, 0.15),
    "core_attn": (12.0, 0.02),
    "attn_proj": (0.8, 0.05),
    "mlp_norm": (0.5, 0.001),
    "expert_fc1": (6.0, 0.3),
    "moe_act": (3.0, 0.05),
}


def get_recommended_strategy_for_hardware(
    pcie_bandwidth_gbps: float,
    gpu_tflops: float,
    batch_size: int = 32,
    seq_len: int = 4096,
    hidden_size: int = 2048,
) -> str:
    """
    Get recommended offload modules string for run_offload.sh based on hardware.

    Example:
        >>> strategy = get_recommended_strategy_for_hardware(50, 100)
        >>> print(strategy)
        "attn_norm mlp_norm moe_act attn_proj"
    """
    # Scale activation sizes by batch and sequence
    scale = (batch_size * seq_len * hidden_size) / (32 * 4096 * 2048)

    activation_sizes = {}
    compute_flops = {}

    for module, (size_mb, gflops) in QWEN3_30B_MODULE_SPECS.items():
        activation_sizes[module] = size_mb * scale
        compute_flops[module] = gflops * 1e9 * scale

    strategy = calculate_optimal_strategy(
        list(QWEN3_30B_MODULE_SPECS.keys()),
        activation_sizes,
        compute_flops,
        pcie_bandwidth_gbps,
        gpu_tflops,
    )

    offload_modules = [m for m, s in strategy.items() if s == "offload"]
    return " ".join(offload_modules)
