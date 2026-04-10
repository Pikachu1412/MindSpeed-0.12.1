# Copyright (c) 2025, Xiaohongshu Inc. All rights reserved.
#
# Adaptive Memory Profiler for Megatron-LM
#
# This module implements a lightweight, non-intrusive profiler that collects
# per-layer statistics during training warmup iterations to guide the adaptive
# memory scheduling decisions (offload vs. recompute vs. keep).
#
# Key features:
# - Per-PP-rank independent profiling and decisions
# - Memory-budget-aware optimization (GPU memory constraint)
# - Per-layer differentiated decisions (MoE vs Dense layers)
# - Dynamic re-optimization (periodic re-profiling)

import json
import logging
import math
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerModuleStats:
    """
    Per-module statistics collected during profiling for a single transformer layer.

    Fields
    ------
    compute_time_ms : float
        Wall-clock time (ms) of the forward computation, measured with CUDA Events.
    peak_memory_delta_bytes : int
        Peak GPU memory increase (bytes) during this module's forward pass,
        measured via ``torch.cuda.max_memory_allocated()``.
    sample_count : int
        Number of iterations over which the above stats were accumulated.
    """

    compute_time_ms: float = 0.0
    peak_memory_delta_bytes: int = 0
    sample_count: int = 0

    def update(self, compute_time_ms: float, peak_memory_delta_bytes: int) -> None:
        """Accumulate one sample using a running average."""
        n = self.sample_count
        self.compute_time_ms = (self.compute_time_ms * n + compute_time_ms) / (n + 1)
        self.peak_memory_delta_bytes = (
            self.peak_memory_delta_bytes * n + peak_memory_delta_bytes
        ) // (n + 1)
        self.sample_count += 1


@dataclass
class LayerStats:
    """
    Aggregated statistics for one transformer layer (all its sub-modules).

    Keyed by module name: "attn_norm", "attention", "mlp_norm", "mlp".
    """

    layer_number: int = 0
    is_moe: bool = False
    modules: Dict[str, LayerModuleStats] = field(default_factory=dict)

    def get_or_create(self, module_name: str) -> LayerModuleStats:
        if module_name not in self.modules:
            self.modules[module_name] = LayerModuleStats()
        return self.modules[module_name]

    def total_compute_time_ms(self) -> float:
        return sum(m.compute_time_ms for m in self.modules.values())


@dataclass
class PCIeBandwidthStats:
    """
    Measured PCIe bandwidth between GPU and CPU (pinned memory).

    bandwidth_gbps : float
        Effective bandwidth in GB/s (average of D2H and H2D).
    d2h_gbps : float
        Device-to-host bandwidth in GB/s.
    h2d_gbps : float
        Host-to-device bandwidth in GB/s.
    """

    bandwidth_gbps: float = 0.0
    d2h_gbps: float = 0.0
    h2d_gbps: float = 0.0
    measured: bool = False


# ---------------------------------------------------------------------------
# Per-offload-group backward profiling data
# ---------------------------------------------------------------------------


@dataclass
class OffloadGroupStats:
    """
    Per-offload-group statistics collected during profiling.

    Captures the actual GPU stall time measured at the reload wait point,
    the backward compute time of the group body, and the total bytes offloaded.
    These three metrics together enable the three-way decision:
    OFFLOAD / RECOMPUTE / KEEP.

    Fields
    ------
    group_name : str
        Name of the offload group (e.g., "qkv_linear", "core_attn").
    reload_stall_time_ms : float
        Average GPU stall time (ms) measured at the wait_event in
        on_group_commit_backward, or estimated for sync-reload (tensor_pop)
        paths. This directly captures how much the GPU blocks waiting for
        H2D data to arrive. A stall of 0 means the H2D transfer was fully
        hidden by computation overlap.
    backward_compute_time_ms : float
        Average backward compute time (ms) for this group's body, measured
        between on_group_commit_backward and on_group_start_backward via
        deferred CUDA Events. This is the time available to hide H2D transfer.
        Also used as the recompute cost estimate (forward ≈ backward / 2 for
        most ops; we use backward time directly as a conservative upper bound).
    total_offload_bytes : int
        Total bytes offloaded for this group.
    sample_count : int
        Number of stall-time samples averaged.
    backward_compute_sample_count : int
        Number of backward compute time samples averaged.
    offload_bytes_sample_count : int
        Number of offload-bytes samples taken.
    layer_number : int
        Layer number for per-layer differentiation. -1 means aggregated
        across all layers (backward-compatible default).
    is_moe : bool
        Whether this group comes from a MoE layer.
    """

    group_name: str = ""
    reload_stall_time_ms: float = 0.0
    backward_compute_time_ms: float = 0.0
    forward_compute_time_ms: float = 0.0
    total_offload_bytes: int = 0
    sample_count: int = 0
    backward_compute_sample_count: int = 0
    forward_compute_sample_count: int = 0
    offload_bytes_sample_count: int = 0
    layer_number: int = -1
    is_moe: bool = False

    def update_reload_stall(self, stall_time_ms: float) -> None:
        """Accumulate one sample of reload stall time."""
        n = self.sample_count
        self.reload_stall_time_ms = (self.reload_stall_time_ms * n + stall_time_ms) / (
            n + 1
        )
        self.sample_count += 1

    def update_backward_compute_time(self, compute_time_ms: float) -> None:
        """Accumulate one sample of backward compute time."""
        n = self.backward_compute_sample_count
        self.backward_compute_time_ms = (
            self.backward_compute_time_ms * n + compute_time_ms
        ) / (n + 1)
        self.backward_compute_sample_count += 1

    def update_forward_compute_time(self, compute_time_ms: float) -> None:
        """Accumulate one sample of forward compute time (direct recompute cost)."""
        n = self.forward_compute_sample_count
        self.forward_compute_time_ms = (
            self.forward_compute_time_ms * n + compute_time_ms
        ) / (n + 1)
        self.forward_compute_sample_count += 1

    def update_offload_bytes(self, nbytes: int) -> None:
        """Update total offload bytes (takes the max across samples for stability)."""
        self.total_offload_bytes = max(self.total_offload_bytes, nbytes)
        self.offload_bytes_sample_count += 1


# ---------------------------------------------------------------------------
# CUDA Event timer context manager
# ---------------------------------------------------------------------------


class CudaTimer:
    """
    Lightweight CUDA Event-based timer.

    Usage::

        with CudaTimer() as t:
            some_cuda_op()
        elapsed_ms = t.elapsed_ms()
    """

    def __init__(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._elapsed: Optional[float] = None

    def __enter__(self):
        self._start.record()
        return self

    def __exit__(self, *args):
        self._end.record()
        return False

    def elapsed_ms(self) -> float:
        """Synchronize the default stream and return elapsed time in ms."""
        if self._elapsed is None:
            # Only synchronize the default (compute) stream instead of all
            # streams.  The start/end events are recorded on the default
            # stream, so waiting for it is sufficient.
            torch.cuda.current_stream().synchronize()
            self._elapsed = self._start.elapsed_time(self._end)
        return self._elapsed


# ---------------------------------------------------------------------------
# PCIe bandwidth measurement
# ---------------------------------------------------------------------------


def measure_pcie_bandwidth(
    tensor_size_mb: float = 256.0,
    num_warmup: int = 3,
    num_trials: int = 5,
) -> PCIeBandwidthStats:
    """
    Measure effective PCIe bandwidth between GPU and pinned CPU memory.

    Parameters
    ----------
    tensor_size_mb : float
        Size of the test tensor in MB.
    num_warmup : int
        Number of warmup transfers before timing.
    num_trials : int
        Number of timed transfers to average.

    Returns
    -------
    PCIeBandwidthStats
        Measured D2H, H2D, and average bandwidth in GB/s.
    """
    if not torch.cuda.is_available():
        return PCIeBandwidthStats()

    numel = int(tensor_size_mb * 1024 * 1024 / 4)  # float32 elements
    gpu_tensor = torch.randn(numel, dtype=torch.float32, device="cuda")
    cpu_tensor = torch.empty(numel, dtype=torch.float32, pin_memory=True)

    d2h_stream = torch.cuda.Stream()
    h2d_stream = torch.cuda.Stream()

    def _time_transfer(src, dst, stream, n_warmup, n_trials):
        # Warmup
        for _ in range(n_warmup):
            with torch.cuda.stream(stream):
                dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()

        # Timed trials
        elapsed_list = []
        for _ in range(n_trials):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start_evt.record(stream)
                dst.copy_(src, non_blocking=True)
                end_evt.record(stream)
            torch.cuda.synchronize()
            elapsed_list.append(start_evt.elapsed_time(end_evt))  # ms

        return sum(elapsed_list) / len(elapsed_list)  # average ms

    # D2H
    d2h_ms = _time_transfer(gpu_tensor, cpu_tensor, d2h_stream, num_warmup, num_trials)
    # H2D
    h2d_ms = _time_transfer(cpu_tensor, gpu_tensor, h2d_stream, num_warmup, num_trials)

    size_gb = tensor_size_mb / 1024.0
    d2h_gbps = size_gb / (d2h_ms / 1000.0)
    h2d_gbps = size_gb / (h2d_ms / 1000.0)
    avg_gbps = (d2h_gbps + h2d_gbps) / 2.0

    # Cleanup
    del gpu_tensor, cpu_tensor

    return PCIeBandwidthStats(
        bandwidth_gbps=avg_gbps,
        d2h_gbps=d2h_gbps,
        h2d_gbps=h2d_gbps,
        measured=True,
    )


# ---------------------------------------------------------------------------
# OffloadRecomputeOptimizer — the decision engine
# ---------------------------------------------------------------------------

# Recompute cost ratio: forward_time ≈ backward_time * RECOMPUTE_COST_RATIO.
# For most ops (linear, attention), forward is roughly half the backward cost
# (backward computes two gradients). Use 0.5 as default; can be overridden via
# OFFLOAD_RECOMPUTE_COST_RATIO env var.
_DEFAULT_RECOMPUTE_COST_RATIO = 0.5


class OffloadRecomputeOptimizer:
    """
    Three-way optimizer: decides OFFLOAD / RECOMPUTE / KEEP for each offload
    group based on directly measured GPU stall times and backward compute times.

    Decision logic
    --------------
    For each group, three options are evaluated:

    1. **OFFLOAD** (default): Activations are D2H during forward and H2D during
       backward. Cost = reload_stall_ms (measured). Benefit = saves activation
       memory. Chosen when stall ≤ stall_threshold_ms (transfer is hidden).

    2. **RECOMPUTE**: Activations are discarded after forward; the forward pass
       is re-run during backward to regenerate them. Cost = recompute_ms
       (estimated as backward_compute_ms × recompute_cost_ratio). Benefit =
       saves activation memory (same as OFFLOAD). Chosen when stall >
       stall_threshold_ms AND recompute_ms < stall_ms (recomputing is cheaper
       than waiting for H2D).

    3. **KEEP**: Activations stay on GPU throughout. Cost = extra GPU memory.
       Benefit = zero compute overhead. Chosen when stall > stall_threshold_ms
       AND recompute_ms ≥ stall_ms (both alternatives are expensive).

    Enhanced features:
    - Memory-budget-aware: can enforce a GPU memory budget that prevents
      too many KEEP decisions from OOMing.
    - Per-PP-rank aware: each pipeline stage makes independent decisions
      based on its own measured stall times.
    """

    def __init__(
        self,
        group_stats: Dict[str, OffloadGroupStats],
        pcie_stats: PCIeBandwidthStats,
        stall_threshold_ms: float = 1.0,
        recompute_cost_ratio: float = _DEFAULT_RECOMPUTE_COST_RATIO,
        memory_budget_mb: float = 0.0,
        num_layers_on_this_rank: int = 1,
        pp_rank: int = -1,
    ):
        """
        Parameters
        ----------
        group_stats : Dict[str, OffloadGroupStats]
            Per-group profiling data with measured reload stall times and
            backward compute times.
        pcie_stats : PCIeBandwidthStats
            Measured PCIe bandwidth (used for reporting only).
        stall_threshold_ms : float
            Minimum measured stall (ms) to trigger skip/recompute. Groups with
            stall below this threshold are kept as offloaded.
        recompute_cost_ratio : float
            Ratio of forward compute time to backward compute time.
            recompute_ms = backward_compute_ms × recompute_cost_ratio.
            Default 0.5 (forward ≈ half of backward for linear/attention ops).
        memory_budget_mb : float
            Maximum additional GPU memory (MB) allowed for KEEP decisions
            across all layers on this PP rank. 0 = unlimited (no budget
            constraint). When set, the optimizer solves a knapsack problem:
            maximize stall reduction subject to the memory budget.
        num_layers_on_this_rank : int
            Number of transformer layers on this PP rank. Used to compute
            total memory impact of KEEP decisions (keep_bytes × num_layers).
        pp_rank : int
            Pipeline parallel rank of this optimizer instance. -1 means
            unknown / single-PP-rank mode (backward-compatible).
        """
        self._group_stats = group_stats
        self._pcie_stats = pcie_stats
        self._stall_threshold_ms = stall_threshold_ms
        self._recompute_cost_ratio = recompute_cost_ratio
        self._memory_budget_mb = memory_budget_mb
        self._num_layers = num_layers_on_this_rank
        self._pp_rank = pp_rank

    def compute_decisions(self) -> Tuple[Set[str], Set[str]]:
        """
        Analyze measured stall, backward compute, and forward compute times,
        then return the three-way decision for each active offload group.

        Enhanced decision logic (v2):
        - Uses directly measured forward compute time as recompute cost when
          available, falling back to backward_compute_ms × ratio.
        - Coupled-group awareness: attention sub-groups (qkv_linear, core_attn,
          attn_proj) are decided as a unit since recompute wraps the entire
          self_attention call. Similarly for MoE sub-groups (expert_fc1, moe_act).
        - Net-benefit ranking: decisions are ranked by net_benefit = stall_saved
          - recompute_cost, enabling better budget allocation.
        - Overlap-aware thresholding: a group's stall is compared against the
          backward compute time of the *preceding* group in the backward
          execution order, which determines how much overlap is available.

        Returns
        -------
        skip_groups : Set[str]
            Groups to keep on GPU (KEEP decision).
        recompute_groups : Set[str]
            Groups to recompute during backward (RECOMPUTE decision).
        """
        # The backward execution order per layer (reverse of forward commit order).
        backward_order = [
            "mlp_norm",
            "moe_act",
            "expert_fc1",
            "attn_norm",
            "attn_proj",
            "core_attn",
            "qkv_linear",
        ]

        # Coupled groups: these sub-groups are recomputed together.
        # If ANY sub-group in a coupled set is marked for RECOMPUTE, ALL of them
        # are recomputed (because the checkpoint wraps the entire module).
        _ATTN_COUPLED = {"qkv_linear", "core_attn", "attn_proj"}
        _MOE_COUPLED = {"expert_fc1", "moe_act"}

        # Filter to only groups that are actually being offloaded (have stats)
        active_groups = [g for g in backward_order if g in self._group_stats]
        if not active_groups:
            return set(), set()

        h2d_gbps = (
            self._pcie_stats.h2d_gbps
            if (self._pcie_stats and self._pcie_stats.measured)
            else 0.0
        )

        skip_groups: Set[str] = set()
        recompute_groups: Set[str] = set()

        pp_label = f" (PP rank {self._pp_rank})" if self._pp_rank >= 0 else ""
        analysis_lines = [
            "",
            f"[OffloadRecomputeOptimizer] Three-Way Analysis v2{pp_label}:",
            f"  stall_threshold={self._stall_threshold_ms:.1f}ms  "
            f"recompute_cost_ratio={self._recompute_cost_ratio:.2f}  "
            f"PCIe H2D={h2d_gbps:.2f} GB/s  "
            f"memory_budget={'unlimited' if self._memory_budget_mb <= 0 else f'{self._memory_budget_mb:.0f}MB'}  "
            f"num_layers={self._num_layers}",
        ]
        analysis_lines.append(
            f"  {'Group':<14} {'Offload(MB)':>11} {'Stall(ms)':>10} "
            f"{'BwdComp(ms)':>12} {'FwdComp(ms)':>12} {'Recomp(ms)':>11} "
            f"{'NetBenefit':>11} {'Samples':>8} {'Decision':>10}"
        )
        analysis_lines.append("  " + "-" * 112)

        # --- Phase 1: Compute per-group metrics ---
        # For each group, compute the recompute cost using directly measured
        # forward compute time when available.
        group_metrics: Dict[str, Dict] = {}
        for group_name in active_groups:
            stats = self._group_stats[group_name]
            if stats.total_offload_bytes == 0:
                continue

            stall_ms = stats.reload_stall_time_ms
            bwd_compute_ms = stats.backward_compute_time_ms
            fwd_compute_ms = stats.forward_compute_time_ms

            # Use directly measured forward time if available, otherwise
            # fall back to backward * ratio estimate.
            if fwd_compute_ms > 0 and stats.forward_compute_sample_count >= 2:
                recompute_ms = fwd_compute_ms
                recompute_source = "measured"
            else:
                recompute_ms = bwd_compute_ms * self._recompute_cost_ratio
                recompute_source = "estimated"

            offload_mb = stats.total_offload_bytes / (1024**2)

            group_metrics[group_name] = {
                "stall_ms": stall_ms,
                "bwd_compute_ms": bwd_compute_ms,
                "fwd_compute_ms": fwd_compute_ms,
                "recompute_ms": recompute_ms,
                "recompute_source": recompute_source,
                "offload_mb": offload_mb,
                "total_offload_bytes": stats.total_offload_bytes,
                "sample_count": stats.sample_count,
            }

        # --- Phase 2: Coupled-group-aware decisions ---
        # For coupled groups (attention sub-groups, MoE sub-groups), compute
        # aggregate metrics and make a single decision for the entire set.
        #
        # Decision priority (v3):
        #   1. All stalls < threshold → OFFLOAD (well overlapped, no action needed)
        #   2. KEEP if memory budget allows (zero recompute overhead, pure win)
        #   3. RECOMPUTE if net_benefit > 0 (stall saved > recompute cost)
        #   4. OFFLOAD otherwise (accept the stall, cheaper than recompute)
        #
        # Previous logic (v2) prioritized RECOMPUTE over KEEP, causing large
        # groups like expert_fc1+moe_act to be recomputed even when 50+ GB of
        # GPU memory was sitting idle.  v3 fixes this by treating KEEP as the
        # preferred option whenever the memory budget permits.
        def _decide_coupled_group(
            coupled_set: Set[str],
            metrics: Dict[str, Dict],
        ) -> Dict[str, str]:
            """Decide strategy for a coupled group as a unit."""
            active_in_set = [g for g in coupled_set if g in metrics]
            if not active_in_set:
                return {}

            # Aggregate: total stall saved by not offloading the entire set,
            # and total recompute cost if the entire set is recomputed.
            total_stall = sum(metrics[g]["stall_ms"] for g in active_in_set)
            total_recompute = sum(metrics[g]["recompute_ms"] for g in active_in_set)
            total_mem_mb = sum(metrics[g]["offload_mb"] for g in active_in_set)

            # All stalls below threshold → OFFLOAD (well overlapped)
            all_low_stall = all(
                metrics[g]["stall_ms"] <= self._stall_threshold_ms
                for g in active_in_set
            )
            if all_low_stall:
                return {g: "OFFLOAD" for g in active_in_set}

            # Any stall above threshold → needs intervention (KEEP or RECOMPUTE).
            # KEEP is always better in terms of time (zero recompute overhead),
            # so prefer KEEP.  Memory budget enforcement is in Phase 4.
            return {g: "KEEP" for g in active_in_set}

        # Decide coupled groups
        coupled_decisions: Dict[str, str] = {}
        coupled_decisions.update(_decide_coupled_group(_ATTN_COUPLED, group_metrics))
        coupled_decisions.update(_decide_coupled_group(_MOE_COUPLED, group_metrics))

        # --- Phase 3: Decide independent groups (attn_norm, mlp_norm) ---
        independent_groups = set(group_metrics.keys()) - _ATTN_COUPLED - _MOE_COUPLED
        for group_name in independent_groups:
            m = group_metrics[group_name]
            stall_ms = m["stall_ms"]

            if stall_ms <= self._stall_threshold_ms:
                coupled_decisions[group_name] = "OFFLOAD"
            else:
                # Prefer KEEP; budget enforcement in Phase 4 may downgrade.
                coupled_decisions[group_name] = "KEEP"

        raw_decisions = coupled_decisions

        # --- Phase 4: Memory-budget-aware adjustment ---
        # KEEP decisions that would exceed the memory budget are downgraded
        # using a benefit-density-aware greedy knapsack.  Downgrade goes to
        # RECOMPUTE if recompute is beneficial, otherwise stays as OFFLOAD.
        if self._memory_budget_mb > 0 and self._num_layers > 0:
            keep_candidates = []
            for g, dec in raw_decisions.items():
                if dec == "KEEP" and g in group_metrics:
                    m = group_metrics[g]
                    total_mem_mb = m["offload_mb"] * self._num_layers
                    # Net benefit of KEEP = stall eliminated (no compute cost)
                    net_benefit = m["stall_ms"]
                    keep_candidates.append((g, total_mem_mb, net_benefit))

            # Sort by benefit density: stall_saved / memory_cost (descending)
            # Groups with high stall per MB of memory are kept first.
            keep_candidates.sort(key=lambda x: x[2] / max(x[1], 1e-6), reverse=True)

            remaining_budget_mb = self._memory_budget_mb
            for g, total_mem_mb, net_benefit in keep_candidates:
                if total_mem_mb <= remaining_budget_mb:
                    remaining_budget_mb -= total_mem_mb
                else:
                    # Exceeds budget: downgrade from KEEP.
                    m = group_metrics[g]
                    recompute_net = m["stall_ms"] - m["recompute_ms"]
                    if recompute_net > 0:
                        raw_decisions[g] = "RECOMPUTE*"
                        analysis_lines.append(
                            f"  [BUDGET] {g}: KEEP→RECOMPUTE* "
                            f"(total_mem={total_mem_mb:.1f}MB exceeds remaining budget "
                            f"{remaining_budget_mb:.1f}MB, recompute net_benefit={recompute_net:.2f}ms)"
                        )
                    else:
                        raw_decisions[g] = "OFFLOAD*"
                        analysis_lines.append(
                            f"  [BUDGET] {g}: KEEP→OFFLOAD* "
                            f"(total_mem={total_mem_mb:.1f}MB exceeds remaining budget "
                            f"{remaining_budget_mb:.1f}MB, recompute net_benefit={recompute_net:.2f}ms NEGATIVE)"
                        )
            # For coupled groups, if any member is downgraded, ALL members of
            # the coupled set must be downgraded to the same decision (since
            # recompute wraps the entire coupled module).
            for coupled_set in [_ATTN_COUPLED, _MOE_COUPLED]:
                active_in_set = [g for g in coupled_set if g in raw_decisions]
                if not active_in_set:
                    continue
                decisions_in_set = [raw_decisions[g] for g in active_in_set]
                # If any member was downgraded (has '*'), apply the most conservative
                # downgraded decision to all members.
                has_downgrade = any(d.endswith("*") for d in decisions_in_set)
                if has_downgrade:
                    # Prefer RECOMPUTE* if any member can benefit; otherwise OFFLOAD*.
                    if any(d == "RECOMPUTE*" for d in decisions_in_set):
                        for g in active_in_set:
                            raw_decisions[g] = "RECOMPUTE*"
                    else:
                        for g in active_in_set:
                            raw_decisions[g] = "OFFLOAD*"

        # --- Phase 5: Apply final decisions and generate report ---
        for group_name in active_groups:
            if group_name not in raw_decisions or group_name not in group_metrics:
                continue
            m = group_metrics[group_name]
            decision = raw_decisions[group_name]

            if decision in ("RECOMPUTE", "RECOMPUTE*"):
                recompute_groups.add(group_name)
            elif decision == "KEEP":
                skip_groups.add(group_name)
            # OFFLOAD / OFFLOAD*: no action needed (default behavior)

            # Compute net_benefit for display
            if decision in ("RECOMPUTE", "RECOMPUTE*"):
                net_benefit = m["stall_ms"] - m["recompute_ms"]
            elif decision == "KEEP":
                net_benefit = m["stall_ms"]  # pure stall saving, no compute cost
            else:
                net_benefit = 0.0  # OFFLOAD / OFFLOAD*: stall is accepted

            src_tag = (
                f"({m['recompute_source'][0]})"
                if decision.startswith("RECOMPUTE")
                else ""
            )

            analysis_lines.append(
                f"  {group_name:<14} {m['offload_mb']:>11.1f} "
                f"{m['stall_ms']:>10.2f} "
                f"{m['bwd_compute_ms']:>12.2f} "
                f"{m['fwd_compute_ms']:>12.2f} "
                f"{m['recompute_ms']:>11.2f} "
                f"{net_benefit:>11.2f} "
                f"{m['sample_count']:>8} {decision + src_tag:>10}"
            )

        analysis_lines.append("")

        # --- Summary ---
        keep_bytes = sum(
            self._group_stats[g].total_offload_bytes
            for g in skip_groups
            if g in self._group_stats
        )
        recompute_bytes = sum(
            self._group_stats[g].total_offload_bytes
            for g in recompute_groups
            if g in self._group_stats
        )
        keep_stall_saved = sum(
            group_metrics[g]["stall_ms"] for g in skip_groups if g in group_metrics
        )
        recompute_stall_saved = sum(
            group_metrics[g]["stall_ms"] for g in recompute_groups if g in group_metrics
        )
        recompute_cost = sum(
            group_metrics[g]["recompute_ms"]
            for g in recompute_groups
            if g in group_metrics
        )
        net_time_saved = (keep_stall_saved + recompute_stall_saved) - recompute_cost

        analysis_lines += [
            f"  KEEP groups     : {skip_groups if skip_groups else 'none'}",
            f"    Extra GPU memory : {keep_bytes / (1024**2):.1f} MB/layer  "
            f"({keep_bytes / (1024**2) * self._num_layers:.1f} MB total on this PP rank)",
            f"    Stall eliminated : {keep_stall_saved:.2f} ms/layer",
            f"  RECOMPUTE groups: {recompute_groups if recompute_groups else 'none'}",
            f"    Memory saved     : {recompute_bytes / (1024**2):.1f} MB/layer (vs KEEP)",
            f"    Stall eliminated : {recompute_stall_saved:.2f} ms/layer",
            f"    Recompute cost   : {recompute_cost:.2f} ms/layer",
            f"  Net time saved vs all-OFFLOAD: {net_time_saved:.2f} ms/layer",
            f"  (KEEP saves time at memory cost; RECOMPUTE saves both time and memory)",
        ]

        msg = "\n".join(analysis_lines)
        logger.info(msg)
        _global_rank = (
            torch.distributed.get_rank()
            if (torch.distributed.is_available() and torch.distributed.is_initialized())
            else -1
        )
        print(f"[global_rank={_global_rank}] {msg}", flush=True)

        return skip_groups, recompute_groups

    # Keep backward-compatible alias
    def compute_skip_groups(self) -> Set[str]:
        """Backward-compatible: returns only skip_groups (KEEP decisions)."""
        skip_groups, _ = self.compute_decisions()
        return skip_groups


# ---------------------------------------------------------------------------
# Main profiler class
# ---------------------------------------------------------------------------


class AdaptiveMemoryProfiler:
    """
    Singleton profiler that collects per-layer memory and compute statistics
    during the warmup phase of training.

    Lifecycle
    ---------
    1. Call ``AdaptiveMemoryProfiler.get_instance()`` to obtain the singleton.
    2. At the start of each iteration, call ``on_iteration_start(iteration)``.
    3. In ``TransformerLayer.forward``, wrap sub-module calls with
       ``profile_module(layer_number, module_name, is_moe)`` context manager.
    4. After ``num_profile_iters`` iterations, ``is_profiling_done()`` returns
       True and the collected stats are available via ``get_layer_stats()``.
    5. Optionally call ``save_to_json(path)`` to persist results.

    Parameters
    ----------
    num_profile_iters : int
        Number of training iterations to collect profiling data.
    profile_rank : int
        Only the rank with this global rank ID performs profiling (default 0).
    measure_pcie : bool
        Whether to measure PCIe bandwidth at initialization.
    """

    _instance: Optional["AdaptiveMemoryProfiler"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "AdaptiveMemoryProfiler":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = AdaptiveMemoryProfiler()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (useful for unit tests)."""
        with cls._lock:
            cls._instance = None

    def __init__(
        self,
        num_profile_iters: int = 10,
        profile_rank: int = 0,
        measure_pcie: bool = True,
    ):
        # Read from environment variables so users can override without code changes
        self._num_profile_iters = int(
            os.environ.get("ADAPTIVE_MEM_PROFILE_ITERS", str(num_profile_iters))
        )
        self._profile_rank = int(
            os.environ.get("ADAPTIVE_MEM_PROFILE_RANK", str(profile_rank))
        )
        self._measure_pcie = (
            os.environ.get("ADAPTIVE_MEM_MEASURE_PCIE", "1") == "1" and measure_pcie
        )

        # --- Per-PP-Rank profiling ---
        # When enabled, each PP stage independently profiles and makes decisions.
        # The stall data recording (cheap: just CUDA events) runs on one rank per
        # PP stage, while expensive module profiling stays on the original profile_rank.
        self._per_pp_rank_enabled: bool = (
            os.environ.get("ADAPTIVE_MEM_PER_PP_RANK", "1") == "1"
        )
        # Will be populated lazily when torch.distributed is initialized:
        self._pp_rank: int = -1
        self._pp_world_size: int = 1
        self._tp_rank: int = -1
        self._dp_rank: int = -1
        self._num_layers_on_this_rank: int = 0  # set externally or detected
        # True if this rank should collect stall/bytes data (one per PP stage)
        self._is_stall_profile_rank: bool = False
        self._pp_rank_initialized: bool = False

        # --- Memory budget ---
        # Maximum additional GPU memory (MB) allowed for KEEP decisions.
        # 0 = unlimited. Auto-detected from available GPU memory if set to "auto".
        _budget_env = os.environ.get("ADAPTIVE_MEM_BUDGET_MB", "0")
        if _budget_env.lower() == "auto":
            self._memory_budget_mb = -1.0  # sentinel for auto-detect
        else:
            self._memory_budget_mb = float(_budget_env)

        # --- Dynamic re-optimization ---
        # Re-profile every N iterations after the initial optimization.
        # 0 = disabled (one-shot optimization).
        self._reoptimize_interval: int = int(
            os.environ.get("ADAPTIVE_MEM_REOPTIMIZE_INTERVAL", "0")
        )
        self._last_optimize_iter: int = -1
        self._reoptimize_count: int = 0

        # Per-layer statistics: layer_number -> LayerStats
        self._layer_stats: Dict[int, LayerStats] = {}

        # Per-offload-group statistics: group_name -> OffloadGroupStats
        self._offload_group_stats: Dict[str, OffloadGroupStats] = {}

        # Pending CUDA event pairs for deferred stall measurement.
        # Each entry: (group_name, stall_start_event, stall_end_event)
        self._pending_stall_events: List[
            Tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []

        # Pending CUDA event pairs for deferred backward compute time measurement.
        # Each entry: (group_name, bwd_start_event, bwd_end_event)
        # bwd_start is recorded at on_group_commit_backward (after wait_event),
        # bwd_end is recorded at on_group_start_backward.
        # The interval captures the actual backward compute time of the group body.
        self._pending_bwd_compute_events: List[
            Tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []

        # Pending CUDA event pairs for deferred forward compute time measurement.
        # Each entry: (group_name, fwd_start_event, fwd_end_event)
        # fwd_start is recorded at on_group_start_forward,
        # fwd_end is recorded at on_group_commit_forward.
        # The interval captures the actual forward compute time — the direct
        # cost of recomputing this group during backward.
        self._pending_fwd_compute_events: List[
            Tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []

        # Optimization result: set of group names to recompute (instead of offload/keep)
        self._recompute_groups: Set[str] = set()

        # PCIe bandwidth
        self._pcie_stats: Optional[PCIeBandwidthStats] = None

        # Iteration tracking
        self._current_iter: int = 0
        self._profiling_done: bool = False

        # Whether this rank should do profiling (module profiling — expensive)
        self._is_profile_rank: bool = self._check_is_profile_rank()

        # Optimization result: set of group names to skip offloading
        self._skip_offload_groups: Set[str] = set()
        self._optimization_applied: bool = False

        # Skip the first N iterations before starting to profile.
        # This lets the training pipeline reach a stable state (loss scaling,
        # pipeline warmup) before we inject heavy synchronization points.
        self._warmup_skip_iters: int = int(
            os.environ.get("ADAPTIVE_MEM_WARMUP_SKIP_ITERS", "2")
        )

        # Diagnostic counter to limit debug prints
        self._diag_print_count: int = 0
        self._diag_print_limit: int = 20  # only print first N diagnostic messages

        # Flag to track whether PCIe bandwidth has been measured.
        # Measurement is deferred to the first profiling-active iteration
        # (after warmup_skip_iters) to avoid injecting heavy
        # torch.cuda.synchronize() calls at the very start of training,
        # which disrupts pipeline-parallel timing and causes NaN on
        # non-profile ranks.
        self._pcie_measured: bool = False

        logger.info(
            f"[AdaptiveMemoryProfiler] Initialized: "
            f"num_profile_iters={self._num_profile_iters}, "
            f"profile_rank={self._profile_rank}, "
            f"is_profile_rank={self._is_profile_rank}, "
            f"warmup_skip_iters={self._warmup_skip_iters}, "
            f"per_pp_rank={self._per_pp_rank_enabled}, "
            f"memory_budget_mb={self._memory_budget_mb}, "
            f"reoptimize_interval={self._reoptimize_interval}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_is_profile_rank(self) -> bool:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return True  # single-GPU mode: always profile
        # Use get_rank() without a process group to get the global rank.
        # This is called lazily on first use so that distributed is guaranteed
        # to be initialized at call time.
        try:
            return torch.distributed.get_rank() == self._profile_rank
        except Exception:
            return False

    def _refresh_is_profile_rank(self) -> None:
        """Re-evaluate _is_profile_rank in case distributed was not yet initialized at __init__."""
        self._is_profile_rank = self._check_is_profile_rank()
        # Also initialize per-PP-rank profiling if not yet done.
        if not self._pp_rank_initialized:
            self._init_pp_rank_profiling()

    def _init_pp_rank_profiling(self) -> None:
        """
        Initialize per-PP-rank profiling state.

        Determines this rank's PP rank, TP rank, DP rank, and whether it should
        serve as the stall-profiling representative for its PP stage.

        Called lazily when torch.distributed is first confirmed initialized.
        """
        if self._pp_rank_initialized:
            return
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return

        try:
            from megatron.core import parallel_state

            self._pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            self._pp_world_size = (
                parallel_state.get_pipeline_model_parallel_world_size()
            )
            self._tp_rank = parallel_state.get_tensor_model_parallel_rank()
            self._dp_rank = parallel_state.get_data_parallel_rank()
        except Exception:
            # parallel_state not yet initialized; retry next time
            return

        self._pp_rank_initialized = True

        if self._per_pp_rank_enabled and self._pp_world_size > 1:
            # Each PP stage needs exactly one rank to profile stall data.
            # Choose the rank with TP-rank=0 and DP-rank=0 within each PP stage.
            self._is_stall_profile_rank = self._tp_rank == 0 and self._dp_rank == 0
        else:
            # Fallback: only the original profile_rank does stall profiling.
            self._is_stall_profile_rank = self._is_profile_rank

        # Auto-detect memory budget from available GPU memory.
        # Use a generous fraction (50%) of free memory.  The offload mechanism
        # already frees activation memory during forward — the budget only
        # constrains how many activations are KEPT on GPU instead of being
        # offloaded.  With L20Y 80GB GPUs and typical 20-30 GB base usage,
        # 50% of free (~25-30 GB) is safe and allows KEEP for large groups
        # like expert_fc1 (1024 MB × 2 layers = ~2.0 GB) that would otherwise
        # be unnecessarily recomputed.  The old 10% setting was far too
        # conservative and wasted 50+ GB of idle GPU memory.
        _budget_fraction = float(os.environ.get("ADAPTIVE_MEM_BUDGET_FRACTION", "0.50"))
        if self._memory_budget_mb < 0 and torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0]
            self._memory_budget_mb = (free_mem / (1024**2)) * _budget_fraction
            logger.info(
                f"[AdaptiveMemoryProfiler] Auto-detected memory budget: "
                f"{self._memory_budget_mb:.0f} MB "
                f"({_budget_fraction * 100:.0f}% of {free_mem / (1024**2):.0f} MB free)"
            )

        global_rank = torch.distributed.get_rank()
        if global_rank < 8 or self._is_stall_profile_rank:
            print(
                f"[AdaptiveMemProfiler][PP-INIT] global_rank={global_rank} "
                f"pp_rank={self._pp_rank}/{self._pp_world_size} "
                f"tp_rank={self._tp_rank} dp_rank={self._dp_rank} "
                f"is_profile_rank={self._is_profile_rank} "
                f"is_stall_profile_rank={self._is_stall_profile_rank} "
                f"per_pp_rank={self._per_pp_rank_enabled} "
                f"memory_budget_mb={self._memory_budget_mb:.0f}",
                flush=True,
            )

    def set_num_layers_on_this_rank(self, num_layers: int) -> None:
        """Set the number of transformer layers on this PP rank (for memory budget)."""
        self._num_layers_on_this_rank = num_layers

    def is_stall_profiling_active(self) -> bool:
        """
        Return True if stall/bytes data recording should be active this iteration.

        Stall profiling is lightweight (just CUDA events, no synchronization) and
        runs on one rank per PP stage when per-PP-rank profiling is enabled.
        This is separate from ``is_profiling_active()`` which gates the expensive
        module profiling (with torch.cuda.synchronize()).
        """
        if self._profiling_done and self._reoptimize_interval <= 0:
            return False
        if self._current_iter < self._warmup_skip_iters:
            return False
        return self._is_stall_profile_rank

    def _measure_pcie_bandwidth_safe(self) -> PCIeBandwidthStats:
        """Measure PCIe bandwidth, catching any errors gracefully."""
        try:
            stats = measure_pcie_bandwidth()
            logger.info(
                f"[AdaptiveMemoryProfiler] PCIe bandwidth: "
                f"D2H={stats.d2h_gbps:.2f} GB/s, "
                f"H2D={stats.h2d_gbps:.2f} GB/s, "
                f"avg={stats.bandwidth_gbps:.2f} GB/s"
            )
            return stats
        except Exception as e:
            logger.warning(f"[AdaptiveMemoryProfiler] PCIe measurement failed: {e}")
            return PCIeBandwidthStats()

    def _get_or_create_layer(self, layer_number: int, is_moe: bool) -> LayerStats:
        if layer_number not in self._layer_stats:
            self._layer_stats[layer_number] = LayerStats(
                layer_number=layer_number, is_moe=is_moe
            )
        return self._layer_stats[layer_number]

    def _get_or_create_offload_group(self, group_name: str) -> OffloadGroupStats:
        if group_name not in self._offload_group_stats:
            self._offload_group_stats[group_name] = OffloadGroupStats(
                group_name=group_name
            )
        return self._offload_group_stats[group_name]

    # ------------------------------------------------------------------
    # Public API: iteration lifecycle
    # ------------------------------------------------------------------

    def on_iteration_start(self, iteration: int) -> None:
        """
        Called at the beginning of each training iteration.

        Parameters
        ----------
        iteration : int
            Current training iteration index (0-based or 1-based, consistent
            with the caller).
        """
        # Dynamic re-optimization: check if we should re-profile
        if self._profiling_done and self._reoptimize_interval > 0:
            iters_since = iteration - self._last_optimize_iter
            if iters_since >= self._reoptimize_interval:
                self._start_reoptimization(iteration)

        if self._profiling_done:
            return
        self._current_iter = iteration

        # Re-evaluate _is_profile_rank on the first few iterations in case
        # torch.distributed was not yet initialized when the singleton was created.
        if not self._is_profile_rank or not self._pp_rank_initialized:
            self._refresh_is_profile_rank()

        # Deferred PCIe bandwidth measurement: perform on the first iteration
        # where profiling becomes active (after warmup_skip).  This avoids
        # injecting torch.cuda.synchronize() calls during early iterations
        # where pipeline timing is being established.
        if (
            not self._pcie_measured
            and self._is_stall_profile_rank
            and self._measure_pcie
            and torch.cuda.is_available()
            and iteration >= self._warmup_skip_iters
        ):
            self._pcie_stats = self._measure_pcie_bandwidth_safe()
            self._pcie_measured = True

        if iteration < 3 and (self._is_profile_rank or self._is_stall_profile_rank):
            _global_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else -1
            )
            print(
                f"[AdaptiveMemProfiler][DIAG] on_iteration_start: "
                f"global_rank={_global_rank} "
                f"iter={iteration} is_profile_rank={self._is_profile_rank} "
                f"is_stall_profile_rank={self._is_stall_profile_rank} "
                f"pp_rank={self._pp_rank}/{self._pp_world_size} "
                f"profiling_done={self._profiling_done} "
                f"id(self)={id(self)} "
                f"warmup_skip_iters={self._warmup_skip_iters}",
                flush=True,
            )

    def on_iteration_end(self, iteration: int) -> None:
        """
        Called at the end of each training iteration.

        Processes any pending CUDA Event pairs collected during backward to
        compute stall times (deferred measurement), then marks profiling as
        done after ``num_profile_iters`` iterations.
        """
        if self._profiling_done and self._reoptimize_interval <= 0:
            return

        # --- Deferred measurement: stall + backward/forward compute time ---
        # During forward/backward, CUDA Event pairs were recorded but NOT
        # synchronized (to avoid blocking the compute path). Now we synchronize
        # once and read all elapsed times.
        has_pending = (
            self._pending_stall_events
            or self._pending_bwd_compute_events
            or self._pending_fwd_compute_events
        )
        if has_pending:
            torch.cuda.synchronize()

            for group_name, start_evt, end_evt in self._pending_stall_events:
                try:
                    stall_ms = start_evt.elapsed_time(end_evt)
                except RuntimeError:
                    stall_ms = 0.0
                stats = self._get_or_create_offload_group(group_name)
                stats.update_reload_stall(stall_ms)

            for group_name, start_evt, end_evt in self._pending_bwd_compute_events:
                try:
                    bwd_ms = start_evt.elapsed_time(end_evt)
                except RuntimeError:
                    bwd_ms = 0.0
                stats = self._get_or_create_offload_group(group_name)
                stats.update_backward_compute_time(bwd_ms)

            for group_name, start_evt, end_evt in self._pending_fwd_compute_events:
                try:
                    fwd_ms = start_evt.elapsed_time(end_evt)
                except RuntimeError:
                    fwd_ms = 0.0
                stats = self._get_or_create_offload_group(group_name)
                stats.update_forward_compute_time(fwd_ms)

            if self._diag_print_count < self._diag_print_limit:
                print(
                    f"[AdaptiveMemProfiler][DIAG] iter={iteration} "
                    f"pp_rank={self._pp_rank} "
                    f"processed {len(self._pending_stall_events)} stall events, "
                    f"{len(self._pending_bwd_compute_events)} bwd_compute events, "
                    f"{len(self._pending_fwd_compute_events)} fwd_compute events, "
                    f"offload_group_stats keys={list(self._offload_group_stats.keys())}",
                    flush=True,
                )
                self._diag_print_count += 1

            self._pending_stall_events.clear()
            self._pending_bwd_compute_events.clear()
            self._pending_fwd_compute_events.clear()

        # Profiling completes after collecting num_profile_iters worth of data.
        # Account for warmup_skip_iters: profiling starts at iteration
        # warmup_skip_iters, so it finishes at iteration
        # warmup_skip_iters + num_profile_iters - 1.
        profile_end_iter = self._warmup_skip_iters + self._num_profile_iters - 1
        if not self._profiling_done and iteration >= profile_end_iter:
            self._profiling_done = True
            self._last_optimize_iter = iteration
            # In per-PP-rank mode, each stall profile rank runs optimization
            # independently. In legacy mode, only the original profile_rank runs it.
            if self._per_pp_rank_enabled and self._pp_world_size > 1:
                if self._is_stall_profile_rank:
                    self._log_summary()
                    self._run_optimization()
            else:
                if self._is_profile_rank:
                    self._log_summary()
                    self._run_optimization()

    def is_profiling_active(self) -> bool:
        """Return True if profiling hooks should be active this iteration.

        Profiling is skipped during the first ``_warmup_skip_iters`` iterations
        to let the training pipeline reach a stable state (loss scaling, pipeline
        warmup) before injecting heavy synchronization points that could cause
        pipeline timing mismatches.
        """
        return (
            self._is_profile_rank
            and not self._profiling_done
            and self._current_iter >= self._warmup_skip_iters
        )

    def is_profiling_done(self) -> bool:
        """Return True once enough iterations have been collected."""
        return self._profiling_done

    # ------------------------------------------------------------------
    # Public API: per-offload-group reload stall profiling
    # ------------------------------------------------------------------

    def enqueue_stall_events(
        self,
        group_name: str,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
    ) -> None:
        """
        Enqueue a pair of CUDA Events for deferred stall measurement.

        Called from ``on_group_commit_backward`` during backward. The events
        bracket the ``wait_event`` call.  Actual ``elapsed_time`` is computed
        later in ``on_iteration_end`` to avoid blocking the backward pass.

        Parameters
        ----------
        group_name : str
            Name of the offload group.
        start_event : torch.cuda.Event
            Event recorded just before ``wait_event``.
        end_event : torch.cuda.Event
            Event recorded just after ``wait_event``.
        """
        self._pending_stall_events.append((group_name, start_event, end_event))

    def enqueue_backward_compute_events(
        self,
        group_name: str,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
    ) -> None:
        """
        Enqueue a pair of CUDA Events for deferred backward compute time measurement.

        Called from ``on_group_commit_backward`` (start) and
        ``on_group_start_backward`` (end). The interval between these two hooks
        is exactly the backward compute time of the group body.

        Parameters
        ----------
        group_name : str
            Name of the offload group.
        start_event : torch.cuda.Event
            Event recorded at the end of ``on_group_commit_backward``
            (after wait_event, so stall is excluded from the compute time).
        end_event : torch.cuda.Event
            Event recorded at the start of ``on_group_start_backward``
            (before h2d_stream.wait_stream, so next-group prefetch is excluded).
        """
        self._pending_bwd_compute_events.append((group_name, start_event, end_event))

    def enqueue_forward_compute_events(
        self,
        group_name: str,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
    ) -> None:
        """
        Enqueue a pair of CUDA Events for deferred forward compute time measurement.

        Called from ``on_group_start_forward`` (start) and
        ``on_group_commit_forward`` (end). The interval between these two hooks
        is the forward compute time of the group body — the direct cost of
        recomputing this group during backward.

        Parameters
        ----------
        group_name : str
            Name of the offload group.
        start_event : torch.cuda.Event
            Event recorded at ``on_group_start_forward``.
        end_event : torch.cuda.Event
            Event recorded at ``on_group_commit_forward``.
        """
        self._pending_fwd_compute_events.append((group_name, start_event, end_event))

    def record_group_reload_stall(self, group_name: str, stall_time_ms: float) -> None:
        """
        Record one sample of reload stall time for an offload group.

        Called from ChunkOffloadHandler to track the GPU stall time incurred
        when waiting for H2D reload to complete. This is measured:
        - In ``on_group_commit_backward``: via CUDA Events around ``wait_event``
          for groups that were async-prefetched (deferred via enqueue_stall_events).
        - In ``tensor_pop``: estimated from PCIe bandwidth for groups that fall
          back to synchronous reload (no async prefetch).

        Parameters
        ----------
        group_name : str
            Name of the offload group (e.g., "qkv_linear").
        stall_time_ms : float
            GPU stall time (ms).
        """
        # Use is_stall_profiling_active() so per-PP-rank stall recording works
        if not self.is_stall_profiling_active():
            return
        stats = self._get_or_create_offload_group(group_name)
        stats.update_reload_stall(stall_time_ms)
        if self._diag_print_count < self._diag_print_limit:
            print(
                f"[AdaptiveMemProfiler][DIAG] record_group_reload_stall: "
                f"pp_rank={self._pp_rank} group={group_name!r} stall={stall_time_ms:.3f}ms "
                f"sample_count={stats.sample_count}",
                flush=True,
            )
            self._diag_print_count += 1

    def record_group_offload_bytes(self, group_name: str, nbytes: int) -> None:
        """
        Record the total bytes offloaded for an offload group.

        Called from ChunkOffloadHandler during bulk_offload_group.

        Parameters
        ----------
        group_name : str
            Name of the offload group.
        nbytes : int
            Total bytes offloaded for this group.
        """
        # Use is_stall_profiling_active() so per-PP-rank bytes recording works
        if not self.is_stall_profiling_active():
            return
        stats = self._get_or_create_offload_group(group_name)
        if (
            stats.total_offload_bytes == 0
            and self._diag_print_count < self._diag_print_limit
        ):
            # Only log first time per group (bytes are constant across iterations)
            print(
                f"[AdaptiveMemProfiler][DIAG] record_group_offload_bytes: "
                f"pp_rank={self._pp_rank} group={group_name!r} "
                f"nbytes={nbytes} ({nbytes / 1024 / 1024:.1f}MB) "
                f"iter={self._current_iter}",
                flush=True,
            )
            self._diag_print_count += 1
        stats.update_offload_bytes(nbytes)

    # ------------------------------------------------------------------
    # Public API: per-module profiling context manager
    # ------------------------------------------------------------------

    class _ModuleProfileContext:
        """
        Context manager returned by ``profile_module()``.

        Measures:
        - Forward compute time (via CUDA Events)
        - Peak GPU memory delta during forward
        """

        def __init__(
            self,
            profiler: "AdaptiveMemoryProfiler",
            layer_number: int,
            module_name: str,
            is_moe: bool,
        ):
            self._profiler = profiler
            self._layer_number = layer_number
            self._module_name = module_name
            self._is_moe = is_moe

            self._cuda_timer = CudaTimer()
            self._mem_before: int = 0

        def __enter__(self):
            if not torch.cuda.is_available():
                return self
            # Synchronize only the default (compute) stream rather than ALL
            # streams.  The previous torch.cuda.synchronize() blocked until
            # d2h_stream and h2d_stream completed too, which slowed rank 0
            # relative to other ranks and disrupted pipeline timing.
            torch.cuda.current_stream().synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._mem_before = torch.cuda.memory_allocated()
            self._cuda_timer.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not torch.cuda.is_available():
                return False
            self._cuda_timer.__exit__(exc_type, exc_val, exc_tb)

            if exc_type is not None:
                # Don't record stats if an exception occurred
                return False

            # Same as __enter__: synchronize only the compute stream.
            torch.cuda.current_stream().synchronize()
            mem_peak = torch.cuda.max_memory_allocated()

            compute_time_ms = self._cuda_timer.elapsed_ms()
            peak_delta_bytes = max(0, mem_peak - self._mem_before)

            layer_stats = self._profiler._get_or_create_layer(
                self._layer_number, self._is_moe
            )
            module_stats = layer_stats.get_or_create(self._module_name)
            module_stats.update(compute_time_ms, peak_delta_bytes)

            return False

    def profile_module(
        self,
        layer_number: int,
        module_name: str,
        is_moe: bool = False,
    ) -> "_ModuleProfileContext":
        """
        Return a context manager that profiles a single sub-module forward pass.

        Parameters
        ----------
        layer_number : int
            Global layer index (as stored in ``TransformerLayer.layer_number``).
        module_name : str
            One of: "attn_norm", "attention", "mlp_norm", "mlp".
        is_moe : bool
            Whether this layer is a Mixture-of-Experts layer.

        Example
        -------
        ::

            with profiler.profile_module(self.layer_number, "mlp", self.is_moe_layer):
                mlp_output = self.mlp(pre_mlp_layernorm_output)
        """
        return self._ModuleProfileContext(self, layer_number, module_name, is_moe)

    # ------------------------------------------------------------------
    # Public API: query results
    # ------------------------------------------------------------------

    def get_layer_stats(self, layer_number: int) -> Optional[LayerStats]:
        """Return collected stats for a specific layer, or None if not available."""
        return self._layer_stats.get(layer_number)

    def get_all_layer_stats(self) -> Dict[int, LayerStats]:
        """Return all collected layer stats, sorted by layer number."""
        return dict(sorted(self._layer_stats.items()))

    def get_offload_group_stats(self) -> Dict[str, OffloadGroupStats]:
        """Return all collected per-offload-group stats."""
        return dict(self._offload_group_stats)

    def get_pcie_stats(self) -> Optional[PCIeBandwidthStats]:
        """Return measured PCIe bandwidth stats."""
        return self._pcie_stats

    def get_skip_offload_groups(self) -> Set[str]:
        """
        Return the set of group names that should NOT be offloaded (KEEP decision).

        Available after profiling completes and optimization runs.
        """
        return self._skip_offload_groups

    def get_recompute_groups(self) -> Set[str]:
        """
        Return the set of group names that should be recomputed during backward
        instead of offloaded (RECOMPUTE decision).

        Available after profiling completes and optimization runs.
        """
        return self._recompute_groups

    def is_optimization_applied(self) -> bool:
        """Return True if the optimization has been applied."""
        return self._optimization_applied

    def estimate_offload_time_ms(self, activation_bytes: int) -> float:
        """
        Estimate the time (ms) to offload ``activation_bytes`` bytes to CPU.

        Uses measured D2H bandwidth. Returns infinity if bandwidth is unknown.
        """
        if self._pcie_stats is None or not self._pcie_stats.measured:
            return float("inf")
        if self._pcie_stats.d2h_gbps <= 0:
            return float("inf")
        size_gb = activation_bytes / (1024**3)
        return size_gb / self._pcie_stats.d2h_gbps * 1000.0  # ms

    def estimate_reload_time_ms(self, activation_bytes: int) -> float:
        """
        Estimate the time (ms) to reload ``activation_bytes`` bytes from CPU.

        Uses measured H2D bandwidth. Returns infinity if bandwidth is unknown.
        """
        if self._pcie_stats is None or not self._pcie_stats.measured:
            return float("inf")
        if self._pcie_stats.h2d_gbps <= 0:
            return float("inf")
        size_gb = activation_bytes / (1024**3)
        return size_gb / self._pcie_stats.h2d_gbps * 1000.0  # ms

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _run_optimization(self) -> None:
        """
        Run the three-way offload/recompute optimization after profiling completes.

        Uses OffloadRecomputeOptimizer to determine:
        - skip_groups (KEEP): activations stay on GPU
        - recompute_groups (RECOMPUTE): forward is re-run during backward
        - remaining groups (OFFLOAD): default, activations are D2H/H2D

        Enhanced with:
        - Memory budget constraint (greedy knapsack for KEEP decisions)
        - Per-PP-rank awareness (each stage optimizes independently)
        - Num-layers scaling for memory budget
        """
        if not self._offload_group_stats:
            # Use print() instead of logger.warning() to ensure visibility
            # regardless of log level configuration.
            print(
                f"[AdaptiveMemoryProfiler] WARNING (pp_rank={self._pp_rank}): "
                "Cannot run optimization — "
                "no offload group stats were recorded during profiling. "
                "This likely means record_group_offload_bytes / enqueue_stall_events "
                "were never called on this rank.",
                flush=True,
            )
            return

        stall_threshold_ms = float(os.environ.get("OFFLOAD_STALL_THRESHOLD_MS", "1.0"))
        recompute_cost_ratio = float(
            os.environ.get(
                "OFFLOAD_RECOMPUTE_COST_RATIO", str(_DEFAULT_RECOMPUTE_COST_RATIO)
            )
        )

        # Auto-detect num_layers if not set externally
        num_layers = self._num_layers_on_this_rank
        if num_layers <= 0:
            # Estimate from the number of distinct layers in layer_stats
            num_layers = max(len(self._layer_stats), 1)

        optimizer = OffloadRecomputeOptimizer(
            group_stats=self._offload_group_stats,
            pcie_stats=self._pcie_stats,
            stall_threshold_ms=stall_threshold_ms,
            recompute_cost_ratio=recompute_cost_ratio,
            memory_budget_mb=max(0, self._memory_budget_mb),
            num_layers_on_this_rank=num_layers,
            pp_rank=self._pp_rank,
        )

        self._skip_offload_groups, self._recompute_groups = (
            optimizer.compute_decisions()
        )
        # NOTE: do NOT set _optimization_applied here!
        # _optimization_applied is set in apply_optimization_results() which is
        # a collective operation (allgather) that must be called by ALL ranks.
        # Setting it here would cause the stall-profile-rank to skip the
        # collective call in training.py, leading to a hang or silent skip.

    def apply_optimization_results(self) -> Tuple[Set[str], Set[str]]:
        """
        Called after profiling to apply optimization results.

        This is a COLLECTIVE operation — all ranks must call it together.

        In per-PP-rank mode, each PP stage's stall-profile-rank has independently
        computed decisions. These are gathered via allgather and distributed so
        that every rank applies the decisions corresponding to its own PP stage.

        In legacy mode, broadcasts from the single profile_rank to all ranks.

        Returns
        -------
        skip_groups : Set[str]
            Names of groups to keep on GPU (KEEP decision).
        recompute_groups : Set[str]
            Names of groups to recompute during backward (RECOMPUTE decision).
        """
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return self._skip_offload_groups, self._recompute_groups

        if self._per_pp_rank_enabled and self._pp_world_size > 1:
            return self._apply_per_pp_rank_results()
        else:
            return self._apply_legacy_broadcast_results()

    def _apply_legacy_broadcast_results(self) -> Tuple[Set[str], Set[str]]:
        """Legacy: single profile_rank broadcasts to all ranks."""
        if self._is_profile_rank:
            payload = [list(self._skip_offload_groups), list(self._recompute_groups)]
        else:
            payload = None

        object_list = [payload]
        torch.distributed.broadcast_object_list(object_list, src=self._profile_rank)
        received = object_list[0]
        if received:
            self._skip_offload_groups = set(received[0])
            self._recompute_groups = set(received[1])
        else:
            self._skip_offload_groups = set()
            self._recompute_groups = set()
        self._optimization_applied = True

        return self._skip_offload_groups, self._recompute_groups

    def _apply_per_pp_rank_results(self) -> Tuple[Set[str], Set[str]]:
        """
        Per-PP-rank broadcast: each PP stage's stall-profile-rank computed
        its own decisions. We use allgather to share per-PP-stage decisions
        across all ranks, then each rank picks the decisions for its own
        PP stage.

        Protocol:
        1. Each rank contributes its PP rank's decisions (or empty if not
           a stall-profile-rank).
        2. allgather_object collects all contributions.
        3. Each rank finds the decisions from the stall-profile-rank that
           shares its PP rank.
        """
        # Each rank provides: (pp_rank, skip_groups_list, recompute_groups_list)
        # Only stall-profile-ranks have valid data; others send sentinel.
        if self._is_stall_profile_rank:
            my_data = (
                self._pp_rank,
                list(self._skip_offload_groups),
                list(self._recompute_groups),
            )
        else:
            my_data = (self._pp_rank, None, None)  # no data

        # Allgather across all ranks
        world_size = torch.distributed.get_world_size()
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, my_data)

        # Build a map: pp_rank -> (skip_groups, recompute_groups)
        pp_decisions: Dict[int, Tuple[Set[str], Set[str]]] = {}
        for entry in gathered:
            if entry is not None:
                pp_r, skip_list, recompute_list = entry
                if skip_list is not None:
                    pp_decisions[pp_r] = (set(skip_list), set(recompute_list))

        # Apply decisions for my PP rank
        if self._pp_rank in pp_decisions:
            self._skip_offload_groups, self._recompute_groups = pp_decisions[
                self._pp_rank
            ]
        else:
            # Fallback: no decisions for my PP rank (shouldn't happen normally)
            self._skip_offload_groups = set()
            self._recompute_groups = set()

        self._optimization_applied = True

        global_rank = torch.distributed.get_rank()
        if global_rank < 8 or self._is_stall_profile_rank:
            print(
                f"[AdaptiveOffload][PP-APPLY] global_rank={global_rank} "
                f"pp_rank={self._pp_rank} "
                f"skip_groups={self._skip_offload_groups} "
                f"recompute_groups={self._recompute_groups} "
                f"(per-PP-rank decisions from {len(pp_decisions)} stages)",
                flush=True,
            )

        return self._skip_offload_groups, self._recompute_groups

    # --- Dynamic re-optimization ---

    def _start_reoptimization(self, iteration: int) -> None:
        """
        Begin a new profiling cycle for dynamic re-optimization.

        Resets the offload group stats and pending events, and marks
        profiling as not done so new data will be collected.
        """
        self._reoptimize_count += 1
        self._profiling_done = False
        self._optimization_applied = False
        # Reset measured stats for fresh data collection
        self._offload_group_stats.clear()
        self._pending_stall_events.clear()
        self._pending_bwd_compute_events.clear()
        self._pending_fwd_compute_events.clear()
        # Reset warmup to start profiling immediately (no skip for re-optimization)
        self._warmup_skip_iters = iteration
        # Keep existing decisions active until new ones are ready
        # (skip_offload_groups and recompute_groups are NOT cleared)

        if self._is_stall_profile_rank:
            global_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else -1
            )
            print(
                f"[AdaptiveMemProfiler][REOPT] Starting re-optimization #{self._reoptimize_count} "
                f"at iter={iteration} global_rank={global_rank} pp_rank={self._pp_rank}",
                flush=True,
            )

    # Backward-compatible alias
    def apply_skip_offload_groups(self) -> Set[str]:
        """Backward-compatible: broadcasts and returns only skip_groups."""
        skip_groups, _ = self.apply_optimization_results()
        return skip_groups

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_json(self, path: str) -> None:
        """
        Persist profiling results to a JSON file.

        Only the ``profile_rank`` writes the file to avoid conflicts.
        """
        if not self._is_profile_rank:
            return
        data = {
            "num_profile_iters": self._num_profile_iters,
            "pcie_stats": asdict(self._pcie_stats) if self._pcie_stats else None,
            "layer_stats": {
                str(layer_num): {
                    "layer_number": ls.layer_number,
                    "is_moe": ls.is_moe,
                    "modules": {
                        mod_name: asdict(mod_stats)
                        for mod_name, mod_stats in ls.modules.items()
                    },
                }
                for layer_num, ls in self.get_all_layer_stats().items()
            },
            "offload_group_stats": {
                name: asdict(stats) for name, stats in self._offload_group_stats.items()
            },
            "skip_offload_groups": list(self._skip_offload_groups),
            "recompute_groups": list(self._recompute_groups),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[AdaptiveMemoryProfiler] Saved profiling results to {path}")

    @classmethod
    def load_from_json(cls, path: str) -> "AdaptiveMemoryProfiler":
        """
        Load profiling results from a JSON file (for offline analysis).

        Returns a new (non-singleton) profiler instance with the loaded data.
        """
        with open(path, "r") as f:
            data = json.load(f)

        profiler = cls.__new__(cls)
        profiler._num_profile_iters = data["num_profile_iters"]
        profiler._profile_rank = 0
        profiler._measure_pcie = False
        profiler._current_iter = profiler._num_profile_iters
        profiler._profiling_done = True
        profiler._is_profile_rank = True
        profiler._lock = threading.Lock()
        profiler._skip_offload_groups = set(data.get("skip_offload_groups", []))
        profiler._recompute_groups = set(data.get("recompute_groups", []))
        profiler._optimization_applied = bool(
            profiler._skip_offload_groups or profiler._recompute_groups
        )
        profiler._pending_stall_events = []
        profiler._pending_bwd_compute_events = []
        profiler._diag_print_count = 0
        profiler._diag_print_limit = 20
        profiler._warmup_skip_iters = 0  # no warmup needed for offline analysis
        profiler._pcie_measured = True  # already loaded from JSON

        if data["pcie_stats"]:
            profiler._pcie_stats = PCIeBandwidthStats(**data["pcie_stats"])
        else:
            profiler._pcie_stats = None

        profiler._layer_stats = {}
        for layer_num_str, ls_data in data["layer_stats"].items():
            ls = LayerStats(
                layer_number=ls_data["layer_number"],
                is_moe=ls_data["is_moe"],
            )
            for mod_name, mod_data in ls_data["modules"].items():
                ls.modules[mod_name] = LayerModuleStats(**mod_data)
            profiler._layer_stats[int(layer_num_str)] = ls

        profiler._offload_group_stats = {}
        for name, stats_data in data.get("offload_group_stats", {}).items():
            profiler._offload_group_stats[name] = OffloadGroupStats(**stats_data)

        return profiler

    # ------------------------------------------------------------------
    # Logging / summary
    # ------------------------------------------------------------------

    def _log_summary(self) -> None:
        """Print a human-readable summary of collected profiling data."""
        # Diagnostic: show what offload group stats we have
        print(
            f"[AdaptiveMemProfiler][DIAG] _log_summary called. "
            f"offload_group_stats keys={list(self._offload_group_stats.keys())}, "
            f"layer_stats keys={list(self._layer_stats.keys())}, "
            f"is_profile_rank={self._is_profile_rank}, "
            f"pending_stall_events={len(self._pending_stall_events)}",
            flush=True,
        )
        lines = [
            "",
            "=" * 80,
            "[AdaptiveMemoryProfiler] Profiling Summary",
            f"  Iterations collected : {self._num_profile_iters}",
        ]
        if self._pcie_stats and self._pcie_stats.measured:
            lines += [
                f"  PCIe D2H bandwidth   : {self._pcie_stats.d2h_gbps:.2f} GB/s",
                f"  PCIe H2D bandwidth   : {self._pcie_stats.h2d_gbps:.2f} GB/s",
            ]
        lines.append(
            f"  {'Layer':>6}  {'MoE':>4}  "
            f"{'Module':<12}  {'Compute(ms)':>12}  {'PeakDelta(MB)':>14}"
        )
        lines.append("  " + "-" * 56)

        for layer_num, ls in self.get_all_layer_stats().items():
            for mod_name, ms in ls.modules.items():
                lines.append(
                    f"  {layer_num:>6}  {'Y' if ls.is_moe else 'N':>4}  "
                    f"{mod_name:<12}  "
                    f"{ms.compute_time_ms:>12.3f}  "
                    f"{ms.peak_memory_delta_bytes / (1024**2):>14.2f}"
                )

        # Also print per-offload-group stats with backward and forward compute time
        if self._offload_group_stats:
            lines.append("")
            lines.append(
                "  Per-Offload-Group Stats (Stall + Backward/Forward Compute Time):"
            )
            lines.append(
                f"  {'Group':<14} {'Stall(ms)':>10} {'BwdComp(ms)':>12} "
                f"{'FwdComp(ms)':>12} {'Offload(MB)':>12} {'Samples':>8}"
            )
            lines.append("  " + "-" * 76)
            for name, stats in sorted(self._offload_group_stats.items()):
                lines.append(
                    f"  {name:<14} {stats.reload_stall_time_ms:>10.3f} "
                    f"{stats.backward_compute_time_ms:>12.3f} "
                    f"{stats.forward_compute_time_ms:>12.3f} "
                    f"{stats.total_offload_bytes / (1024**2):>12.1f} "
                    f"{stats.sample_count:>8}"
                )

        lines.append("=" * 80)
        logger.info("\n".join(lines))
        # Also print to stdout so it appears in training logs
        print("\n".join(lines), flush=True)
