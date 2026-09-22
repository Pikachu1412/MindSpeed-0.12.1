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
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed

from mindspeed.core.pipeline_parallel.adaptive_offload.transport_host_profile import estimate_host_costs
from mindspeed.core.pipeline_parallel.adaptive_offload.unified_offload_optimizer import (
    BACKWARD_ORDER,
    MODULE_GROUPS,
    GroupProfile,
    MemoryBudget,
    ModuleProfile,
    NoFeasiblePlanError,
    ScheduleConfig,
    StrategyPlan,
    UnifiedOffloadOptimizer,
)

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
    input_bytes: int = 0
    recompute_time_ms: float = 0.0
    recompute_sample_count: int = 0

    def update(self, compute_time_ms: float, peak_memory_delta_bytes: int, input_bytes: int = 0) -> None:
        """Accumulate one sample using a running average."""
        n = self.sample_count
        self.compute_time_ms = (self.compute_time_ms * n + compute_time_ms) / (n + 1)
        self.peak_memory_delta_bytes = max(self.peak_memory_delta_bytes, peak_memory_delta_bytes)
        self.input_bytes = max(self.input_bytes, input_bytes)
        self.sample_count += 1

    def update_recompute(self, duration_ms: float, peak_bytes: int, input_bytes: int) -> None:
        count = self.recompute_sample_count
        self.recompute_time_ms = (self.recompute_time_ms * count + duration_ms) / (count + 1)
        self.recompute_sample_count += 1
        self.peak_memory_delta_bytes = max(self.peak_memory_delta_bytes, peak_bytes)
        self.input_bytes = max(self.input_bytes, input_bytes)


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
    logical_saved_bytes: int = 0
    aliased_saved_bytes: int = 0
    non_offloadable_bytes: int = 0
    keep_storage_bytes: int = 0
    resident_storage_bytes: int = 0
    storage_footprint_sample_count: int = 0
    storage_footprint_incomplete: bool = False
    d2h_storage_ratio: float = 1.0
    sample_count: int = 0
    backward_compute_sample_count: int = 0
    forward_compute_sample_count: int = 0
    d2h_time_ms: float = 0.0
    h2d_time_ms: float = 0.0
    d2h_sample_count: int = 0
    h2d_sample_count: int = 0
    released_bytes: int = 0
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

    def update_transfer(self, direction: str, duration_ms: float) -> None:
        count_name = f"{direction}_sample_count"
        time_name = f"{direction}_time_ms"
        count = getattr(self, count_name)
        setattr(self, time_name, (getattr(self, time_name) * count + duration_ms) / (count + 1))
        setattr(self, count_name, count + 1)


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
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
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


def _memory_budget_from_telemetry(memory, extra_budget_mb=0.0, *, planning=False):
    capacity_mb = memory["capacity_bytes"] / (1024 ** 2)
    reserve_mb = max(
        float(os.environ.get("ADAPTIVE_MEM_RESERVE_MB", "1024")),
        capacity_mb * float(os.environ.get("ADAPTIVE_MEM_RESERVE_FRACTION", "0.05")),
    )
    if planning:
        capacity_mb = min(capacity_mb, memory.get("planning_capacity_bytes", memory["capacity_bytes"]) / (1024 ** 2))
    return MemoryBudget(
        baseline_peak_mb=memory["baseline_peak_bytes"] / (1024 ** 2),
        capacity_mb=capacity_mb,
        reserve_mb=reserve_mb,
        extra_budget_mb=extra_budget_mb if extra_budget_mb > 0 else None,
        activation_margin=float(os.environ.get("ADAPTIVE_MEM_ACTIVATION_MARGIN", "1.1")),
        allocator_unavailable_mb=memory.get("allocator_peak_overhead_bytes", 0) / (1024 ** 2),
        activation_baseline_peak_mb=(memory['activation_baseline_peak_bytes'] / 1024 ** 2
                                     if memory.get('activation_baseline_peak_bytes', 0) > 0 else None),
    )


class OffloadRecomputeOptimizer:
    """Compatibility facade over the single joint, memory-constrained solver."""

    def __init__(
        self,
        group_stats: Dict[str, OffloadGroupStats],
        pcie_stats: Optional[PCIeBandwidthStats],
        stall_threshold_ms: float = 1.0,
        recompute_cost_ratio: float = _DEFAULT_RECOMPUTE_COST_RATIO,
        memory_budget_mb: float = 0.0,
        num_layers_on_this_rank: int = 1,
        pp_rank: int = -1,
        *,
        layer_stats: Optional[Dict[int, LayerStats]] = None,
        memory_telemetry: Optional[Dict] = None,
        schedule_options: Optional[Dict] = None,
    ):
        self._group_stats = group_stats
        self._pcie_stats = pcie_stats
        self._memory_budget_mb = memory_budget_mb
        self._num_layers = num_layers_on_this_rank
        self._pp_rank = pp_rank
        self._layer_stats = layer_stats or {}
        self._memory = memory_telemetry or {}
        self._schedule_options = schedule_options or {}
        self.plan = None

    def compute_decisions(self) -> Tuple[Set[str], Set[str]]:
        if not self._memory.get("samples"):
            raise RuntimeError("Unified offload requires measured peak memory and device capacity")
        groups = {}
        for name in BACKWARD_ORDER:
            stats = self._group_stats.get(name)
            if stats is None or stats.total_offload_bytes + stats.non_offloadable_bytes <= 0:
                continue
            transfer_times = {}
            for direction in ("d2h", "h2d"):
                duration = getattr(stats, f"{direction}_time_ms")
                samples = getattr(stats, f"{direction}_sample_count")
                bandwidth = getattr(self._pcie_stats, f"{direction}_gbps", 0.0)
                if samples and duration >= 0:
                    transfer_times[direction] = max(duration, 0.001)
                elif bandwidth > 0:
                    transfer_times[direction] = stats.total_offload_bytes / (1024 ** 3) / bandwidth * 1000
                else:
                    raise RuntimeError(f"Missing {direction} transfer measurements for {name}")
            groups[name] = GroupProfile(
                activation_mb=stats.total_offload_bytes / (1024 ** 2),
                forward_ms=stats.forward_compute_time_ms,
                backward_ms=stats.backward_compute_time_ms,
                d2h_ms=transfer_times["d2h"],
                h2d_ms=transfer_times["h2d"],
                released_mb=stats.released_bytes / (1024 ** 2),
            )
        modules = {}
        for name in MODULE_GROUPS:
            samples = [layer.modules[name] for layer in self._layer_stats.values()
                       if name in layer.modules and layer.modules[name].sample_count >= 2
                       and layer.modules[name].input_bytes > 0]
            if samples:
                modules[name] = ModuleProfile(
                    recompute_ms=max(sample.recompute_time_ms if sample.recompute_sample_count >= 2
                                     else sample.compute_time_ms for sample in samples),
                    input_mb=max(sample.input_bytes for sample in samples) / (1024 ** 2),
                    workspace_mb=max(sample.peak_memory_delta_bytes for sample in samples) / (1024 ** 2)
                    + sum(groups[group].activation_mb for group in MODULE_GROUPS[name] if group in groups),
                )
        if (self._layer_stats and any(group in groups for group in MODULE_GROUPS["mlp"])
                and not any(layer.is_moe for layer in self._layer_stats.values())):
            raise ValueError("MoE activation groups were measured, but no profiled layer is marked as MoE")
        layers = []
        for layer_number, layer in sorted(self._layer_stats.items()):
            layers.append(tuple(group for group in BACKWARD_ORDER if group in groups and (
                layer.is_moe or group not in MODULE_GROUPS["mlp"]
            )))
        if not layers:
            layers = [tuple(groups)] * max(self._num_layers, 1)
        memory = _memory_budget_from_telemetry(self._memory, self._memory_budget_mb)
        schedule = ScheduleConfig(
            layers=tuple(layers),
            inflight_microbatches=max(1, self._memory["max_inflight_microbatches"]),
            recompute_factor=float(os.environ.get("ADAPTIVE_RECOMPUTE_FACTOR", "1.25")),
            transfer_factor=float(os.environ.get("ADAPTIVE_TRANSFER_FACTOR", "1.1")),
            **self._schedule_options,
        )
        self.plan = UnifiedOffloadOptimizer(groups, modules, memory, schedule).solve()
        return self.plan.keep_groups, self.plan.recompute_groups

    def compute_skip_groups(self) -> Set[str]:
        return self.compute_decisions()[0]


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
    4. Call ``on_iteration_end`` to resolve events and collect memory telemetry.
    5. Once profiling is done, all ranks call ``apply_optimization_results``.
    6. Optionally call ``save_to_json(path)`` to persist measurements and the plan.

    Parameters
    ----------
    num_profile_iters : int
        Number of training iterations to collect profiling data.
    profile_rank : int
        Source rank when per-PP profiling is disabled. Per-PP mode measures all
        ranks and merges conservative statistics within each stage.
    measure_pcie : bool
        Whether to benchmark PCIe bandwidth in the first profiling iteration.
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
        _budget_env = os.environ.get("ADAPTIVE_MEM_BUDGET_MB", "auto")
        if _budget_env.lower() == "auto":
            self._memory_budget_mb = 0.0
        else:
            self._memory_budget_mb = float(_budget_env)
        if not math.isfinite(self._memory_budget_mb) or self._memory_budget_mb < 0:
            raise ValueError("ADAPTIVE_MEM_BUDGET_MB must be auto or a nonnegative number")

        # --- Dynamic re-optimization ---
        # Re-profile every N iterations after the initial optimization.
        # 0 = disabled (one-shot optimization).
        self._reoptimize_interval: int = int(
            os.environ.get("ADAPTIVE_MEM_REOPTIMIZE_INTERVAL", "0")
        )
        self._last_optimize_iter: int = -1
        self._last_profile_iter: int = -1
        self._profile_refresh_pending: bool = False
        self._host_cost_feedback_keys: Set[str] = set()
        self._host_cost_feedback_requests: int = 0
        self._transfer_cost_feedback_requests: int = 0
        self._auto_transfer_feedback = None
        self._reoptimize_count: int = 0

        # Per-layer statistics: layer_number -> LayerStats
        self._layer_stats: Dict[int, LayerStats] = {}
        self._auto_module_specs = {}
        self._auto_child_profiles = {}
        self._pending_child_profiles = []

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
        self._pending_module_events = []
        self._pending_transfer_events = []
        self._iteration_peak_bytes = 0
        self._iteration_reserved_peak_bytes = 0
        self._activation_phase_peak_bytes = 0
        self._iteration_footprint_peak_bytes = 0
        self._observed_peak_allocated_bytes = 0
        self._observed_peak_footprint_bytes = 0
        self._allocator_snapshot = {}
        self._plan = None
        self._plan_baseline_bytes = 0
        self._plan_activation_baseline_bytes = 0
        from .microbatch_liveness import MicrobatchLiveness
        self._microbatch_liveness = MicrobatchLiveness(self.record_inflight_microbatches)
        self._live_microbatches = self._microbatch_liveness.live
        self._enabled = os.environ.get("MEGATRON_ADAPTIVE_OFFLOAD", "0") == "1"
        self._memory_pressure = False
        self._capacity_observations = {}
        self._memory_guard_interval = max(1, int(os.environ.get("ADAPTIVE_MEM_GUARD_INTERVAL", "1")))
        self._memory_telemetry = {
            "baseline_peak_bytes": 0,
            "activation_baseline_peak_bytes": 0,
            "capacity_bytes": 0,
            "allocator_unavailable_bytes": 0,
            "allocator_stats_samples": 0,
            "allocator_peak_overhead_bytes": 0,
            "observed_peak_allocated_bytes": 0,
            "observed_peak_footprint_bytes": 0,
            "max_inflight_microbatches": 1,
            "samples": 0,
        }

        # Optimization result: set of group names to recompute (instead of offload/keep)
        self._recompute_groups: Set[str] = set()

        # PCIe bandwidth
        self._pcie_stats: Optional[PCIeBandwidthStats] = None

        # Iteration tracking
        self._current_iter: int = 0
        self._profiling_done: bool = False

        # Whether this rank should do profiling (module profiling — expensive)
        self._is_profile_rank: bool = self._check_is_profile_rank()
        self._is_stall_profile_rank = self._is_profile_rank

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
        self._auto_calibration_iteration: Optional[int] = None
        self._auto_calibration_telemetry: Optional[dict] = None

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

        self._is_stall_profile_rank = self._per_pp_rank_enabled or self._is_profile_rank

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

        Transfer, group, and module events are resolved at iteration end. Per-PP
        mode samples all ranks so capacity and routing skew are not hidden.
        """
        if not self._enabled or self._profiling_done:
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
        self._layer_stats[layer_number].is_moe |= is_moe
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
        self._current_iter = iteration
        self._microbatch_liveness.start_iteration()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._iteration_peak_bytes = 0
        self._activation_phase_peak_bytes = 0
        self._iteration_reserved_peak_bytes = 0
        self._iteration_footprint_peak_bytes = 0
        self._capture_memory_peak(reset=True)
        self._sample_memory_capacity()
        calibration_iteration = getattr(self, '_auto_calibration_iteration', None)
        if calibration_iteration is not None and iteration >= calibration_iteration:
            from .bandwidth_calibration import coordinate_bandwidth_calibration

            if not self._pp_rank_initialized:
                self._refresh_is_profile_rank()
            eligible = (not self._profiling_done and not self._pcie_measured
                        and self._is_stall_profile_rank and self._measure_pcie
                        and torch.cuda.is_available())
            stats, telemetry = coordinate_bandwidth_calibration(
                self._measure_pcie_bandwidth_safe, eligible, torch)
            self._auto_calibration_iteration = None
            self._auto_calibration_telemetry = dict(telemetry, iteration=iteration)
            if eligible:
                self._pcie_stats = stats
                self._pcie_measured = True
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        if self._profiling_done:
            return

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
            torch.cuda.reset_peak_memory_stats()

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
        self._finish_memory_sample()
        if self._profiling_done:
            return

        # --- Deferred measurement: stall + backward/forward compute time ---
        # During forward/backward, CUDA Event pairs were recorded but NOT
        # synchronized (to avoid blocking the compute path). Now we synchronize
        # once and read all elapsed times.
        has_pending = (
            self._pending_stall_events
            or self._pending_bwd_compute_events
            or self._pending_fwd_compute_events
            or self._pending_module_events
            or self._pending_transfer_events
            or self._pending_child_profiles
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

            for group_name, start_evt, end_evt, excluded in self._pending_bwd_compute_events:
                try:
                    bwd_ms = max(0.0, start_evt.elapsed_time(end_evt) - sum(
                        start.elapsed_time(end) for start, end in excluded
                    ))
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

            for layer_number, name, is_moe, recompute, start_evt, end_evt, peak_bytes, input_bytes in self._pending_module_events:
                elapsed = start_evt.elapsed_time(end_evt)
                stats = self._get_or_create_layer(layer_number, is_moe).get_or_create(name)
                update = stats.update_recompute if recompute else stats.update
                update(elapsed, peak_bytes, input_bytes)
                if name in self._auto_module_specs and not recompute and not any(event[0] == name for event in self._pending_fwd_compute_events):
                    self._get_or_create_offload_group(name).update_forward_compute_time(elapsed)
            for name, direction, start_evt, end_evt in self._pending_transfer_events:
                self._get_or_create_offload_group(name).update_transfer(
                    direction, start_evt.elapsed_time(end_evt)
                )

            if self._pending_child_profiles:
                from .dense_activation_boundary import finish_child_observations
                finish_child_observations(self)

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
            self._pending_module_events.clear()
            self._pending_transfer_events.clear()

        # Profiling completes after collecting num_profile_iters worth of data.
        # Account for warmup_skip_iters: profiling starts at iteration
        # warmup_skip_iters, so it finishes at iteration
        # warmup_skip_iters + num_profile_iters - 1.
        profile_end_iter = self._warmup_skip_iters + self._num_profile_iters - 1
        if not self._profiling_done and iteration >= profile_end_iter:
            self._profiling_done = True
            self._profile_refresh_pending = True
            self._last_optimize_iter = iteration
            # In per-PP-rank mode, each stall profile rank runs optimization
            # independently. In legacy mode, only the original profile_rank runs it.
            if self._per_pp_rank_enabled and self._pp_world_size > 1:
                if self._is_stall_profile_rank:
                    self._log_summary()
            else:
                if self._is_profile_rank:
                    self._log_summary()

    def is_profiling_active(self) -> bool:
        """Return True if profiling hooks should be active this iteration.

        Profiling is skipped during the first ``_warmup_skip_iters`` iterations
        to let the training pipeline reach a stable state (loss scaling, pipeline
        warmup) before injecting heavy synchronization points that could cause
        pipeline timing mismatches.
        """
        return (
            self._enabled
            and self._is_stall_profile_rank
            and not self._profiling_done
            and self._current_iter >= self._warmup_skip_iters
        )

    def is_profiling_done(self) -> bool:
        """Return True once enough iterations have been collected."""
        return self._profiling_done

    def _capture_memory_peak(self, reset=False) -> Optional[Dict[str, int]]:
        if torch.cuda.is_available():
            try:
                stats = torch.cuda.memory_stats()
            except (AttributeError, RuntimeError):
                stats = {}
            peak_bytes = stats.get("allocated_bytes.all.peak") if isinstance(stats, dict) else None
            if peak_bytes is None:
                peak_bytes = torch.cuda.max_memory_allocated()
            self._iteration_peak_bytes = max(self._iteration_peak_bytes, peak_bytes)
            if isinstance(stats, dict) and "inactive_split_bytes.all.current" in stats:
                inactive = max(stats["inactive_split_bytes.all.current"],
                               stats.get("inactive_split_bytes.all.peak", 0))
                pending = max(0, stats.get("active_bytes.all.current", 0)
                              - stats.get("allocated_bytes.all.current", 0))
                self._allocator_snapshot = {
                    "inactive_split_current_bytes": stats["inactive_split_bytes.all.current"],
                    "inactive_split_peak_bytes": inactive,
                    "pending_free_bytes": pending,
                    "reserved_bytes": stats.get("reserved_bytes.all.current", 0),
                    "allocated_bytes": stats.get("allocated_bytes.all.current", 0),
                }
                self._memory_telemetry["allocator_unavailable_bytes"] = max(
                    self._memory_telemetry.get("allocator_unavailable_bytes", 0), inactive + pending
                )
                self._memory_telemetry["allocator_stats_samples"] = self._memory_telemetry.get("allocator_stats_samples", 0) + 1
                self._iteration_reserved_peak_bytes = max(
                    self._iteration_reserved_peak_bytes, stats.get("reserved_bytes.all.peak", 0)
                )
                self._iteration_footprint_peak_bytes = max(
                    self._iteration_footprint_peak_bytes,
                    stats.get("active_bytes.all.current", 0) + stats["inactive_split_bytes.all.current"],
                )
            self._iteration_footprint_peak_bytes = max(self._iteration_footprint_peak_bytes, self._iteration_peak_bytes)
            self._observed_peak_allocated_bytes = max(self._observed_peak_allocated_bytes, self._iteration_peak_bytes)
            self._observed_peak_footprint_bytes = max(self._observed_peak_footprint_bytes, self._iteration_footprint_peak_bytes)
            self._memory_telemetry["observed_peak_allocated_bytes"] = self._observed_peak_allocated_bytes
            self._memory_telemetry["observed_peak_footprint_bytes"] = self._observed_peak_footprint_bytes
            self._memory_telemetry["allocator_peak_overhead_bytes"] = max(
                0, self._observed_peak_footprint_bytes - self._observed_peak_allocated_bytes
            )
            if reset:
                torch.cuda.reset_peak_memory_stats()
            return stats if isinstance(stats, dict) else {}

    def _current_memory_limit_mb(self) -> float:
        limit = _memory_budget_from_telemetry(self._memory_telemetry, self._memory_budget_mb).limit_mb
        return min(limit, self._plan.memory_limit_mb) if self._plan is not None else limit

    def _sample_memory_capacity(self) -> None:
        if not torch.cuda.is_available():
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        capacity = min(total_bytes, free_bytes + torch.cuda.memory_reserved())
        previous = self._memory_telemetry["capacity_bytes"]
        if os.environ.get("AUTO_ACTIVATION_MEMORY") == "1" and previous and abs(capacity - previous) > previous * 0.02:
            self._capacity_observations.clear()
        if os.environ.get('AUTO_ACTIVATION_MEMORY') == '1':
            self._memory_telemetry['capacity_bytes'] = capacity
            if self._plan is not None and previous and capacity > previous * 1.02:
                self._memory_pressure = True
        else:
            self._memory_telemetry["capacity_bytes"] = min(previous, capacity) if previous else capacity
        if self._plan is not None:
            self._memory_pressure |= self._plan.predicted_peak_mb > self._current_memory_limit_mb()

        if os.environ.get("AUTO_ACTIVATION_MEMORY") == "1":
            self._record_capacity_observation(capacity)

    def _record_capacity_observation(self, capacity) -> None:
        if self._current_iter < self._warmup_skip_iters:
            return
        observed = self._capacity_observations.setdefault(self._current_iter, [capacity, capacity])
        observed[0] = min(observed[0], capacity)
        observed[1] = max(observed[1], capacity)
        oldest = self._current_iter - max(1, self._num_profile_iters) + 1
        for iteration in tuple(self._capacity_observations):
            if iteration < oldest:
                self._capacity_observations.pop(iteration)
        minimum = min(value[0] for value in self._capacity_observations.values())
        maximum = max(value[1] for value in self._capacity_observations.values())
        self._memory_telemetry["planning_capacity_bytes"] = max(0, minimum - (maximum - minimum))
        self._memory_telemetry["capacity_variation_bytes"] = maximum - minimum
        self._memory_telemetry["capacity_window_iterations"] = len(self._capacity_observations)

    def _finish_memory_sample(self) -> None:
        if not torch.cuda.is_available():
            return
        self._capture_memory_peak()
        self._sample_memory_capacity()
        memory = self._memory_telemetry
        if self._plan is None:
            memory["baseline_peak_bytes"] = max(memory["baseline_peak_bytes"], self._iteration_peak_bytes)
            memory['activation_baseline_peak_bytes'] = max(memory.get('activation_baseline_peak_bytes', 0),
                                                         getattr(self, '_activation_phase_peak_bytes', 0))
        else:
            excess = max(0, self._iteration_peak_bytes - int(self._plan.predicted_peak_mb * 1024 ** 2))
            memory["baseline_peak_bytes"] = max(
                memory["baseline_peak_bytes"], self._plan_baseline_bytes + excess
            )
            if memory.get('activation_baseline_peak_bytes', 0):
                memory['activation_baseline_peak_bytes'] = max(memory['activation_baseline_peak_bytes'],
                                                             self._plan_activation_baseline_bytes + excess)
            self._memory_pressure |= (
                self._plan.predicted_peak_mb + excess / (1024 ** 2) > self._current_memory_limit_mb()
            )
        memory["samples"] += 1
        self._write_iteration_trace()

    def end_activation_phase(self) -> None:
        self._capture_memory_peak()
        self._activation_phase_peak_bytes = max(getattr(self, '_activation_phase_peak_bytes', 0), self._iteration_peak_bytes)

    def _write_iteration_trace(self) -> None:
        prefix = os.environ.get("ADAPTIVE_MEM_ITERATION_JSONL")
        if not prefix:
            return
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        path = f"{prefix}.pp{self._pp_rank}.rank{global_rank}.jsonl"
        record = {
            "iteration_zero_based": self._current_iter,
            "pp_rank": self._pp_rank,
            "rank": global_rank,
            "peak_allocated_bytes": self._iteration_peak_bytes,
            "activation_phase_peak_bytes": getattr(self, '_activation_phase_peak_bytes', 0),
            "peak_reserved_bytes": self._iteration_reserved_peak_bytes,
            "observed_footprint_peak_bytes": self._iteration_footprint_peak_bytes,
            "memory": dict(self._memory_telemetry),
            "microbatch_liveness": self._microbatch_liveness.snapshot(),
            "allocator_snapshot": dict(self._allocator_snapshot),
            "memory_limit_mb": self._current_memory_limit_mb(),
            "plan_applied": self._optimization_applied,
            "plan": self._plan.to_dict() if self._plan is not None else None,
            "memory_pressure": self._memory_pressure,
            "policy_feedback": {
                "profile_anchor_iteration": self._last_profile_iter,
                "last_solve_iteration": self._last_optimize_iter,
                "host_feedback_requests": self._host_cost_feedback_requests,
                "consumed_host_cost_keys": sorted(self._host_cost_feedback_keys),
                "transfer_feedback_requests": self._transfer_cost_feedback_requests,
                "transfer_probe": (self._auto_transfer_feedback.last_probe
                                   if self._auto_transfer_feedback is not None else None),
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    def record_inflight_microbatches(self, count: int) -> None:
        self._memory_telemetry["max_inflight_microbatches"] = max(
            self._memory_telemetry["max_inflight_microbatches"], count
        )

    def begin_microbatch(self, token) -> None:
        self._microbatch_liveness.begin(token)

    def track_microbatch_backward(self, token, output):
        return self._microbatch_liveness.track(token, output)

    def enqueue_transfer_events(self, name, direction, start, end) -> None:
        if self.is_stall_profiling_active():
            self._pending_transfer_events.append((name, direction, start, end))

    def _measured_host_cost_keys(self) -> Set[str]:
        return {key for key, values in getattr(self, '_transport_host_costs', {}).items()
                if values.get('sample_count', 0) >= 2}

    def _host_cost_feedback_ready(self) -> bool:
        if (os.environ.get('AUTO_ACTIVATION_MEMORY') != '1' or not self._profiling_done
                or not self._optimization_applied or self._plan is None
                or 'OFFLOAD' not in self._plan.decisions.values()):
            return False
        return bool(self._measured_host_cost_keys() - self._host_cost_feedback_keys)

    def _transfer_cost_feedback_ready(self, iteration: int) -> bool:
        feedback = getattr(self, '_auto_transfer_feedback', None)
        if (os.environ.get('AUTO_ACTIVATION_MEMORY') != '1' or not self._profiling_done
                or not self._optimization_applied or self._plan is None or feedback is None):
            return False
        remaining = 32
        if self._reoptimize_interval > 0 and self._last_profile_iter >= 0:
            remaining = min(remaining, self._last_profile_iter + self._reoptimize_interval - iteration)
        end = getattr(self, '_auto_training_end_iteration', 0)
        if end > 0:
            remaining = min(remaining, end - iteration)
        return feedback.ready(self._offload_group_stats, iteration, max(0, remaining))

    def synchronize_memory_guard(self, iteration: int) -> None:
        """Collectively request a new safe plan before a subsequent training step."""
        if (not self._profiling_done and self._plan is None) or iteration % self._memory_guard_interval:
            return
        host_feedback = not self._memory_pressure and self._host_cost_feedback_ready()
        transfer_feedback = not self._memory_pressure and not host_feedback and self._transfer_cost_feedback_ready(iteration)
        pressure = int(self._memory_pressure or host_feedback or transfer_feedback)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            flag = torch.tensor([pressure], device="cuda", dtype=torch.int32)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            pressure = int(flag.item())
        if pressure:
            self._host_cost_feedback_requests += int(host_feedback)
            self._transfer_cost_feedback_requests = getattr(self, '_transfer_cost_feedback_requests', 0) + int(transfer_feedback)
            self._profiling_done = True
            self._optimization_applied = False
            self._memory_pressure = False
        elif self._profiling_done and self._reoptimize_interval > 0:
            auto_enabled = os.environ.get('AUTO_ACTIVATION_MEMORY') == '1'
            if auto_enabled and not self._optimization_applied:
                return
            profile_iteration = (self._last_profile_iter if auto_enabled
                                 and self._last_profile_iter >= 0 else self._last_optimize_iter)
            if iteration - profile_iteration >= self._reoptimize_interval:
                self._start_reoptimization(iteration)

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
        excluded_events=(),
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
        self._pending_bwd_compute_events.append((group_name, start_event, end_event, tuple(excluded_events)))

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

    def record_group_offload_bytes(self, group_name: str, nbytes: int, released_bytes: int = 0,
                                  logical_bytes: Optional[int] = None) -> None:
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
        logical_bytes = nbytes if logical_bytes is None else logical_bytes
        stats.logical_saved_bytes = max(stats.logical_saved_bytes, logical_bytes)
        stats.aliased_saved_bytes = max(stats.aliased_saved_bytes, logical_bytes - nbytes)
        stats.released_bytes = max(stats.released_bytes, released_bytes)

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
            input_tensor=None,
            recompute=False,
        ):
            self._profiler = profiler
            self._layer_number = layer_number
            self._module_name = module_name
            self._is_moe = is_moe
            self._recompute = recompute

            self._cuda_timer = CudaTimer()
            self._mem_before: int = 0
            inputs = input_tensor if isinstance(input_tensor, (tuple, list)) else (input_tensor,)
            self._input_bytes = sum(tensor.numel() * tensor.element_size() for tensor in inputs if isinstance(tensor, torch.Tensor))

        def __enter__(self):
            if not torch.cuda.is_available():
                return self
            stats = self._profiler._capture_memory_peak(reset=True)
            current_bytes = stats.get("allocated_bytes.all.current") if isinstance(stats, dict) else None
            self._mem_before = torch.cuda.memory_allocated() if current_bytes is None else current_bytes
            self._cuda_timer.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not torch.cuda.is_available():
                return False
            self._cuda_timer.__exit__(exc_type, exc_val, exc_tb)

            if exc_type is not None:
                # Don't record stats if an exception occurred
                return False

            stats = self._profiler._capture_memory_peak()
            peak_bytes = stats.get("allocated_bytes.all.peak") if isinstance(stats, dict) else None
            mem_peak = torch.cuda.max_memory_allocated() if peak_bytes is None else peak_bytes
            peak_delta_bytes = max(0, mem_peak - self._mem_before)
            self._profiler._pending_module_events.append((
                self._layer_number, self._module_name, self._is_moe, self._recompute,
                self._cuda_timer._start, self._cuda_timer._end,
                peak_delta_bytes, self._input_bytes,
            ))

            return False

    def profile_module(
        self,
        layer_number: int,
        module_name: str,
        is_moe: bool = False,
        input_tensor=None,
        recompute=False,
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
        return self._ModuleProfileContext(self, layer_number, module_name, is_moe, input_tensor, recompute)

    def profile_checkpoint_callable(self, function, layer_number, module_name, is_moe=False):
        """Measure the backward replay of the actual checkpoint call boundary."""
        if not self.is_profiling_active():
            return function
        called = False

        def wrapped(*args, **kwargs):
            nonlocal called
            recompute = called
            called = True
            if recompute and self.is_profiling_active():
                with self.profile_module(layer_number, module_name, is_moe, args[0], recompute=True):
                    return function(*args, **kwargs)
            return function(*args, **kwargs)

        return wrapped

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
        from mindspeed.core.pipeline_parallel.adaptive_offload.fine_grained_activation_offload import (
            H2D_PREFETCH_ENABLED, CROSS_LAYER_PREFETCH_ENABLED,
            LAST_LAYER_NO_OFFLOAD_ENABLED, PREFETCH_DEPTH,
        )

        if os.environ.get('AUTO_ACTIVATION_MEMORY') == '1':
            self._plan = self._solve_auto_modules()
            self._skip_offload_groups = self._plan.keep_groups
            self._recompute_groups = self._plan.recompute_groups
            self._plan_baseline_bytes = self._memory_telemetry['baseline_peak_bytes']
            self._plan_activation_baseline_bytes = self._memory_telemetry.get('activation_baseline_peak_bytes', 0)
            print(f'[AutoActivationMemory][JOINT-PLAN] pp_rank={self._pp_rank} '
                  f'{json.dumps(self._plan.to_dict(), sort_keys=True)}', flush=True)
            return

        optimizer = OffloadRecomputeOptimizer(
            group_stats=self._offload_group_stats,
            pcie_stats=self._pcie_stats,
            memory_budget_mb=self._memory_budget_mb,
            num_layers_on_this_rank=max(self._num_layers_on_this_rank, 1),
            pp_rank=self._pp_rank,
            layer_stats=self._layer_stats,
            memory_telemetry=self._memory_telemetry,
            schedule_options={
                "adjacent_prefetch": PREFETCH_DEPTH if H2D_PREFETCH_ENABLED else 0,
                "cross_layer_prefetch": CROSS_LAYER_PREFETCH_ENABLED,
                "last_layer_keep": LAST_LAYER_NO_OFFLOAD_ENABLED,
            },
        )
        self._skip_offload_groups, self._recompute_groups = optimizer.compute_decisions()
        self._plan = optimizer.plan
        self._plan_baseline_bytes = self._memory_telemetry["baseline_peak_bytes"]
        if self._is_stall_profile_rank:
            print(f"[AdaptiveOffload][JOINT-PLAN] pp_rank={self._pp_rank} "
                  f"{json.dumps(self._plan.to_dict(), sort_keys=True)}", flush=True)

    def _profile_payload(self):
        return {
            "pp_rank": self._pp_rank,
            "memory": dict(self._memory_telemetry),
            "num_layers": self._num_layers_on_this_rank,
            "profile": {
                "groups": {name: asdict(stats) for name, stats in self._offload_group_stats.items()},
                "layers": {number: asdict(stats) for number, stats in self._layer_stats.items()},
                "pcie": asdict(self._pcie_stats) if self._pcie_stats else None,
                "transport_host": getattr(self, '_transport_host_costs', {}),
                "children": getattr(self, '_auto_child_profiles', {}),
            } if self._is_stall_profile_rank else None,
        }

    def apply_optimization_results(self) -> Tuple[Set[str], Set[str]]:
        """One collective input merge, one solver, one atomic policy application."""
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        host_feedback_keys = self._measured_host_cost_keys()
        payload = self._profile_payload()
        if distributed:
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, payload)
        else:
            gathered = [payload]
        error = None
        try:
            stage_entries = [entry for entry in gathered if (
                not self._per_pp_rank_enabled or entry["pp_rank"] == self._pp_rank
            )]
            if not stage_entries or any(not entry["memory"]["samples"] for entry in stage_entries):
                raise RuntimeError("Missing memory telemetry on a participating rank")
            profiles = [entry["profile"] for entry in stage_entries if entry["profile"]]
            if not profiles:
                raise RuntimeError(f"Missing profiling representative for PP stage {self._pp_rank}")
            self._offload_group_stats = {}
            self._layer_stats = {}
            for profile in profiles:
                for name, data in profile["groups"].items():
                    previous = self._offload_group_stats.get(name)
                    merged = dict(data)
                    if previous is not None:
                        for key, value in data.items():
                            if isinstance(value, (int, float)):
                                merged[key] = max(getattr(previous, key), value)
                    self._offload_group_stats[name] = OffloadGroupStats(**merged)
                for number, data in profile["layers"].items():
                    layer = self._get_or_create_layer(int(number), data["is_moe"])
                    for name, values in data["modules"].items():
                        previous = layer.modules.get(name)
                        merged = {key: max(getattr(previous, key), value) if previous else value
                                  for key, value in values.items()}
                        layer.modules[name] = LayerModuleStats(**merged)
            for name, stats in self._offload_group_stats.items():
                stats.storage_footprint_sample_count = min(
                    profile["groups"].get(name, {}).get("storage_footprint_sample_count", 0)
                    for profile in profiles
                )
            bandwidths = [profile["pcie"] for profile in profiles if profile["pcie"] and profile["pcie"]["measured"]]
            self._pcie_stats = PCIeBandwidthStats(
                d2h_gbps=min(data["d2h_gbps"] for data in bandwidths),
                h2d_gbps=min(data["h2d_gbps"] for data in bandwidths),
                measured=True,
            ) if bandwidths else None
            self._transport_host_costs = {}
            from .dense_activation_boundary import merge_child_measurements
            self._auto_child_profiles = {}
            for name in {name for profile in profiles for name in profile.get('children', {})}:
                merged = merge_child_measurements([profile.get('children', {}).get(name) for profile in profiles], replicas=True)
                if merged is not None:
                    self._auto_child_profiles[name] = merged
            for host_key in {key for profile in profiles for key in profile.get('transport_host', {})}:
                measurements = [profile.get('transport_host', {}).get(host_key) for profile in profiles]
                measurements = [measurement for measurement in measurements if measurement]
                if measurements:
                    self._transport_host_costs[host_key] = {
                        'module_cpu_ms': max(measurement['module_cpu_ms'] for measurement in measurements),
                        'offload_cpu_ms': max(measurement['offload_cpu_ms'] for measurement in measurements),
                        'sample_count': min(measurement['sample_count'] for measurement in measurements),
                    }
            self._num_layers_on_this_rank = max(entry["num_layers"] for entry in stage_entries)
            self._memory_telemetry = {
                "baseline_peak_bytes": max(entry["memory"]["baseline_peak_bytes"] for entry in stage_entries),
                "activation_baseline_peak_bytes": max(entry['memory'].get('activation_baseline_peak_bytes', 0) for entry in stage_entries),
                "capacity_bytes": min(entry["memory"]["capacity_bytes"] for entry in stage_entries),
                "planning_capacity_bytes": min(entry["memory"].get("planning_capacity_bytes", entry["memory"]["capacity_bytes"])
                                               for entry in stage_entries),
                "capacity_variation_bytes": max(entry["memory"].get("capacity_variation_bytes", 0) for entry in stage_entries),
                "capacity_window_iterations": min(entry["memory"].get("capacity_window_iterations", 0) for entry in stage_entries),
                "allocator_unavailable_bytes": max(entry["memory"].get("allocator_unavailable_bytes", 0) for entry in stage_entries),
                "allocator_stats_samples": min(entry["memory"].get("allocator_stats_samples", 0) for entry in stage_entries),
                "allocator_peak_overhead_bytes": max(entry["memory"].get("allocator_peak_overhead_bytes", 0) for entry in stage_entries),
                "observed_peak_allocated_bytes": max(entry["memory"].get("observed_peak_allocated_bytes", 0) for entry in stage_entries),
                "observed_peak_footprint_bytes": max(entry["memory"].get("observed_peak_footprint_bytes", 0) for entry in stage_entries),
                "max_inflight_microbatches": max(entry["memory"]["max_inflight_microbatches"] for entry in stage_entries),
                "samples": min(entry["memory"]["samples"] for entry in stage_entries),
            }
            self._run_optimization()
        except Exception as exception:
            error = f"PP stage {self._pp_rank}: {type(exception).__name__}: {exception}"
        if distributed:
            errors = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(errors, error)
        else:
            errors = [error]
        failures = sorted({message for message in errors if message})
        if failures:
            raise RuntimeError("Unified activation policy rejected on all ranks: " + "; ".join(failures))
        self._optimization_applied = True
        self._last_optimize_iter = self._current_iter
        self._host_cost_feedback_keys.update(host_feedback_keys | self._measured_host_cost_keys())
        if self._last_profile_iter < 0 or self._profile_refresh_pending:
            self._last_profile_iter = self._current_iter
            self._profile_refresh_pending = False
        if os.environ.get('AUTO_ACTIVATION_MEMORY') == '1':
            from mindspeed.core.pipeline_parallel.adaptive_offload.auto_activation_memory import save_profile_cache
            save_profile_cache(self)
        output = os.environ.get("ADAPTIVE_MEM_PROFILE_JSON")
        if output:
            rank = torch.distributed.get_rank() if distributed else 0
            try:
                self.save_to_json(f"{output}.pp{self._pp_rank}.rank{rank}.json")
            except OSError as exception:
                logger.warning("Cannot save activation profile: %s", exception)
        return self._skip_offload_groups, self._recompute_groups

    def _solve_auto_modules(self):
        from dataclasses import replace
        from .transfer_cost_feedback import TransferCostFeedback

        started = time.perf_counter_ns()

        groups = {}
        modules = {}
        for key, spec in self._auto_module_specs.items():
            stats = self._offload_group_stats.get(key)
            if stats is None or stats.total_offload_bytes <= 0:
                continue
            times = {}
            for direction in ('d2h', 'h2d'):
                samples = getattr(stats, direction + '_sample_count')
                bandwidth = getattr(self._pcie_stats, direction + '_gbps', 0.0)
                if stats.total_offload_bytes == 0:
                    times[direction] = 0.0
                elif samples:
                    times[direction] = max(getattr(stats, direction + '_time_ms'), 0.001)
                elif bandwidth > 0:
                    times[direction] = stats.total_offload_bytes / (1024 ** 3) / bandwidth * 1000
                else:
                    raise RuntimeError(f'Missing {direction} calibration for module {key}')
            storage_measured = stats.storage_footprint_sample_count >= 2 and not stats.storage_footprint_incomplete
            resident_bytes = max(stats.non_offloadable_bytes, stats.resident_storage_bytes)
            groups[key] = GroupProfile(stats.total_offload_bytes / 1024 ** 2, stats.forward_compute_time_ms,
                                       stats.backward_compute_time_ms, times['d2h'], times['h2d'],
                                       resident_mb=resident_bytes / 1024 ** 2,
                                       keep_mb=stats.keep_storage_bytes / 1024 ** 2 if storage_measured else None,
                                       d2h_storage_ratio=stats.d2h_storage_ratio)
            layer = self._layer_stats.get(spec['layer'])
            measured = layer.modules.get(key) if layer else None
            if measured is not None and measured.sample_count >= 2 and measured.input_bytes > 0:
                modules[key] = ModuleProfile(
                    measured.recompute_time_ms if measured.recompute_sample_count >= 2 else measured.compute_time_ms,
                    measured.input_bytes / 1024 ** 2,
                    measured.peak_memory_delta_bytes / 1024 ** 2 + groups[key].activation_mb,
                )
        unexpected = {name for name, stats in self._offload_group_stats.items()
                      if stats.total_offload_bytes > 0 and name not in self._auto_module_specs}
        if unexpected:
            raise RuntimeError(f'Unregistered activation boundaries: {sorted(unexpected)}')
        layer_numbers = sorted({spec['layer'] for spec in self._auto_module_specs.values()})
        layers = tuple(tuple(key for key in sorted(groups, key=lambda key: self._auto_module_specs[key]['order'], reverse=True)
                             if self._auto_module_specs[key]['layer'] == layer) for layer in layer_numbers)
        if not layers:
            raise RuntimeError('No supported Transformer module instances were registered')
        memory = _memory_budget_from_telemetry(self._memory_telemetry, planning=True)
        base = ScheduleConfig(layers=layers,
                              inflight_microbatches=max(1, self._memory_telemetry['max_inflight_microbatches']),
                              adjacent_prefetch=0, cross_layer_prefetch=False, last_layer_keep=False,
                              serialized_recompute=True, baseline_recompute=True, synchronous_d2h=False,
                              immediate_backward_keep=True, unified_transport=True)
        plans = []
        models = {}
        errors = []
        from .dense_activation_boundary import solver_child_profiles
        children = solver_child_profiles(getattr(self, '_auto_child_profiles', {}), self._auto_module_specs, groups)
        candidates = [('synchronous', 0, 1), ('synchronous', 1, 1)]
        candidates.extend(('asynchronous', depth, slots) for depth in (0, 1, 2) for slots in (1, 2))
        for mode, depth, slots in candidates:
            host_key = f'{mode}:{depth}:{slots}'
            host_profiles = getattr(self, '_transport_host_costs', {})
            host, host_source, host_references = estimate_host_costs(host_profiles, mode, depth, slots)
            schedule = replace(base, adjacent_prefetch=depth, d2h_slots=slots,
                               synchronous_d2h=mode == 'synchronous',
                               offload_only_callbacks=mode == 'synchronous',
                               host_module_ms=host.get('module_cpu_ms', 0.0),
                               host_offload_ms=host.get('offload_cpu_ms', 0.0))
            try:
                optimizer = UnifiedOffloadOptimizer(groups, modules, memory, schedule,
                                                    module_groups={key: (key,) for key in groups}, nested_groups={},
                                                    child_recomputes=children)
                plan = optimizer.solve()
                largest = max((groups[key].activation_mb for key, action in plan.decisions.items() if action == 'OFFLOAD'), default=0.0)
                plan.execution.update({'adjacent_prefetch': depth, 'cross_layer_prefetch': False,
                                  'last_layer_keep': False, 'module_granularity': 'layer_instance',
                                  'bootstrap': 'RECOMPUTE', 'bounded_d2h_staging': True,
                                  'transport_version': 1, 'd2h_slots': slots,
                                  'transport_mode': mode, 'transport_active': bool(largest),
                                  'host_cost_source': host_source,
                                  'host_profile_samples': host.get('sample_count', 0),
                                  'host_cost_key': host_key,
                                  'host_cost_reference_keys': host_references,
                                  'host_reference_samples': sum(host_profiles[key]['sample_count'] for key in host_references),
                                  'largest_offload_mb': largest, 'd2h_pending_mb': largest * slots,
                                  'h2d_live_mb': largest * (depth + 1),
                                  'timing_model': 'bounded_stage_queues'})
                plans.append(plan)
                models[id(plan)] = optimizer
            except NoFeasiblePlanError as exception:
                errors.append(str(exception))
        if not plans:
            raise NoFeasiblePlanError('; '.join(errors))
        selected = min(plans, key=lambda plan: (plan.predicted_overhead_ms, plan.transfer_work_ms,
                                              plan.transfer_count, plan.predicted_peak_mb, plan.execution['adjacent_prefetch'],
                                              plan.execution['d2h_slots'], plan.execution['transport_mode'] != 'synchronous'))
        self._auto_transfer_feedback = TransferCostFeedback(
            tuple(models.values()), models[id(selected)], selected, self._offload_group_stats,
            self._current_iter, (time.perf_counter_ns() - started) / 1e6)
        return selected

    def _start_reoptimization(self, iteration: int) -> None:
        """
        Begin a new profiling cycle for dynamic re-optimization.

        Starts fresh event collection while retaining cached measurements for
        groups that the current policy keeps or recomputes instead of offloading.
        """
        self._reoptimize_count += 1
        self._auto_transfer_feedback = None
        self._profiling_done = False
        self._optimization_applied = False
        # Reset measured stats for fresh data collection
        self._pending_stall_events.clear()
        self._pending_bwd_compute_events.clear()
        self._pending_fwd_compute_events.clear()
        self._pending_module_events.clear()
        self._pending_transfer_events.clear()
        # Reset warmup to start profiling immediately (no skip for re-optimization)
        self._pending_child_profiles.clear()
        self._warmup_skip_iters = iteration
        self._capacity_observations.clear()
        for key in ("planning_capacity_bytes", "capacity_variation_bytes", "capacity_window_iterations"):
            self._memory_telemetry.pop(key, None)
        if os.environ.get('AUTO_ACTIVATION_MEMORY') == '1':
            self._plan = None
            self._auto_baseline_validated = False
            self._auto_cache_saved = False
            self.__dict__.pop('_auto_reference_validation', None)
            self._memory_telemetry['baseline_peak_bytes'] = 0
            self._memory_telemetry['activation_baseline_peak_bytes'] = 0
            self._memory_telemetry['samples'] = 0
            self._offload_group_stats.clear()
            self._layer_stats.clear()
            self._auto_child_profiles.clear()
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

        Automatic exports add PP/global rank suffixes to avoid write conflicts.
        """
        if not (self._is_profile_rank or self._is_stall_profile_rank):
            return
        data = {
            "schema_version": 2,
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
            "memory_telemetry": self._memory_telemetry,
            "num_layers_on_this_rank": self._num_layers_on_this_rank,
            "pp_rank": self._pp_rank,
            "plan": self._plan.to_dict() if self._plan else None,
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

        profiler = cls()
        profiler._num_profile_iters = data["num_profile_iters"]
        profiler._profile_rank = 0
        profiler._measure_pcie = False
        profiler._current_iter = profiler._num_profile_iters
        profiler._profiling_done = True
        profiler._is_profile_rank = True
        profiler._is_stall_profile_rank = True
        profiler._lock = threading.Lock()
        profiler._skip_offload_groups = set(data.get("skip_offload_groups", []))
        profiler._recompute_groups = set(data.get("recompute_groups", []))
        profiler._optimization_applied = bool(
            profiler._skip_offload_groups or profiler._recompute_groups
        )
        profiler._pending_stall_events = []
        profiler._pending_bwd_compute_events = []
        profiler._pending_fwd_compute_events = []
        profiler._pending_module_events = []
        profiler._pending_transfer_events = []
        profiler._diag_print_count = 0
        profiler._diag_print_limit = 20
        profiler._warmup_skip_iters = 0  # no warmup needed for offline analysis
        profiler._pcie_measured = True  # already loaded from JSON
        profiler._memory_telemetry.update(data.get("memory_telemetry", {}))
        profiler._num_layers_on_this_rank = data.get("num_layers_on_this_rank", len(data["layer_stats"]))
        profiler._pp_rank = data.get("pp_rank", -1)
        profiler._plan = StrategyPlan(**data["plan"]) if data.get("plan") else None
        profiler._plan_baseline_bytes = profiler._memory_telemetry["baseline_peak_bytes"]

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
