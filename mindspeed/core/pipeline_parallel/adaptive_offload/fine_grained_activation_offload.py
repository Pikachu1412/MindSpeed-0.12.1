# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import time
from collections import deque
from contextlib import nullcontext
from typing import Any, Optional

import torch

# CPU offload implementation for pipeline parallelism
DEBUG = False
DEBUG_RANK = 0
# Enable detailed profiling by setting MEGATRON_OFFLOAD_PROFILING=1
OFFLOAD_PROFILING = os.environ.get("MEGATRON_OFFLOAD_PROFILING", "0") == "1"
# Enable shape logging for offloaded tensors by setting MEGATRON_OFFLOAD_SHAPE_LOG=1
OFFLOAD_SHAPE_LOG = os.environ.get("MEGATRON_OFFLOAD_SHAPE_LOG", "0") == "1"
# Enable pinned memory pool to reuse CPU buffers.
# Default: DISABLED.  The pool avoids repeated cudaHostAlloc, but its buffer
# recycling introduces cross-stream ordering requirements that can cause data
# corruption (NaN) if any H2D reload runs on the compute stream (tensor_pop
# sync-reload path) rather than h2d_stream.  Enable only after verifying no
# sync-reload occurs in your workload.
PINNED_MEMORY_POOL_ENABLED = os.environ.get("MEGATRON_PINNED_MEMORY_POOL", "0") == "1"
# ---- H2D Prefetch master switch ----
# Controls whether H2D prefetch (adjacent-group reload) is used.  When enabled,
# each group's commit_backward triggers the H2D reload of the NEXT group,
# overlapping the transfer with the current group's backward compute.
# When disabled, tensors are only reloaded on-demand when accessed by backward
# compute (sync reload in tensor_pop), which maximizes stall.
H2D_PREFETCH_ENABLED = os.environ.get("MEGATRON_H2D_PREFETCH", "1") == "1"
# ---- Cross-layer prefetch switch ----
# When enabled, after a group's backward compute finishes, prefetch the SAME
# module from the PREVIOUS layer (e.g., after L1-core_attn backward completes,
# prefetch L0-core_attn).  This gives the H2D transfer a full layer's worth of
# backward compute time to overlap with, instead of just one group's compute.
# Requires H2D_PREFETCH_ENABLED=1 to take effect.
CROSS_LAYER_PREFETCH_ENABLED = (
    os.environ.get("MEGATRON_CROSS_LAYER_PREFETCH", "1") == "1"
)
# ---- Last-layer-no-offload strategy switch ----
# When enabled, the last layer (layer N) in each PP stage keeps its activations
# on GPU (no D2H offload).  The preceding N-1 layers offload normally.  During
# backward, when computing layer N (whose activations are already on GPU), the
# handler prefetches layer N-1's activations from CPU; when computing layer N-1,
# it prefetches layer N-2, and so on.  This provides one-layer-ahead prefetch
# at layer granularity, giving a full layer's backward compute time for the
# H2D transfer to overlap with.
#
# INCOMPATIBLE with H2D_PREFETCH and CROSS_LAYER_PREFETCH — those strategies
# operate at group granularity and conflict with this layer-granularity prefetch.
LAST_LAYER_NO_OFFLOAD_ENABLED = (
    os.environ.get("MEGATRON_LAST_LAYER_NO_OFFLOAD", "0") == "1"
)
# ---- Last-layer-no-offload timing instrumentation ----
# When enabled, records GPU-side CUDA event timings for:
#   - Step 2 stall: how long compute_stream waits for H2D reload to finish
#   - Prefetch H2D batch: how long the H2D transfers take on h2d_stream
#   - Backward compute: how long each group's backward compute runs
# Timings are accumulated per iteration and printed at reset() (iteration end).
# Only prints on rank 0 and only for the first N iterations (to avoid flooding).
LAST_LAYER_TIMING_ENABLED = os.environ.get("MEGATRON_LAST_LAYER_TIMING", "0") == "1"
# Number of iterations to print timing data for (then auto-disable).
LAST_LAYER_TIMING_MAX_ITERS = int(
    os.environ.get("MEGATRON_LAST_LAYER_TIMING_ITERS", "3")
)
# Global iteration counter for timing (shared across all chunks).
_timing_iter_count = 0
# ---- Adaptive offload master switch ----
# When enabled, the adaptive memory profiler collects stall/compute data and
# applies three-way decisions (OFFLOAD / RECOMPUTE / KEEP) after profiling.
# When disabled, all groups are offloaded (default behavior, no profiling overhead).
ADAPTIVE_OFFLOAD_ENABLED = os.environ.get("MEGATRON_ADAPTIVE_OFFLOAD", "0") == "1"
# Only log on rank 0 to avoid duplicate output
OFFLOAD_SHAPE_LOG_RANK = 0

# Lazy import profiler to avoid circular dependency
_profiler = None

# Lazy import adaptive profiler
_adaptive_profiler = None


def get_adaptive_profiler():
    """Get the AdaptiveMemoryProfiler singleton if adaptive offload is enabled."""
    global _adaptive_profiler
    if not ADAPTIVE_OFFLOAD_ENABLED:
        return None
    if _adaptive_profiler is None:
        from mindspeed.core.pipeline_parallel.adaptive_offload.adaptive_memory_profiler import (
            AdaptiveMemoryProfiler,
        )

        _adaptive_profiler = AdaptiveMemoryProfiler.get_instance()
    return _adaptive_profiler


def get_profiler():
    """Get the OffloadProfiler instance if profiling is enabled."""
    global _profiler
    if not OFFLOAD_PROFILING:
        return None
    if _profiler is None:
        from mindspeed.core.pipeline_parallel.adaptive_offload.offload_profiler import OffloadProfiler

        _profiler = OffloadProfiler.get_instance()
    return _profiler


def set_ideal_affinity_for_current_gpu():
    """Set CPU affinity for the current GPU to optimize host-device transfers."""
    return


def debug_rank(message):
    """Print debug message for a specific rank when DEBUG is enabled."""
    # pylint: disable=bad-builtin
    if not DEBUG:
        return
    assert torch.distributed.is_initialized()
    if torch.distributed.get_rank() == DEBUG_RANK:
        print(message)


def shape_log_rank(message):
    """Print shape log message for rank 0 when OFFLOAD_SHAPE_LOG is enabled."""
    # pylint: disable=bad-builtin
    if not OFFLOAD_SHAPE_LOG:
        return
    if (
        not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == OFFLOAD_SHAPE_LOG_RANK
    ):
        print(message)


class PinnedMemoryPool:
    """
    Pool for reusing pinned CPU memory buffers to avoid repeated cudaHostAlloc.

    cudaHostAlloc (pin_memory=True) is expensive because it involves OS-level
    page locking. In training loops, each iteration offloads tensors with
    identical shapes, so we can cache and reuse pinned buffers.

    Safety: Used buffers are not returned to the free pool immediately because
    non_blocking H2D DMA may still be reading from them. Instead, they are
    collected in a pending list and returned at the next iteration boundary
    (reset()), where all prior DMA operations are guaranteed to be complete
    (gradient allreduce + optimizer step provide implicit synchronization).
    """

    def __init__(self):
        # Free buffers available for reuse, keyed by (shape_tuple, dtype)
        self._free: dict[tuple, list[torch.Tensor]] = {}
        # Buffers used in current iteration, pending return to free pool
        self._pending: list[torch.Tensor] = []
        # Statistics
        self._hit_count = 0
        self._miss_count = 0

    def get(self, size, dtype, layout=torch.strided) -> torch.Tensor:
        """
        Get a pinned CPU buffer. Reuses from pool if available, otherwise allocates.

        Args:
            size: Tensor size (shape tuple or torch.Size)
            dtype: Tensor dtype
            layout: Tensor layout (default: torch.strided)
        Returns:
            A pinned CPU tensor with the requested size/dtype
        """
        if PINNED_MEMORY_POOL_ENABLED:
            key = (tuple(size), dtype)
            if key in self._free and self._free[key]:
                self._hit_count += 1
                return self._free[key].pop()
        self._miss_count += 1
        return torch.empty(
            size, dtype=dtype, layout=layout, device="cpu", pin_memory=True
        )

    def mark_used(self, tensor: torch.Tensor):
        """
        Mark a buffer as used. It will be returned to the free pool on reset().

        Must be called after reload() consumes a cpu_backup tensor, so the buffer
        is not reused while DMA may still be in-flight.
        """
        if PINNED_MEMORY_POOL_ENABLED:
            self._pending.append(tensor)

    def has_pending(self) -> bool:
        """Check if there are any buffers pending return to the free pool."""
        return len(self._pending) > 0

    def reset(self):
        """
        Return all pending buffers to the free pool.

        Must be called at iteration boundaries where all prior CUDA operations
        are guaranteed to be complete.
        """
        for buf in self._pending:
            key = (tuple(buf.size()), buf.dtype)
            if key not in self._free:
                self._free[key] = []
            self._free[key].append(buf)
        self._pending.clear()

    @property
    def stats(self):
        """Return pool statistics for debugging."""
        total_free = sum(len(v) for v in self._free.values())
        total_pending = len(self._pending)
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / max(1, self._hit_count + self._miss_count),
            "free_buffers": total_free,
            "pending_buffers": total_pending,
            "unique_shapes": len(self._free),
        }


class PipelineOffloadManager:
    """
    Singleton manager for coordinating activation offloading across pipeline stages.
    Manages chunk handlers, synchronizes GPU-CPU transfers,
    and handles virtual pipeline parallelism.
    """

    OFFLOAD_MGR = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of PipelineOffloadManager."""
        if cls.OFFLOAD_MGR is None:
            cls.OFFLOAD_MGR = PipelineOffloadManager()
        return cls.OFFLOAD_MGR

    def __init__(self):
        """Initialize the manager with queues and dedicated CUDA streams."""
        # Queue to store chunk handlers for backward pass
        self._queue = deque()
        # Cache chunk handlers for each virtual pipeline stage
        self._stages = None
        # allocate streams and events for synchronization
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()
        # Pinned memory pool for reusing CPU buffers across iterations
        self._pinned_memory_pool = PinnedMemoryPool()
        # --- Adaptive offload decision state ---
        # Groups to keep on GPU (KEEP decision from V3 optimizer)
        self._skip_offload_groups: set = set()
        # Groups to recompute during backward (RECOMPUTE decision)
        self._recompute_groups: set = set()
        self.reset()
        # ---- Incompatibility check ----
        # LAST_LAYER_NO_OFFLOAD uses its own layer-granularity prefetch and
        # is incompatible with group-granularity prefetch strategies.
        if LAST_LAYER_NO_OFFLOAD_ENABLED:
            if H2D_PREFETCH_ENABLED:
                raise RuntimeError(
                    "MEGATRON_LAST_LAYER_NO_OFFLOAD=1 is incompatible with "
                    "MEGATRON_H2D_PREFETCH=1.  Please set MEGATRON_H2D_PREFETCH=0 "
                    "when using last-layer-no-offload strategy."
                )
            if CROSS_LAYER_PREFETCH_ENABLED:
                raise RuntimeError(
                    "MEGATRON_LAST_LAYER_NO_OFFLOAD=1 is incompatible with "
                    "MEGATRON_CROSS_LAYER_PREFETCH=1.  Please set "
                    "MEGATRON_CROSS_LAYER_PREFETCH=0 when using "
                    "last-layer-no-offload strategy."
                )
        # Log configuration switches once on rank 0
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(
                f"[OffloadManager] Configuration switches:\n"
                f"  H2D_PREFETCH (adjacent-group reload) : "
                f"{'ON' if H2D_PREFETCH_ENABLED else 'OFF'}\n"
                f"  CROSS_LAYER_PREFETCH (same-module)    : "
                f"{'ON' if CROSS_LAYER_PREFETCH_ENABLED else 'OFF'}\n"
                f"  LAST_LAYER_NO_OFFLOAD (layer prefetch): "
                f"{'ON' if LAST_LAYER_NO_OFFLOAD_ENABLED else 'OFF'}\n"
                f"  PINNED_MEMORY_POOL                   : "
                f"{'ON' if PINNED_MEMORY_POOL_ENABLED else 'OFF'}\n"
                f"  ADAPTIVE_OFFLOAD (profile + V3)      : "
                f"{'ON' if ADAPTIVE_OFFLOAD_ENABLED else 'OFF'}\n"
                f"  LAST_LAYER_TIMING (GPU event timing) : "
                f"{'ON (first ' + str(LAST_LAYER_TIMING_MAX_ITERS) + ' iters)' if LAST_LAYER_TIMING_ENABLED else 'OFF'}",
                flush=True,
            )

    @property
    def d2h_stream(self):
        """Get the device-to-host (GPU to CPU) transfer stream."""
        return self._d2h_stream

    @property
    def h2d_stream(self):
        """Get the host-to-device (CPU to GPU) transfer stream."""
        return self._h2d_stream

    @property
    def pinned_memory_pool(self):
        """Get the pinned memory pool for CPU buffer reuse."""
        return self._pinned_memory_pool

    @property
    def recompute_groups(self):
        """Return groups marked for RECOMPUTE by the V3 decision engine.

        When ADAPTIVE_OFFLOAD is enabled, this returns the set of offload group
        names that should be recomputed during backward instead of offloaded.
        TransformerLayer reads this to dynamically wrap modules in checkpoint().
        """
        return self._recompute_groups

    @recompute_groups.setter
    def recompute_groups(self, value):
        """Set the recompute groups (called from training.py after profiling)."""
        self._recompute_groups = set(value) if value else set()

    @property
    def skip_offload_groups(self):
        """Return groups marked for KEEP by the V3 decision engine.

        When ADAPTIVE_OFFLOAD is enabled, this returns the set of offload group
        names that should stay on GPU (skip D2H offload entirely).
        TransformerLayer reads this to skip group_start/commit calls.
        """
        return self._skip_offload_groups

    @skip_offload_groups.setter
    def skip_offload_groups(self, value):
        """Set the skip-offload groups (called from training.py after profiling)."""
        self._skip_offload_groups = set(value) if value else set()

    def reset(self):
        """Reset manager state for a new training iteration."""
        # ---- Flush last-layer timing data before resetting state ----
        # Guard with hasattr because reset() is called from __init__ before
        # _cur_backward_chunk / _queue are assigned.
        global _timing_iter_count
        if (
            LAST_LAYER_TIMING_ENABLED
            and _timing_iter_count < LAST_LAYER_TIMING_MAX_ITERS
            and hasattr(self, "_cur_backward_chunk")
        ):
            # Flush timing events from the current backward chunk (if any)
            if self._cur_backward_chunk is not None:
                self._cur_backward_chunk.flush_timing_events()
            # Also flush from any remaining chunks in the queue
            for chunk in self._queue:
                chunk.flush_timing_events()
            _timing_iter_count += 1

        set_ideal_affinity_for_current_gpu()
        self._inside_context = False
        self._cur_forward_chunk = None
        self._cur_backward_chunk = None
        # Track the first microbatch of the last virtual pipeline stage
        self._is_first_last_vpp_chunk = True
        # Establish cross-stream GPU-side dependencies at iteration boundary.
        # This ensures correct ordering when streams are reused across iterations:
        #   - d2h_stream's future D2H writes happen AFTER h2d_stream's past H2D reads
        #   - h2d_stream's future H2D reads happen AFTER d2h_stream's past D2H writes
        # Required regardless of pinned memory pool, since the bulk_reload while-loop
        # prefetch may leave h2d_stream work that overlaps with next iteration's d2h_stream.
        # Using wait_stream (GPU-side) instead of synchronize (CPU-blocking) to avoid
        # stalling the CPU thread, which would create pipeline bubbles.
        self._d2h_stream.wait_stream(self._h2d_stream)
        self._h2d_stream.wait_stream(self._d2h_stream)
        self._pinned_memory_pool.reset()

    def flush(self):
        """Flush all staged chunks to the backward queue in reverse order."""
        # Ensure all virtual pipeline stages have the same number of chunks
        if len(self._stages[0]) == len(self._stages[-1]):
            lens = [len(e) for e in self._stages]
            assert min(lens) == max(lens), "All stages must have same chunk count"
            # Clear the last stage and push all chunks in reverse order for backward
            self._stages[-1] = []
            for chunks in reversed(self._stages):
                for chunk in chunks:
                    self.push(chunk)
            # Clear all stages after flushing
            for i in range(self._vpp):
                self._stages[i] = []

    def push(self, handler):
        """Add a chunk handler to the backward queue."""
        debug_rank(f"pushing handler {handler}")
        self._queue.append(handler)

    def pop(self):
        """Remove and set the next non-empty chunk as the current backward chunk."""
        assert self.size(), "Cannot pop from empty queue"
        while self._queue:
            self._cur_backward_chunk = self._queue.popleft()
            if not self._cur_backward_chunk.is_empty_chunk():
                break
        debug_rank(f"popping handler {self._cur_backward_chunk}")

    def front(self):
        """Get the first non-empty chunk handler without removing it from the queue."""
        if not self.size():
            return None
        for chunk_handler in self._queue:
            if not chunk_handler.is_empty_chunk():
                return chunk_handler
        return None

    def size(self):
        """Return the number of chunk handlers in the queue."""
        return len(self._queue)

    def init_model_chunk_offload_handler(
        self, vp_size, vp_stage, min_offloaded_tensor_size=1024 * 1024
    ):
        """
        Initialize a chunk offload handler for a model chunk (microbatch).

        Args:
            vp_size: Virtual pipeline size
            vp_stage: Virtual pipeline stage index (None means stage 0)
            min_offloaded_tensor_size: Minimum tensor size (in elements) to offload
        """
        if self._stages is None:
            vp_size = 1 if vp_size is None else vp_size
            self._vpp = vp_size
            self._stages = [[] for _ in range(vp_size)]

        if vp_stage is None:
            cur_vpp_rank = 0
        else:
            cur_vpp_rank = vp_stage

        is_first_last_vpp_chunk = self._is_first_last_vpp_chunk
        # Flush staged chunks when reaching the last virtual pipeline stage
        if cur_vpp_rank == self._vpp - 1:
            self.flush()
        # Determine if this is the first microbatch of the last virtual pipeline stage
        is_first_last_vpp_chunk = is_first_last_vpp_chunk and (
            cur_vpp_rank == self._vpp - 1
        )

        cur_chunk = ChunkOffloadHandler(
            is_first_last_vpp_chunk, min_offloaded_tensor_size
        )
        self._stages[cur_vpp_rank].append(cur_chunk)
        # For the last stage, push immediately and flush
        if cur_vpp_rank == self._vpp - 1:
            self._is_first_last_vpp_chunk = False
            self.push(cur_chunk)
            self.flush()
        self._cur_forward_chunk = cur_chunk
        cur_chunk.vpp_rank = cur_vpp_rank

    def set_last_layer(self, is_last_layer):
        """Mark whether the current forward chunk is processing the last layer."""
        self._cur_forward_chunk.is_last_layer = is_last_layer

    def cur_forward_chunk(self):
        """Get the current forward pass chunk handler."""
        return self._cur_forward_chunk

    def cur_backward_chunk(self):
        """Get the current backward pass chunk handler."""
        return self._cur_backward_chunk

    def __enter__(self):
        """Enter context manager to enable activation offloading hooks."""
        debug_rank("----__enter__")
        from megatron.core.extensions.transformer_engine import cpu_offload

        if cpu_offload is not None:
            cpu_offload.CPUOffloadEnabled = True
        else:
            raise RuntimeError("TE CPU offload is not available")
        self.inside_context = True

        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args: Any):
        """Exit context manager and restore original tensor saving behavior."""
        debug_rank("----__exit__")
        from megatron.core.extensions.transformer_engine import cpu_offload

        if cpu_offload is not None:
            cpu_offload.CPUOffloadEnabled = False
        else:
            raise RuntimeError("TE CPU offload is not available")
        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        """
        Hook called when autograd saves a tensor for backward pass.
        Returns a tag to identify the tensor later.
        """
        debug_rank(f"------on_save_for_backward {tensor.shape}")
        assert self.inside_context, "Must be inside offload context"
        return self.cur_forward_chunk().tensor_push(tensor)

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """
        Hook called when autograd retrieves a saved tensor during backward pass.
        Returns the actual tensor (potentially reloading from CPU).
        """
        debug_rank(f"----on_get_saved_tensor {saved_state}")
        return self.cur_backward_chunk().tensor_pop(saved_state)


class ChunkOffloadHandler:
    """
    Handles activation offloading and reloading for a single pipeline chunk (microbatch).
    Manages tensor groups, coordinates asynchronous GPU-CPU transfers, and handles synchronization.
    """

    def offload(self, src_tensor, pin_memory=True):
        """
        Offload a GPU tensor to pinned CPU memory.

        Uses PinnedMemoryPool to reuse pinned buffers, avoiding expensive
        cudaHostAlloc calls on every iteration.
        """
        debug_rank("--------offload")

        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()

        pool = PipelineOffloadManager.get_instance().pinned_memory_pool
        cpu_backup = pool.get(src_tensor.size(), src_tensor.dtype, src_tensor.layout)
        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup)
        return state

    def reload(self, state, non_blocking=None):
        """
        Reload a tensor from CPU back to GPU.

        After the H2D copy is issued, the cpu_backup is marked as used in the
        PinnedMemoryPool. It will be returned to the free pool at the next
        iteration boundary (reset()), ensuring DMA is complete before reuse.
        """
        debug_rank("------reload")
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        gpu_tensor = cpu_backup.to(dev, non_blocking=non_blocking)
        # Mark cpu_backup for deferred return to pool.
        # Not returned immediately because non_blocking DMA may still be reading it.
        pool = PipelineOffloadManager.get_instance().pinned_memory_pool
        pool.mark_used(cpu_backup)
        return gpu_tensor

    def __init__(self, is_first_last_vpp_chunk, min_offloaded_tensor_size):
        # Data Structure to maintain reference to activation tensors
        self._tensor_tag_to_state = {}
        # Mark the first microbatch of the last virtual pipeline stage
        self._is_first_last_vpp_chunk = is_first_last_vpp_chunk

        # Group management for batching offload/reload operations
        self._offloaded_group_index = 0
        self._groups_to_offload = []
        self._groups_to_reload = []
        # Set of gids whose H2D reload has already been issued on h2d_stream.
        # Used by commit_backward to avoid issuing duplicate reloads.
        self._reload_issued = set()
        self._tensor_count_current_group = 0
        # Mapping from group_id to group name for logging
        self._group_id_to_name = {}

        # ---- Cross-layer prefetch state ----
        # Number of offload groups per transformer layer (auto-detected after
        # the first layer's forward pass completes).  Used to compute the gid
        # of the "same module in previous layer" for cross-layer prefetch.
        self._num_groups_per_layer = None
        # Reverse lookup: maps (name, layer_index) -> group_id for fast cross-
        # layer search.  Populated lazily during the first backward pass.
        self._name_layer_to_gid = None
        # Total number of layers in this chunk (auto-detected).
        self._num_layers = None
        # ---- Last-layer-no-offload prefetch state ----
        # Tracks the last layer whose previous layer was prefetched, to avoid
        # duplicate work when multiple groups within the same layer call
        # prefetch_previous_layer.
        self._last_prefetched_layer = None

        # ---- Last-layer-no-offload timing instrumentation ----
        # Accumulated CUDA event pairs for deferred timing measurement.
        # Each entry: (label: str, start_event: cuda.Event, end_event: cuda.Event)
        # Flushed at iteration end in PipelineOffloadManager.reset().
        self._timing_events = []
        # Backward compute start event (set in commit_backward, consumed in start_backward)
        self._ll_bwd_start = None
        self._ll_bwd_gid = None
        self._ll_bwd_name = None

        # Counter for special torch tensor types (FakeTensor, FunctionalTensor)
        self.torch_tensor_count = 0
        mgr = PipelineOffloadManager.get_instance()
        self.d2h_stream = mgr.d2h_stream
        self.h2d_stream = mgr.h2d_stream
        self._offload_events = {}
        self._reload_events = {}
        # group_id-indexed events for cross-layer reload (name key gets overwritten
        # across layers since they share the same group name).
        self._offload_events_by_id = {}
        self._reload_events_by_id = {}
        self.min_offloaded_tensor_size = min_offloaded_tensor_size
        self.is_last_layer = False

    def flush_timing_events(self):
        """Synchronize and print all accumulated CUDA event timings.

        Called from PipelineOffloadManager.reset() at iteration end.
        Uses Event.elapsed_time() which requires the events to have completed,
        so we do a stream synchronize first.  This is acceptable because
        reset() is at the iteration boundary where all GPU work is done.

        Only prints on rank 0 to avoid duplicate output.
        """
        if not self._timing_events:
            return
        is_rank0 = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        if not is_rank0:
            self._timing_events.clear()
            return

        # Synchronize to ensure all events have been recorded on GPU
        torch.cuda.synchronize()

        global _timing_iter_count
        lines = []
        lines.append(
            f"[LAST_LAYER_TIMING] iter={_timing_iter_count} "
            f"chunk={id(self) & 0xFFFF:#06x}"
        )
        for label, start_ev, end_ev in self._timing_events:
            try:
                elapsed_ms = start_ev.elapsed_time(end_ev)
                lines.append(f"  {label}: {elapsed_ms:.3f} ms")
            except RuntimeError:
                lines.append(f"  {label}: <event not recorded>")
        print("\n".join(lines), flush=True)
        self._timing_events.clear()

    def is_empty_chunk(self):
        """Check if this chunk has no tensors to manage."""
        return len(self._tensor_tag_to_state) == 0

    def is_first_last_layer(self):
        """
        Check if this is the last layer of the first microbatch of the last vp stage.
        These tensors should not be offloaded to avoid unnecessary overhead.
        """
        debug_rank(
            f"------is_first_last_layer {self._is_first_last_vpp_chunk} {self.is_last_layer}"
        )
        return self._is_first_last_vpp_chunk and self.is_last_layer

    def tensor_push(self, tensor):
        """Push tensor to the offload handler."""
        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )

        if not torch_stray_tensor:
            # Assign unique tag based on group index and position within group
            tensor_tag = (self._offloaded_group_index, self._tensor_count_current_group)
            self._tensor_count_current_group += 1
            assert tensor_tag not in self._tensor_tag_to_state, "Duplicate tensor tag"
            self._tensor_tag_to_state[tensor_tag] = tensor
        else:
            # Use negative group ID for special tensor types
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self._tensor_tag_to_state[tensor_tag] = tensor
        debug_rank(f"--------tensor_push {tensor_tag}")
        return tensor_tag

    def tensor_pop(self, tensor_tag):
        """Pop tensor from the offload handler.

        All tensors must have been reloaded to GPU by bulk_reload_group
        (issued from on_group_commit_backward) before this is called.
        There is no sync-reload fallback — if a tensor is still in
        offloaded (tuple) form, it means reload was not issued in time
        and we raise an error.
        """
        debug_rank(f"--------tensor_pop {tensor_tag}")
        assert tensor_tag in self._tensor_tag_to_state, f"Tag {tensor_tag} not found"
        tensor = self._tensor_tag_to_state.pop(tensor_tag)

        if isinstance(tensor, tuple):
            # This should not happen — bulk_reload_group should have already
            # converted this tuple back to a GPU tensor.  If we reach here,
            # it means the reload was not issued before backward compute
            # started requesting tensors.  Fall through to a hard error so
            # we can diagnose.
            group_id = tensor_tag[0]
            group_name = self._group_id_to_name.get(
                group_id, f"unknown_group_{group_id}"
            )
            raise RuntimeError(
                f"tensor_pop: tensor {tensor_tag} (group={group_name!r}) is still "
                f"offloaded (tuple). bulk_reload_group should have reloaded it "
                f"before backward compute. _reload_issued={self._reload_issued}"
            )
        debug_rank(f"--------tensor_pop {tensor.shape}")
        return tensor

    def tensor_need_offloading_checker(self, tensor):
        """Check if the tensor needs to be offloaded."""
        if tensor.numel() < self.min_offloaded_tensor_size:
            return False
        # Respect tensor's offload preference if specified
        if (
            hasattr(tensor, "offloading_activation")
            and not tensor.offloading_activation
        ):
            return False
        # Skip model parameters (nn.Parameter instances only).
        # Autograd's save_for_backward saves both activations AND weight
        # tensors.  Weight tensors are model parameters that permanently
        # reside on GPU — offloading them to CPU and reloading is pure
        # waste of PCIe bandwidth.  For expert_fc1, this filter keeps
        # only the activation (1024 MB input tensor) and skips all
        # weight tensors (~768 MB for 128 experts).
        #
        # IMPORTANT: Do NOT use `requires_grad and is_leaf` — some
        # activation tensors (TE internal tensors, P2P recv tensors)
        # are also leaf tensors with requires_grad=True but are NOT
        # model parameters and must be offloaded normally.
        if isinstance(tensor, torch.nn.Parameter):
            return False
        return True

    def bulk_offload_group(self, group_to_offload):
        """offload a group of tensors recorded in tensor_push()."""
        debug_rank("------bulk_offload_group")
        assert not self.is_first_last_layer(), "Should not offload first-last layer"
        group_id_to_offload, name = group_to_offload
        # torch.cuda.nvtx.range_push("activation offloading " + name)

        # Record timing for profiling
        if OFFLOAD_PROFILING:
            start_time = time.time()

        offloaded_tensor = None  # Track for profiling
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_id_to_offload:
                    debug_rank(f"------tensor_tag {tensor_tag}")
                    debug_rank(f"------group_to_offload {group_to_offload}")
                    assert not isinstance(state, tuple), "Tensor already offloaded"
                    tensor_on_device = state
                    if self.tensor_need_offloading_checker(tensor_on_device):
                        offloaded_tensor = tensor_on_device  # Save for profiling
                        tensor_bytes = (
                            tensor_on_device.numel() * tensor_on_device.element_size()
                        )
                        shape_log_rank(
                            f"[OFFLOAD_SHAPE][D2H  ] group={name!r} tag={tensor_tag} "
                            f"shape={list(tensor_on_device.shape)} dtype={tensor_on_device.dtype} "
                            f"nbytes={tensor_bytes / 1024 / 1024:.3f}MB"
                        )
                        state = self.offload(tensor_on_device)
                        event = torch.cuda.Event()
                        event.record(self.d2h_stream)
                        self._offload_events[name] = event
                        self._offload_events_by_id[group_id_to_offload] = event
                        tensor_on_device.record_stream(self.d2h_stream)
                        self._tensor_tag_to_state[tensor_tag] = state

        # Record to profiler
        if OFFLOAD_PROFILING:
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            profiler = get_profiler()
            if profiler is not None and offloaded_tensor is not None:
                profiler.record_offload(name, offloaded_tensor, elapsed_time)

        # torch.cuda.nvtx.range_pop()

    def get_offload_event(self, name):
        """Get the CUDA event for a named offload operation."""
        return self._offload_events.get(name, None)

    def get_reload_event(self, name):
        """Get the CUDA event for a named reload operation."""
        return self._reload_events.get(name, None)

    def bulk_reload_group(self, group_to_reload):
        """Reload all tensors of a group from CPU to GPU on h2d_stream.

        Returns True if any tensors were found for this group, False otherwise.
        Skips groups that have already been issued (tracked by _reload_issued).
        """
        debug_rank("----bulk_reload_group")
        group_id_to_reload, name = group_to_reload

        # Skip if already issued
        if group_id_to_reload in self._reload_issued:
            return True

        found_reload_group = False
        # torch.cuda.nvtx.range_push("activation reloading " + name)

        # Record timing for profiling
        if OFFLOAD_PROFILING:
            start_time = time.time()
            wait_time = 0.0

        with torch.cuda.stream(self.h2d_stream):
            for tensor_label, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_id_to_reload:
                    debug_rank(f"----tensor_label {tensor_label}")
                    found_reload_group = True
                    # Use group_id-indexed event for cross-layer safety;
                    # fall back to name-indexed for backward compat.
                    event = self._offload_events_by_id.get(
                        group_id_to_reload, self.get_offload_event(name)
                    )
                    # Only reload if tensor was offloaded (stored as tuple)
                    if isinstance(state, tuple):
                        # Wait for offload to complete before reloading
                        if OFFLOAD_PROFILING:
                            wait_start_time = time.time()
                        torch.cuda.current_stream().wait_event(event)
                        if OFFLOAD_PROFILING:
                            wait_time = (time.time() - wait_start_time) * 1000

                        recovered_tensor = self.reload(state)
                        shape_log_rank(
                            f"[OFFLOAD_SHAPE][H2D  ] group={name!r} tag={tensor_label} "
                            f"shape={list(recovered_tensor.shape)} dtype={recovered_tensor.dtype} "
                            f"nbytes={recovered_tensor.numel() * recovered_tensor.element_size() / 1024 / 1024:.3f}MB"
                        )
                        event.record(self.h2d_stream)
                        self._reload_events[name] = event
                        self._reload_events_by_id[group_id_to_reload] = event
                        debug_rank(f"----recovered_tensor {recovered_tensor.shape}")
                        self._tensor_tag_to_state[tensor_label] = recovered_tensor

        if found_reload_group:
            self._reload_issued.add(group_id_to_reload)

        # Record to profiler
        if OFFLOAD_PROFILING:
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            profiler = get_profiler()
            if profiler is not None:
                profiler.record_reload(name, elapsed_time, wait_time)

        # torch.cuda.nvtx.range_pop()
        return found_reload_group

    def pre_reload_last_layer(self):
        """Pre-reload the last layer of this chunk to hide reload latency."""
        debug_rank("pre_reload_last_layer")
        assert not self._is_first_last_vpp_chunk, "Should not pre-reload first chunk"
        debug_rank(f"len(self._groups_to_reload) {len(self._groups_to_reload)}")
        # Loop to skip groups already consumed by tensor_pop (sync reload path)
        while len(self._groups_to_reload) > 0:
            if self.bulk_reload_group(self._groups_to_reload[-1]):
                self._groups_to_reload.pop()
                break
            else:
                # Group entries already consumed by tensor_pop, skip it
                self._groups_to_reload.pop()

    def should_bulk_offload(self):
        """Determine if the current group should be offloaded."""
        # Don't offload the first backward chunk's last layer
        if self.is_first_last_layer():
            return False

        # Check if next backward chunk is this chunk (for last pipeline stage)
        next_backward_chunk = PipelineOffloadManager.get_instance().front()
        if next_backward_chunk is not None and next_backward_chunk is self:
            # Don't offload last layer if it's about to be used immediately
            if self.is_last_layer:
                return False

        # ---- Last-layer-no-offload strategy ----
        # When enabled, keep the last layer's activations on GPU.
        # The backward pass will use these on-GPU activations directly and
        # prefetch the previous layer's activations during this time.
        if LAST_LAYER_NO_OFFLOAD_ENABLED and self.is_last_layer:
            return False

        return True

    def bulk_offload(self, forced_released_tensors):
        """Offload a group of tensors and optionally release their GPU memory."""
        debug_rank("----bulk_offload")
        if self.should_bulk_offload():
            group_to_offload = self._groups_to_offload.pop()
            self._groups_to_reload.append(group_to_offload)
            self.bulk_offload_group(group_to_offload)
            # Manually release tensors not auto-freed by torch GC.
            if len(forced_released_tensors) > 0:
                cur_stream = torch.cuda.current_stream()
                for release_tensor in forced_released_tensors:
                    if self.tensor_need_offloading_checker(release_tensor):
                        # Ensure tensor is not in use before freeing
                        release_tensor.record_stream(cur_stream)
                        release_tensor.untyped_storage().resize_(0)

    def on_group_commit_forward(self, forced_released_tensors):
        """Called at the end of a layer group's forward pass to trigger offloading.

        Returns the group_id that was committed, for saving in autograd ctx
        so backward can use the correct gid (avoiding the nesting mismatch).

        Also records forward compute end event and offload bytes for the
        adaptive profiler.
        """
        debug_rank("--on_group_commit_forward")

        # Record forward compute end event for adaptive profiler
        _ap = get_adaptive_profiler()
        if (
            _ap is not None
            and _ap.is_stall_profiling_active()
            and hasattr(self, "_fwd_compute_start_event")
            and self._fwd_compute_start_event is not None
        ):
            _fwd_end = torch.cuda.Event(enable_timing=True)
            _fwd_end.record()
            _ap.enqueue_forward_compute_events(
                self._fwd_compute_group_name,
                self._fwd_compute_start_event,
                _fwd_end,
            )
            self._fwd_compute_start_event = None

        # Wait for compute to finish before starting offload (sync rule #1)
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        # Peek at the gid that will be committed (top of _groups_to_offload stack).
        # Due to nested groups (mlp_norm wraps expert_fc1 + moe_act), the commit
        # order differs from the start order, so the gid at the top of the stack
        # may not be the last-started gid.
        committed_gid = (
            self._groups_to_offload[-1][0] if self._groups_to_offload else -1
        )

        # Record offload bytes for adaptive profiler before bulk_offload
        if _ap is not None and _ap.is_stall_profiling_active():
            group_name = (
                self._groups_to_offload[-1][1] if self._groups_to_offload else None
            )
            if group_name is not None:
                total_bytes = 0
                target_gid = self._groups_to_offload[-1][0]
                for tensor_tag, state in self._tensor_tag_to_state.items():
                    gid, _ = tensor_tag
                    if gid == target_gid and not isinstance(state, tuple):
                        if self.tensor_need_offloading_checker(state):
                            total_bytes += state.numel() * state.element_size()
                if total_bytes > 0:
                    _ap.record_group_offload_bytes(group_name, total_bytes)

        self.bulk_offload(forced_released_tensors)
        return committed_gid

    def _build_cross_layer_index(self):
        """Build reverse-lookup index for cross-layer prefetch (lazy, once).

        Populates ``_name_layer_to_gid`` mapping ``(name, layer_idx) -> gid``
        and auto-detects ``_num_groups_per_layer`` and ``_num_layers``.

        The ``_groups_to_reload`` list is ordered by commit-forward order.
        Groups within one layer repeat in a fixed pattern (e.g., 7 groups per
        layer for MoE), so we can determine per-layer boundaries by counting
        unique names until the first name repeats.
        """
        if self._name_layer_to_gid is not None:
            return  # Already built

        # Determine groups-per-layer by finding the first repeated name.
        seen_names = []
        for _, name in self._groups_to_reload:
            if name in seen_names:
                break
            seen_names.append(name)
        self._num_groups_per_layer = len(seen_names) if seen_names else 0

        if self._num_groups_per_layer == 0:
            self._name_layer_to_gid = {}
            self._num_layers = 0
            self._total_groups_per_layer = 0
            return

        self._num_layers = len(self._groups_to_reload) // self._num_groups_per_layer

        # Compute TOTAL groups-per-layer (including nested groups like mlp_norm
        # that get a gid via on_group_start_forward but don't appear in
        # _groups_to_reload because their commit is absorbed by the inner group).
        # This is needed by prefetch_previous_layer to map gid -> layer index,
        # since gids are assigned sequentially across ALL groups (including
        # non-offloaded ones), not just offloaded groups.
        # Use _group_id_to_name (which records every on_group_start_forward call)
        # to find the first repeated name — that gives the total per-layer count.
        all_seen_names = []
        for gid in sorted(self._group_id_to_name.keys()):
            n = self._group_id_to_name[gid]
            if n in all_seen_names:
                break
            all_seen_names.append(n)
        self._total_groups_per_layer = (
            len(all_seen_names) if all_seen_names else self._num_groups_per_layer
        )

        # Build (name, layer_idx) -> gid mapping
        self._name_layer_to_gid = {}
        for idx, (gid, name) in enumerate(self._groups_to_reload):
            layer_idx = idx // self._num_groups_per_layer
            self._name_layer_to_gid[(name, layer_idx)] = gid

    def prefetch_cross_layer(self, name, current_gid):
        """Prefetch the same-named module from the previous layer.

        Called from ``on_group_start_backward`` AFTER the current group's
        backward compute finishes.  For example, after L1-core_attn backward
        completes, this prefetches L0-core_attn.  The H2D transfer overlaps
        with the subsequent groups' backward compute (e.g., L1-qkv_linear),
        giving a full layer's worth of overlap time.

        Args:
            name: The group name (e.g., 'core_attn').
            current_gid: The group_id of the current group that just finished.
        """
        if not H2D_PREFETCH_ENABLED or not CROSS_LAYER_PREFETCH_ENABLED:
            return
        if self._num_groups_per_layer is None or self._num_groups_per_layer == 0:
            return

        # Find current layer index
        self._build_cross_layer_index()
        if self._num_layers is None or self._num_layers <= 1:
            return  # Only one layer, nothing to cross-prefetch

        # Determine which layer the current gid belongs to
        current_layer = None
        for (n, l), gid in self._name_layer_to_gid.items():
            if gid == current_gid and n == name:
                current_layer = l
                break

        if current_layer is None or current_layer == 0:
            return  # First layer or not found — no previous layer

        # Find the same-named module in the previous layer
        prev_layer = current_layer - 1
        target_gid = self._name_layer_to_gid.get((name, prev_layer))
        if target_gid is None:
            return

        # Skip if already issued
        if target_gid in self._reload_issued:
            return

        # Find the target group entry in _groups_to_reload
        target_entry = None
        target_idx = None
        for idx, entry in enumerate(self._groups_to_reload):
            if entry[0] == target_gid:
                target_entry = entry
                target_idx = idx
                break

        if target_entry is None:
            return  # Already consumed

        # Issue H2D on h2d_stream — sync rule #2 already satisfied by
        # commit_backward before this call.  Do NOT re-issue
        # h2d_stream.wait_stream(current_stream) here: the adjacent-group
        # prefetch (bulk_reload_next) may already be in-flight on h2d_stream,
        # and an extra wait_stream would serialize the cross-layer H2D behind
        # the compute stream, adding unnecessary stall and contending with the
        # adjacent-group prefetch for PCIe bandwidth.
        if self.bulk_reload_group(target_entry):
            # Remove from _groups_to_reload to avoid double-reload
            self._groups_to_reload.pop(target_idx)
            debug_rank(
                f"--prefetch_cross_layer: {name} L{current_layer}->L{prev_layer} "
                f"gid={target_gid}"
            )

    def prefetch_previous_layer(self, current_gid):
        """Prefetch ALL groups of the previous layer (last-layer-no-offload strategy).

        Called from ``on_group_commit_backward`` when LAST_LAYER_NO_OFFLOAD is
        enabled.  When backward enters a new layer (detected by the first
        commit_backward of that layer), this method issues H2D for every group
        belonging to the previous layer in one batch.

        The key difference from ``prefetch_cross_layer`` (which prefetches one
        same-named group from the previous layer) is that this prefetches ALL
        groups of the entire previous layer at once, giving a full layer's
        backward compute time for the transfers to complete.

        For example, with N=4 layers (0,1,2,3) and 7 groups per layer:
        - Layer 3 (last) is NOT offloaded → backward starts here, activations on GPU.
        - First commit_backward of layer 3 → prefetch ALL 7 groups of layer 2.
        - First commit_backward of layer 2 → prefetch ALL 7 groups of layer 1.
        - First commit_backward of layer 1 → prefetch ALL 7 groups of layer 0.
        - Layer 0 → no previous layer to prefetch.

        Args:
            current_gid: The group_id of the current group whose backward is
                about to start.
        """
        if not LAST_LAYER_NO_OFFLOAD_ENABLED:
            return

        self._build_cross_layer_index()
        if self._num_groups_per_layer is None or self._num_groups_per_layer == 0:
            return

        # Compute total layers including the last layer (which is NOT in
        # _groups_to_reload but IS in _group_id_to_name).
        # _num_layers from _build_cross_layer_index counts only offloaded layers
        # (layers 0..N-2).  The actual total is _num_layers + 1.
        total_layers = self._num_layers + 1  # +1 for the non-offloaded last layer
        if total_layers <= 1:
            return  # Only one layer, nothing to prefetch

        # Determine which layer the current gid belongs to.
        # The gid is assigned sequentially starting from 1:
        #   layer 0: gids 1 .. G
        #   layer 1: gids G+1 .. 2G
        #   ...
        #   layer N-1: gids (N-1)*G+1 .. N*G
        # where G = _total_groups_per_layer (ALL groups per layer, including
        # nested groups like mlp_norm that get a gid but don't appear in
        # _groups_to_reload).  Using _num_groups_per_layer (offloaded-only count)
        # would miscalculate the layer for higher gids.
        current_layer = (current_gid - 1) // self._total_groups_per_layer

        # Determine which offloaded layer to prefetch.
        # "previous layer" in terms of backward order (i.e., current_layer - 1).
        prev_layer = current_layer - 1
        if prev_layer < 0:
            return  # First layer — nothing to prefetch

        # prev_layer must be an offloaded layer (i.e., < total_layers - 1).
        # Since the last layer (total_layers - 1) is not offloaded, and
        # prev_layer < current_layer <= total_layers - 1, prev_layer is always
        # in range [0, total_layers - 2], which are all offloaded layers.

        # Check if we already prefetched this layer (avoid duplicate work).
        if self._last_prefetched_layer == prev_layer:
            return  # Already prefetched
        self._last_prefetched_layer = prev_layer

        debug_rank(
            f"--prefetch_previous_layer: starting L{current_layer}->L{prev_layer}"
        )

        # Sync rule #2: h2d_stream must wait for compute_stream ONCE before
        # issuing the batch of H2D transfers for the previous layer.
        # This wait_stream is placed here (after dedup check) instead of in
        # on_group_commit_backward, so that subsequent commit_backward calls
        # within the same layer do NOT insert redundant sync points on
        # h2d_stream that would stall in-flight H2D transfers.
        self.h2d_stream.wait_stream(torch.cuda.current_stream())

        # Collect all groups belonging to prev_layer from _name_layer_to_gid.
        # prev_layer is within [0, _num_layers - 1], so it's in the index.
        prev_layer_gids = []
        for (n, l), gid in self._name_layer_to_gid.items():
            if l == prev_layer:
                prev_layer_gids.append(gid)

        # Sort gids in DESCENDING order so that groups needed first by backward
        # (high gid = last group of the layer = first consumed in backward)
        # are transferred first.  Without this, the H2D order (ascending gid)
        # is the reverse of the backward consumption order, forcing the compute
        # stream to wait for ALL transfers to complete before starting.
        prev_layer_gids.sort(reverse=True)

        _do_ll_timing = (
            LAST_LAYER_TIMING_ENABLED
            and _timing_iter_count < LAST_LAYER_TIMING_MAX_ITERS
        )

        # ---- Last-layer timing: record H2D batch start on h2d_stream ----
        if _do_ll_timing:
            _h2d_start = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.h2d_stream):
                _h2d_start.record()

        # Issue H2D for each group in the previous layer
        reloaded_count = 0
        for target_gid in prev_layer_gids:
            if target_gid in self._reload_issued:
                reloaded_count += 1
                continue
            # Find the target entry in _groups_to_reload
            target_entry = None
            target_idx = None
            for idx, entry in enumerate(self._groups_to_reload):
                if entry[0] == target_gid:
                    target_entry = entry
                    target_idx = idx
                    break
            if target_entry is None:
                continue  # Already consumed
            if self.bulk_reload_group(target_entry):
                self._groups_to_reload.pop(target_idx)
                reloaded_count += 1
                debug_rank(
                    f"--prefetch_previous_layer: L{current_layer}->L{prev_layer} "
                    f"gid={target_gid} name={target_entry[1]}"
                )

        # ---- Last-layer timing: record H2D batch end on h2d_stream ----
        if _do_ll_timing:
            _h2d_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.h2d_stream):
                _h2d_end.record()
            self._timing_events.append(
                (
                    f"PREFETCH_H2D L{current_layer}->L{prev_layer} "
                    f"({reloaded_count} groups, gids={prev_layer_gids})",
                    _h2d_start,
                    _h2d_end,
                )
            )

        debug_rank(
            f"--prefetch_previous_layer: done L{prev_layer}, "
            f"reloaded={reloaded_count}/{len(prev_layer_gids)} "
            f"remaining_reload={len(self._groups_to_reload)}"
        )

        # If all groups reloaded, trigger cross-chunk handoff.
        if len(self._groups_to_reload) == 0:
            mgr = PipelineOffloadManager.get_instance()
            next_backward_chunk = mgr.front()
            if next_backward_chunk is not None:
                next_backward_chunk.pre_reload_last_layer()

    def bulk_reload_next(self):
        """
        Prefetch the next adjacent group from CPU to GPU.

        Called from ``on_group_commit_backward`` BEFORE the current group's
        backward compute starts.  The H2D transfer runs on ``h2d_stream``
        and overlaps with the upcoming backward compute on the default
        (compute) stream.

        Only one group is prefetched per call (adjacent-group prefetch).
        Multi-group lookahead was tested but degrades performance due to
        PCIe bandwidth contention with forward D2H offload in the 1F1B
        pipeline schedule.

        When all groups in this chunk are reloaded, pre-load the last layer
        of the next backward chunk (cross-chunk handoff).
        """
        debug_rank("--bulk_reload_next")

        if not H2D_PREFETCH_ENABLED or len(self._groups_to_reload) == 0:
            # Nothing left to reload — try next chunk handoff.
            if len(self._groups_to_reload) == 0:
                mgr = PipelineOffloadManager.get_instance()
                next_backward_chunk = mgr.front()
                if next_backward_chunk is not None:
                    next_backward_chunk.pre_reload_last_layer()
            return

        # ---- Adjacent-group prefetch ----
        # Pop the next group from the end of _groups_to_reload and reload it.
        # The list is ordered [L0_g0..L0_g6, L1_g0..L1_g6]; backward
        # consumes from the end (L1 last group first), so the next group
        # to be needed is always at the tail.
        while len(self._groups_to_reload) > 0:
            if self.bulk_reload_group(self._groups_to_reload[-1]):
                debug_rank(
                    f"--bulk_reload_next: prefetched {self._groups_to_reload[-1]}"
                )
                self._groups_to_reload.pop()
                break
            else:
                # Group already consumed by tensor_pop (sync path), skip
                self._groups_to_reload.pop()

        # If all groups reloaded, pre-load the next backward chunk's last layer.
        if len(self._groups_to_reload) == 0:
            mgr = PipelineOffloadManager.get_instance()
            next_backward_chunk = mgr.front()
            if next_backward_chunk is not None:
                next_backward_chunk.pre_reload_last_layer()

    def on_group_commit_backward(self, name, committed_gid):
        """
        Called BEFORE a layer group's backward compute starts (autograd reversal).

        In the backward flow (autograd reversal), commit_backward fires FIRST
        for each group, then the backward computation of the group body runs,
        then start_backward fires.

        Args:
            name: The group name (e.g. 'mlp_norm', 'expert_fc1').
            committed_gid: The actual group_id that was committed during
                forward.  Passed from the autograd ctx to fix the gid mismatch
                caused by nested groups (mlp_norm wrapping expert_fc1 + moe_act).

        This method:
          1. Ensures the correct chunk is active.
          2. Issues H2D reload of the CURRENT group if not already issued.
          3. Waits for the current group's reload to complete (sync rule #4).
          4. Triggers prefetch of the NEXT group's activations on h2d_stream,
              so the H2D transfer overlaps with THIS group's backward compute.

        There is NO fallback sync reload in tensor_pop.  Every group's H2D
        must be issued here (or by pre_reload_last_layer for the first group).

        Synchronization rules:
          #2: h2d_stream waits for compute_stream — issued before reload.
          #4: compute_stream waits for h2d reload of the current group.
        """
        debug_rank("--on_group_commit_backward")
        _t0 = time.time()
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        # Switch to this chunk if it's not already current
        if cur_backward_chunk is not self:
            PipelineOffloadManager.get_instance().pop()
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        assert cur_backward_chunk is self, "Chunk mismatch"

        cur_group_id = committed_gid

        # ---- Step 1: Ensure CURRENT group's H2D has been issued ----
        if LAST_LAYER_NO_OFFLOAD_ENABLED:
            # LAST_LAYER_NO_OFFLOAD strategy: groups fall into two categories:
            # (a) Non-offloaded groups (last layer): activations are already on
            #     GPU.  Just mark as "reload issued" to skip further processing.
            #     Do NOT call h2d_stream.wait_stream or bulk_reload_group — those
            #     would insert unnecessary sync points on h2d_stream and disrupt
            #     in-flight H2D prefetch transfers.
            # (b) Offloaded groups (other layers): their H2D was issued as a
            #     batch by prefetch_previous_layer in Step 3 of a prior
            #     commit_backward.  If somehow missed, issue a sync H2D here.
            if cur_group_id not in self._reload_issued:
                # Check if this group has any offloaded tensors (stored as tuple).
                # Non-offloaded groups (last layer) have plain GPU tensors.
                _has_offloaded_tensor = any(
                    isinstance(state, tuple)
                    for tag, state in self._tensor_tag_to_state.items()
                    if tag[0] == cur_group_id
                )
                if _has_offloaded_tensor:
                    # Offloaded group not yet prefetched — issue sync H2D.
                    self.h2d_stream.wait_stream(torch.cuda.current_stream())
                    group_info = self._group_id_to_name.get(cur_group_id)
                    if group_info is not None:
                        self.bulk_reload_group((cur_group_id, group_info))
                else:
                    # Non-offloaded group (last layer) — just mark as done.
                    self._reload_issued.add(cur_group_id)
        else:
            # Default path: sync rule #2 — h2d_stream must wait for previous
            # backward to finish before we can issue H2D.
            self.h2d_stream.wait_stream(torch.cuda.current_stream())
            # Issue H2D for current group if not already done (e.g. by
            # pre_reload_last_layer or a previous prefetch).
            group_info = self._group_id_to_name.get(cur_group_id)
            if group_info is not None and cur_group_id not in self._reload_issued:
                self.bulk_reload_group((cur_group_id, group_info))

        # ---- Step 2: Wait for current group's H2D to complete (sync rule #4) ----
        # Record CUDA events around the wait_event to measure stall time
        # for the adaptive profiler (deferred measurement — no synchronize here).
        if LAST_LAYER_NO_OFFLOAD_ENABLED:
            # In LAST_LAYER_NO_OFFLOAD mode, only look up by gid (no name fallback).
            # The name-indexed _reload_events can contain stale events from a
            # *different* layer's same-named group (e.g., Layer 0's core_attn event
            # would incorrectly match Layer 1's core_attn gid), causing the compute
            # stream to wait on an unrelated H2D transfer.
            reload_event = self._reload_events_by_id.get(cur_group_id)
        else:
            reload_event = self._reload_events_by_id.get(
                cur_group_id, self.get_reload_event(name)
            )

        _ap = get_adaptive_profiler()
        _stall_profiling = _ap is not None and _ap.is_stall_profiling_active()
        _do_ll_timing = (
            LAST_LAYER_TIMING_ENABLED
            and _timing_iter_count < LAST_LAYER_TIMING_MAX_ITERS
        )

        if reload_event is not None:
            # Record stall start event for adaptive profiler
            if _stall_profiling:
                _stall_start = torch.cuda.Event(enable_timing=True)
                _stall_start.record()

            # ---- Last-layer timing: record GPU-side stall duration ----
            if _do_ll_timing:
                _ll_stall_start = torch.cuda.Event(enable_timing=True)
                _ll_stall_start.record()

            if OFFLOAD_PROFILING:
                wait_start_time = time.time()
            torch.cuda.current_stream().wait_event(reload_event)
            if OFFLOAD_PROFILING:
                compute_wait_time = (time.time() - wait_start_time) * 1000
                profiler = get_profiler()
                if profiler is not None and compute_wait_time > 0.1:
                    profiler.record_reload(name, 0, 0, compute_wait_time)

            # Record stall end event and enqueue for deferred measurement
            if _stall_profiling:
                _stall_end = torch.cuda.Event(enable_timing=True)
                _stall_end.record()
                _ap.enqueue_stall_events(name, _stall_start, _stall_end)

            # ---- Last-layer timing: capture stall end ----
            if _do_ll_timing:
                _ll_stall_end = torch.cuda.Event(enable_timing=True)
                _ll_stall_end.record()
                self._timing_events.append(
                    (
                        f"STALL gid={cur_group_id} name={name}",
                        _ll_stall_start,
                        _ll_stall_end,
                    )
                )
        else:
            # No reload event — the group was not offloaded (e.g., last layer).
            # Record a zero-stall marker for completeness.
            if _do_ll_timing:
                _ll_no_stall = torch.cuda.Event(enable_timing=True)
                _ll_no_stall.record()
                self._timing_events.append(
                    (
                        f"NO_STALL gid={cur_group_id} name={name} (on-GPU)",
                        _ll_no_stall,
                        _ll_no_stall,
                    )
                )

        # Record backward compute start event (after stall, before backward compute)
        if _stall_profiling:
            self._bwd_compute_start_event = torch.cuda.Event(enable_timing=True)
            self._bwd_compute_start_event.record()
            self._bwd_compute_group_name = name

        # ---- Last-layer timing: record backward compute start ----
        if _do_ll_timing:
            self._ll_bwd_start = torch.cuda.Event(enable_timing=True)
            self._ll_bwd_start.record()
            self._ll_bwd_gid = cur_group_id
            self._ll_bwd_name = name

        # Save current group info for cross-layer prefetch in on_group_start_backward
        self._last_bwd_group_name = name
        self._last_bwd_group_gid = cur_group_id

        # ---- Step 3: Prefetch ----
        # Profiling guard: during stall profiling, disable prefetch so that
        # the profiler measures TRUE stall times (without overlap).  If
        # prefetch runs during profiling, measured stall drops to ~0 and
        # the optimizer would incorrectly conclude OFFLOAD is free, keeping
        # everything offloaded instead of switching strategies.
        if _ap is not None and not _ap.is_profiling_done():
            # Profiling in progress — skip prefetch to preserve accurate stall data
            pass
        elif LAST_LAYER_NO_OFFLOAD_ENABLED:
            # ---- Last-layer-no-offload: layer-granularity prefetch ----
            # Prefetch ALL groups of the previous layer at once.  The first
            # commit_backward of each layer triggers the prefetch; subsequent
            # groups within the same layer are deduplicated internally.
            # Note: h2d_stream.wait_stream is called INSIDE prefetch_previous_layer
            # only when a new layer actually needs prefetching (after dedup check),
            # to avoid inserting unnecessary sync points on h2d_stream.
            self.prefetch_previous_layer(cur_group_id)
        else:
            # ---- Default: adjacent-group prefetch ----
            # Issue H2D for the next group so it overlaps with this group's
            # backward compute.  h2d_stream already waited on compute_stream
            # above, so we can issue directly.
            self.bulk_reload_next()

        _elapsed = (time.time() - _t0) * 1000
        shape_log_rank(
            f"[COMMIT_BW_TIMING] group={name!r} gid={cur_group_id} "
            f"elapsed={_elapsed:.3f}ms remaining_reload={len(self._groups_to_reload)}"
        )

    def on_group_start_forward(self, name):
        """
        Called at the start of a layer group's forward pass.
        Increments group index and prepares for offloading.
        Also records forward compute start event for adaptive profiling.
        """
        debug_rank(f"--on_group_start_forward")
        self._offloaded_group_index = self._offloaded_group_index + 1
        self._tensor_count_current_group = 0
        self._groups_to_offload.append((self._offloaded_group_index, name))
        self._group_id_to_name[self._offloaded_group_index] = name

        # Record forward compute start event for adaptive profiler
        _ap = get_adaptive_profiler()
        if _ap is not None and _ap.is_stall_profiling_active():
            self._fwd_compute_start_event = torch.cuda.Event(enable_timing=True)
            self._fwd_compute_start_event.record()
            self._fwd_compute_group_name = name
        else:
            self._fwd_compute_start_event = None
            self._fwd_compute_group_name = None

    def on_group_start_backward(self):
        """
        Called AFTER a layer group's backward compute completes (autograd reversal).

        This hook now serves two purposes:
        1. Records backward compute end event for adaptive profiling.
        2. Triggers cross-layer prefetch: after the current group's backward
           compute finishes, prefetch the SAME module from the PREVIOUS layer.
           This gives the H2D transfer a full layer's worth of backward
           compute time to overlap with.
        """
        debug_rank("--on_group_start_backward")
        # Record backward compute end event for adaptive profiler
        _ap = get_adaptive_profiler()
        if (
            _ap is not None
            and _ap.is_stall_profiling_active()
            and hasattr(self, "_bwd_compute_start_event")
            and self._bwd_compute_start_event is not None
        ):
            _bwd_end = torch.cuda.Event(enable_timing=True)
            _bwd_end.record()
            _ap.enqueue_backward_compute_events(
                self._bwd_compute_group_name,
                self._bwd_compute_start_event,
                _bwd_end,
            )
            self._bwd_compute_start_event = None
            self._bwd_compute_group_name = None

        # ---- Last-layer timing: record backward compute duration ----
        if (
            LAST_LAYER_TIMING_ENABLED
            and _timing_iter_count < LAST_LAYER_TIMING_MAX_ITERS
            and hasattr(self, "_ll_bwd_start")
            and self._ll_bwd_start is not None
        ):
            _ll_bwd_end = torch.cuda.Event(enable_timing=True)
            _ll_bwd_end.record()
            self._timing_events.append(
                (
                    f"BWD_COMPUTE gid={self._ll_bwd_gid} name={self._ll_bwd_name}",
                    self._ll_bwd_start,
                    _ll_bwd_end,
                )
            )
            self._ll_bwd_start = None

        # ---- Cross-layer prefetch ----
        # After the current group's backward compute finishes, prefetch the
        # same-named module from the previous layer.  The H2D transfer on
        # h2d_stream will overlap with the next group's backward compute.
        if (
            CROSS_LAYER_PREFETCH_ENABLED
            and hasattr(self, "_last_bwd_group_name")
            and self._last_bwd_group_name is not None
        ):
            self.prefetch_cross_layer(
                self._last_bwd_group_name, self._last_bwd_group_gid
            )


class FineGrainedOffloadingGroupCommitFunction(torch.autograd.Function):
    """
    Identity operation that marks the end of a layer group for offload synchronization.
    Triggers offload during forward and synchronizes reload during backward.
    """

    @staticmethod
    def forward(ctx, *args):
        # pylint: disable=missing-function-docstring
        debug_rank("FineGrainedOffloadingGroupCommitFunction forward")

        forced_released_tensors = args[-1]
        name = args[-2]
        cpu_offload_handler = args[-3]
        tensor = args[:-3]
        committed_gid = cpu_offload_handler.on_group_commit_forward(
            forced_released_tensors
        )
        ctx.cpu_offload_handler = cpu_offload_handler
        ctx.name = name
        # Save the actual committed gid so backward uses the correct one.
        # Without this, nested groups (mlp_norm wrapping expert_fc1 + moe_act)
        # cause the simple decrement counter to assign wrong gids in backward.
        ctx.committed_gid = committed_gid

        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):
        # pylint: disable=missing-function-docstring
        debug_rank("FineGrainedOffloadingGroupCommitFunction backward")

        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward(ctx.name, ctx.committed_gid)
        return grad_output + (None, None, None)


def fine_grained_offloading_group_commit(*tensor, name, forced_released_tensors=[]):
    """
    Specify the tensors to be released after offloading.
    forced_released_tensors is a list of tensors to be released after offloading.
    The tensors will be untyped_storage().resize_(0) after offloading.
    Note: specify the tensors only when they are not automatically released by torch gc.
    """
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return FineGrainedOffloadingGroupCommitFunction.apply(
        *tensor, cur_forward_chunk, name, forced_released_tensors
    )


class FineGrainedOffloadingGroupStartFunction(torch.autograd.Function):
    """
    Identity operation that marks the start of a layer group for offload/reload.
    Prepares for offload during forward and triggers reload during backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler, name):
        # pylint: disable=missing-function-docstring
        ctx.cpu_offload_handler = cpu_offload_handler
        debug_rank("FineGrainedOffloadingGroupStartFunction forward")

        cpu_offload_handler.on_group_start_forward(name)
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        debug_rank("FineGrainedOffloadingGroupStartFunction backward")
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_start_backward()
        return grad_output, None, None


def fine_grained_offloading_group_start(tensor, name=None):
    """Mark the start of a layer group and prepare for offload/reload."""
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return FineGrainedOffloadingGroupStartFunction.apply(
        tensor, cur_forward_chunk, name
    )


def get_fine_grained_offloading_context(flag):
    """Get the fine-grained offload context"""
    return PipelineOffloadManager.get_instance() if flag else nullcontext()


def fine_grained_offloading_set_last_layer(is_last_layer):
    """Set the last layer flag."""
    PipelineOffloadManager.get_instance().set_last_layer(is_last_layer)


def fine_grained_offloading_init_chunk_handler(
    vp_size, vp_stage, min_offloaded_tensor_size
):
    """Initialize the chunk handler, called at the start of a microbatch forward pass."""
    PipelineOffloadManager.get_instance().init_model_chunk_offload_handler(
        vp_size, vp_stage, min_offloaded_tensor_size
    )


def fine_grained_offloading_reset():
    """Reset the chunk handler, called at the start of a training iteration."""
    PipelineOffloadManager.get_instance().reset()
