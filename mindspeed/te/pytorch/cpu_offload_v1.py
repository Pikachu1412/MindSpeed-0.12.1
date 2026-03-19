# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
#
# Adapted from NVIDIA TransformerEngine:
# https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/cpu_offload_v1.py
#
# This module provides a minimal but complete implementation of the TE cpu_offload_v1
# interface required by megatron's fine_grained_activation_offload.  The original
# implementation depends on CUDA-specific helpers (QuantizedTensorStorage, Float8Tensor)
# that are not available on Ascend NPU, so those paths are removed while keeping the
# public API and the CPUOffloadEnabled flag that Megatron reads.

"""Functionality for CPU offloading of tensors saved for backward pass."""
from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch

__all__ = ["get_cpu_offload_context"]

# Global flag read by megatron's fine_grained_activation_offload.__enter__/__exit__
# via:  from megatron.core.extensions.transformer_engine import cpu_offload
#       cpu_offload.CPUOffloadEnabled = True / False
CPUOffloadEnabled = False
CPUOffloadedLayer = False


def mark_activation_offload(*tensors):
    """Mark tensors as candidates for activation offloading.

    Sets the ``activation_offloading`` attribute on each tensor so that the
    offload handler's ``tensor_need_offloading_checker`` can identify them.
    """
    for tensor in tensors:
        if tensor is None:
            continue
        if isinstance(tensor, (torch.Tensor, torch.nn.Parameter)):
            tensor.activation_offloading = True


def is_cpu_offload_enabled() -> bool:
    """Return True if CPU offloading is currently active."""
    return CPUOffloadEnabled


def is_current_layer_offloaded() -> bool:
    """Return True if the current layer is being offloaded."""
    return CPUOffloadedLayer


class CpuOffloadSavedTensorHook:
    """Base context-manager that installs pack/unpack hooks for saved tensors.

    Subclasses must implement ``on_save_for_backward`` and ``on_get_saved_tensor``.
    """

    def __init__(self) -> None:
        self.inside_context = False

    def __enter__(self):
        global CPUOffloadEnabled
        CPUOffloadEnabled = True
        self.inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args: Any):
        global CPUOffloadEnabled
        CPUOffloadEnabled = False
        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        """Called when a tensor is saved for backward. Must be overridden."""
        raise NotImplementedError(
            "`on_save_for_backward` is not implemented in CpuOffloadSavedTensorHook. "
            "Inherit this class and implement your custom hooks."
        )

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """Called when a saved tensor is retrieved during backward. Must be overridden."""
        raise NotImplementedError(
            "`on_get_saved_tensor` is not implemented in CpuOffloadSavedTensorHook. "
            "Inherit this class and implement your custom hooks."
        )


class CpuOffloadHookWithOffloadHandler(CpuOffloadSavedTensorHook):
    """Context-manager that offloads/recovers tensors through an OffloadHandler."""

    def __init__(
        self,
        offload_handler: OffloadHandler,
        handler_extra_kwargs: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> None:
        if handler_extra_kwargs is None:
            handler_extra_kwargs = {}
        self.debug: bool = debug
        self.offload_handler: OffloadHandler = offload_handler
        self.handler_extra_kwargs: Dict[str, Any] = handler_extra_kwargs
        super().__init__()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        return self.offload_handler.tensor_push(tensor, **self.handler_extra_kwargs)

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        return self.offload_handler.tensor_pop(saved_state, **self.handler_extra_kwargs)


class OffloadHandler:
    """Base class for CPU offload handlers."""

    def __init__(self) -> None:
        pass

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        """Push a tensor for offloading. Returns an identifier used by tensor_pop."""
        raise NotImplementedError(
            "`tensor_push` is not implemented in OffloadHandler. "
            "Inherit this class and implement your custom tensor_push."
        )

    def tensor_pop(self, tensor_tag: Any, **kwargs) -> torch.Tensor:
        """Retrieve a previously pushed tensor."""
        raise NotImplementedError(
            "`tensor_pop` is not implemented in OffloadHandler. "
            "Inherit this class and implement your custom tensor_pop."
        )


class GroupCommitFunction(torch.autograd.Function):
    """Dummy op that triggers offload-handler synchronization at group boundaries."""

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        cpu_offload_handler.on_group_commit_forward()
        ctx.cpu_offload_handler = cpu_offload_handler
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        ctx.cpu_offload_handler.on_group_commit_backward()
        return grad_output, None


group_prefetch_offload_commit = GroupCommitFunction.apply


class SynchronizedGroupOffloadHandler(OffloadHandler):
    """Synchronous offload handler: D2H and H2D copies block computation."""

    def __init__(
        self,
        num_offload_group,
        tensor_need_offloading_checker=(lambda _: True),
        debug=False,
    ) -> None:
        super().__init__()
        self.num_offload_group = num_offload_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker
        self.debug = debug
        self.groupid_reset()

    def groupid_reset(self):
        """Reset group counters."""
        self.current_group, self.tensor_count_current_group = (0, 0)
        self.torch_tensor_count = 0
        self.tensor_tag_to_state = {}

    def on_group_commit_forward(self):
        """Advance to the next group."""
        self.current_group += 1
        self.tensor_count_current_group = 0

    def on_group_commit_backward(self):
        """Step back one group."""
        self.current_group -= 1
        assert self.current_group >= 0

    @staticmethod
    def offload(src_tensor, pin_memory=True):
        """Copy tensor to pinned CPU memory."""
        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        return (src_tensor.device, cpu_backup)

    @staticmethod
    def reload(state, non_blocking=None, copy_buffer=None):
        """Copy tensor back from CPU to its original device."""
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        if copy_buffer is None:
            return cpu_backup.to(dev, non_blocking=non_blocking)
        assert cpu_backup.size() == copy_buffer.size()
        copy_buffer.copy_(cpu_backup, non_blocking=non_blocking)
        return copy_buffer

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        """Push tensor; offload immediately if within the offload window."""
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        self.tensor_count_current_group += 1
        assert tensor_tag not in self.tensor_tag_to_state
        if (
            self.current_group < self.num_offload_group
            and self.tensor_need_offloading_checker(tensor)
        ):
            self.tensor_tag_to_state[tensor_tag] = self.offload(tensor)
        else:
            self.tensor_tag_to_state[tensor_tag] = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Pop tensor; reload from CPU if it was offloaded."""
        assert tensor_tag in self.tensor_tag_to_state
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            return self.reload(state)
        return state


class AsyncDoubleBufferGroupOffloadHandler(SynchronizedGroupOffloadHandler):
    """Async offload handler with optional double-buffering for better overlap."""

    def __init__(
        self,
        num_offload_group,
        num_model_group,
        tensor_need_offloading_checker=(lambda t: True),
        double_buffering=False,
        debug=False,
    ) -> None:
        super().__init__(
            num_offload_group=num_offload_group,
            tensor_need_offloading_checker=tensor_need_offloading_checker,
            debug=debug,
        )
        self.num_layers = num_model_group
        self.tensor_tag_to_buf = {}
        self.offloaded_group_count = 0
        self.layer_window_map = {}

        self.double_buffering = double_buffering
        self.reload_double_buffer = [[], []]
        self.double_buffer_created = False

        # Distribute offload windows evenly across layers
        constant = 0
        for i in range(self.num_offload_group):
            self.layer_window_map[i] = ((self.num_layers // self.num_offload_group) * (i + 1)) - 1
            if i < (self.num_layers % self.num_offload_group):
                self.layer_window_map[i] += i + 1
                constant = i + 1
            else:
                self.layer_window_map[i] += constant

        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        global CPUOffloadedLayer

        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )

        if not torch_stray_tensor:
            tensor_tag = (self.current_group, self.tensor_count_current_group)
            self.tensor_count_current_group += 1
            assert tensor_tag not in self.tensor_tag_to_state
            self.tensor_tag_to_state[tensor_tag] = tensor

            if (
                self.current_group < self.num_offload_group
                and self.tensor_need_offloading_checker(tensor)
            ):
                self.tensor_tag_to_buf[tensor_tag] = tensor
                CPUOffloadedLayer = True
        else:
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self.tensor_tag_to_state[tensor_tag] = tensor

        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Pop tensor; reload from CPU if it was offloaded."""
        assert tensor_tag in self.tensor_tag_to_state
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        self.tensor_tag_to_buf.pop(tensor_tag, None)
        assert not isinstance(tensor, tuple)
        return tensor

    def bulk_offload_group(self, group_to_offload):
        """Offload all tensors belonging to the given group."""
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_to_offload and not isinstance(state, tuple):
                    if self.tensor_need_offloading_checker(state):
                        self.tensor_tag_to_state[tensor_tag] = self.offload(state)

    def synchronize_on_group_commit_forward(self, current_group):
        """Trigger offload and synchronization at the right layer boundaries."""
        global CPUOffloadedLayer

        if current_group == 0:
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            if not self.double_buffer_created:
                for tensor_tag, buf in self.tensor_tag_to_buf.items():
                    self.reload_double_buffer[0].append(
                        torch.empty_like(buf) if self.double_buffering else None
                    )
            self.bulk_offload_group(current_group)

        if self.layer_window_map.get(self.offloaded_group_count) == current_group:
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_stream(self.d2h_stream)

            for tensor_tag, tensor_buf in self.tensor_tag_to_buf.items():
                if tensor_tag[0] == self.offloaded_group_count:
                    self.tensor_tag_to_buf[tensor_tag] = None

            if self.offloaded_group_count < (self.num_offload_group - 1):
                self.bulk_offload_group(self.offloaded_group_count + 1)

            self.offloaded_group_count += 1

        if current_group == (self.num_offload_group - 1):
            CPUOffloadedLayer = False

        if not self.double_buffer_created and current_group == (self.num_layers - 1):
            for buf in self.reload_double_buffer[0]:
                self.reload_double_buffer[1].append(
                    torch.empty_like(buf) if self.double_buffering else None
                )
            self.double_buffer_created = True

    def on_group_commit_forward(self):
        """Called at the end of each layer's forward pass."""
        self.synchronize_on_group_commit_forward(self.current_group)
        super().on_group_commit_forward()

    def bulk_reload_group(self, group_to_reload):
        """Reload all tensors belonging to the given group from CPU."""
        assert group_to_reload < self.num_offload_group
        buffer_idx = 0
        double_buffer_idx = group_to_reload % 2
        main_stream = torch.cuda.current_stream()

        with torch.cuda.stream(self.h2d_stream):
            for tensor_label, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_to_reload and isinstance(state, tuple):
                    if self.double_buffering:
                        reload_buffer = self.reload_double_buffer[double_buffer_idx][buffer_idx]
                    else:
                        with torch.cuda.stream(main_stream):
                            reload_buffer = torch.empty_like(
                                state[1], device=torch.cuda.current_device()
                            )
                    recovered_tensor = self.reload(state, True, reload_buffer)
                    buffer_idx += 1
                    self.tensor_tag_to_state[tensor_label] = recovered_tensor

    def on_group_commit_backward(self):
        """Called at the start of each layer's backward pass."""
        self.current_group -= 1
        assert self.current_group >= 0

        if self.layer_window_map.get(self.offloaded_group_count - 1) == self.current_group:
            self.h2d_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_stream(self.h2d_stream)
            self.bulk_reload_group(self.offloaded_group_count - 1)
            self.offloaded_group_count -= 1 if self.offloaded_group_count > 1 else 0

        if self.current_group == 0:
            torch.cuda.current_stream().wait_stream(self.h2d_stream)
            self.offloaded_group_count = 0


def get_cpu_offload_context(
    enabled: bool = False,
    num_layers: int = 1,
    model_layers: int = 1,
    offload_activations: bool = True,
    offload_weights: bool = False,
    double_buffering: bool = False,
):
    """Return a (context, synchronizer) pair for CPU activation offloading.

    This is the v1 code path used when ``NVTE_CPU_OFFLOAD_V1=1``.  On Ascend NPU
    the interface is the same as the NVIDIA implementation but without FP8/CUDA
    quantized-tensor support.

    Parameters
    ----------
    enabled : bool
        When True, CPU offloading is active.
    num_layers : int
        Number of layers to offload.
    model_layers : int
        Total number of layers in the model.
    offload_activations : bool
        Offload activation tensors (the only supported mode on NPU).
    offload_weights : bool
        Deprecated; has no effect.
    double_buffering : bool
        Use double-buffered reload for better overlap.
    """
    if not offload_weights and not offload_activations:
        raise ValueError(
            "CPU Offloading is enabled while it is not "
            "mentioned what to offload (weights/activations)"
        )

    if offload_weights:
        import warnings
        warnings.warn(
            "Offloading weights is deprecated. Using offload_weights=True does not have any"
            " effect.",
            DeprecationWarning,
        )
        if not offload_activations:
            return nullcontext(), lambda x: x

    def tensor_need_offloading_checker_activations(tensor):
        return hasattr(tensor, "activation_offloading")

    cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
        num_offload_group=num_layers,
        num_model_group=model_layers,
        tensor_need_offloading_checker=tensor_need_offloading_checker_activations,
        double_buffering=double_buffering,
    )

    def group_prefetch_offload_commit_async(tensor):
        return group_prefetch_offload_commit(tensor, cpu_offload_handler)

    if enabled:
        return (
            CpuOffloadHookWithOffloadHandler(offload_handler=cpu_offload_handler),
            group_prefetch_offload_commit_async,
        )
    return nullcontext(), group_prefetch_offload_commit_async
