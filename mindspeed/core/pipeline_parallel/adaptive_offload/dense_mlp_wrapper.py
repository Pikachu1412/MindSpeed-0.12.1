from functools import wraps

import torch

from mindspeed.core.pipeline_parallel.adaptive_offload import fine_grained_activation_offload as runtime


def _storage_key(tensor):
    return tensor.device, tensor.untyped_storage().data_ptr()


def _activation_saved_tensor_hooks(module, manager):
    parameter_storages = {_storage_key(parameter) for parameter in module.parameters()}

    def pack(tensor):
        if isinstance(tensor, torch.nn.Parameter):
            return tensor
        if _storage_key(tensor) in parameter_storages:
            return tensor.detach()
        return manager.on_save_for_backward(tensor)

    def unpack(saved_state):
        if isinstance(saved_state, torch.Tensor):
            return saved_state
        return manager.on_get_saved_tensor(saved_state)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def dense_mlp_forward_wrapper(original_forward):
    @wraps(original_forward)
    def wrapper(self, hidden_states, *args, **kwargs):
        if not getattr(self, '_offload_dense_mlp', False) or not torch.is_grad_enabled():
            return original_forward(self, hidden_states, *args, **kwargs)

        if runtime.ADAPTIVE_OFFLOAD_ENABLED:
            raise RuntimeError('dense_mlp offload requires MEGATRON_ADAPTIVE_OFFLOAD=0 until its joint-policy cost model is supported')

        manager = runtime.PipelineOffloadManager.get_instance()
        hidden_states = runtime.fine_grained_offloading_group_start(hidden_states, name='dense_mlp')
        with runtime.get_fine_grained_offloading_context(True), _activation_saved_tensor_hooks(self, manager):
            output, bias = original_forward(self, hidden_states, *args, **kwargs)
        (output,) = runtime.fine_grained_offloading_group_commit(output, name='dense_mlp')
        return output, bias

    return wrapper
