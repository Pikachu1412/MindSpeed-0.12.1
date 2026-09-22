import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as functional

from mindspeed.core.pipeline_parallel.adaptive_offload import dense_mlp_wrapper as dense
from mindspeed.core.pipeline_parallel.adaptive_offload.transformer_layer_wrapper import transformer_layer_init_wrapper
from mindspeed.features_manager.memory.adaptive_offload import AdaptiveOffloadFeature


class SavedTensorRecorder:
    def __init__(self):
        self.inside_context = False
        self.saved = {}
        self.received = []

    def __enter__(self):
        self.inside_context = True
        return self

    def __exit__(self, *args):
        self.inside_context = False

    def on_save_for_backward(self, tensor):
        assert self.inside_context
        tag = (len(self.received),)
        self.received.append(tensor)
        self.saved[tag] = tensor.detach().clone()
        return tag

    def on_get_saved_tensor(self, tag):
        return self.saved.pop(tag)


class DenseTestMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 24)
        self.fc2 = torch.nn.Linear(12, 8)
        self.output_bias = torch.nn.Parameter(torch.randn(8))
        self._offload_dense_mlp = True

    def forward(self, hidden_states, per_token_scale=None):
        gate, value = self.fc1(hidden_states).chunk(2, dim=-1)
        activated = functional.silu(gate) * value
        if per_token_scale is not None:
            activated = activated * per_token_scale.unsqueeze(-1)
        return self.fc2(activated), self.output_bias


class TestDenseMLPOffload(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.manager = SavedTensorRecorder()
        self.get_manager = self.stack.enter_context(patch.object(dense.runtime.PipelineOffloadManager, 'get_instance', return_value=self.manager))
        self.start = self.stack.enter_context(patch.object(dense.runtime, 'fine_grained_offloading_group_start', side_effect=lambda tensor, **kwargs: tensor))
        self.commit = self.stack.enter_context(patch.object(dense.runtime, 'fine_grained_offloading_group_commit', side_effect=lambda *tensors, **kwargs: tensors))
        self.stack.enter_context(patch.object(dense.runtime, 'get_fine_grained_offloading_context', return_value=self.manager))
        self.stack.enter_context(patch.object(dense.runtime, 'ADAPTIVE_OFFLOAD_ENABLED', False))

    def test_dense_forward_input_and_all_parameter_gradients_match(self):
        torch.manual_seed(2026)
        module = DenseTestMLP()
        hidden = torch.randn(13, 8, requires_grad=True)
        scale = torch.randn(13, requires_grad=True)
        reference, reference_bias = module(hidden, per_token_scale=scale)
        (reference + reference_bias).square().mean().backward()
        reference_hidden_grad = hidden.grad.clone()
        reference_scale_grad = scale.grad.clone()
        reference_parameter_grads = {name: parameter.grad.clone() for name, parameter in module.named_parameters()}
        module.zero_grad(set_to_none=True)
        hidden.grad = None
        scale.grad = None
        output, bias = dense.dense_mlp_forward_wrapper(DenseTestMLP.forward)(module, hidden, per_token_scale=scale)
        (output + bias).square().mean().backward()
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        torch.testing.assert_close(hidden.grad, reference_hidden_grad, rtol=0, atol=0)
        torch.testing.assert_close(scale.grad, reference_scale_grad, rtol=0, atol=0)
        for name, parameter in module.named_parameters():
            torch.testing.assert_close(parameter.grad, reference_parameter_grads[name], rtol=0, atol=0)
        self.assertIs(bias, module.output_bias)
        self.assertTrue(self.manager.received)
        parameter_storages = {dense._storage_key(parameter) for parameter in module.parameters()}
        self.assertTrue(all(dense._storage_key(tensor) not in parameter_storages for tensor in self.manager.received))
        self.assertEqual(self.manager.saved, {})
        self.start.assert_called_once_with(hidden, name='dense_mlp')
        self.assertEqual(self.commit.call_args.kwargs, {'name': 'dense_mlp'})

    def test_parameter_aliases_are_resident_without_mutating_exclusion_flags(self):
        module = DenseTestMLP()
        alias = module.fc1.weight.t()
        activation = torch.randn_like(alias)
        hooks = dense._activation_saved_tensor_hooks(module, self.manager)
        with self.manager, hooks:
            resident = hooks.pack_hook(alias)
            tag = hooks.pack_hook(activation)
            self.assertEqual(resident.data_ptr(), alias.data_ptr())
            self.assertEqual(resident.stride(), alias.stride())
            self.assertFalse(resident.requires_grad)
            self.assertIs(hooks.unpack_hook(resident), resident)
            torch.testing.assert_close(hooks.unpack_hook(tag), activation, rtol=0, atol=0)
        self.assertEqual(len(self.manager.received), 1)
        self.assertFalse(hasattr(alias, 'offloading_activation'))

    def test_parameter_object_and_main_grad_metadata_are_preserved(self):
        module = DenseTestMLP()
        parameter = module.fc1.weight
        parameter.main_grad = torch.zeros_like(parameter)
        hooks = dense._activation_saved_tensor_hooks(module, self.manager)
        with self.manager, hooks:
            resident = hooks.pack_hook(parameter)
            self.assertIs(resident, parameter)
            self.assertIs(hooks.unpack_hook(resident).main_grad, parameter.main_grad)
        self.assertEqual(self.manager.received, [])

    def test_small_and_explicitly_excluded_activations_reach_existing_filter(self):
        module = DenseTestMLP()
        small = torch.ones(1)
        excluded = torch.ones(32)
        excluded.offloading_activation = False
        hooks = dense._activation_saved_tensor_hooks(module, self.manager)
        with self.manager, hooks:
            hooks.pack_hook(small)
            hooks.pack_hook(excluded)
        self.assertIs(self.manager.received[0], small)
        self.assertIs(self.manager.received[1], excluded)
        self.assertFalse(excluded.offloading_activation)

    def test_disabled_forward_is_exact_passthrough(self):
        module = SimpleNamespace(_offload_dense_mlp=False)
        result = (torch.ones(1), None)
        original = Mock(return_value=result)
        actual = dense.dense_mlp_forward_wrapper(original)(module, 'input', 'scale', option=True)
        self.assertIs(actual, result)
        original.assert_called_once_with(module, 'input', 'scale', option=True)
        self.get_manager.assert_not_called()

    def test_unmarked_mlp_is_not_enabled(self):
        original = Mock(return_value=('output', None))
        self.assertEqual(dense.dense_mlp_forward_wrapper(original)(SimpleNamespace(), 'input'), ('output', None))
        self.get_manager.assert_not_called()

    def test_no_grad_forward_has_no_offload_lifecycle(self):
        module = DenseTestMLP()
        with torch.no_grad():
            dense.dense_mlp_forward_wrapper(DenseTestMLP.forward)(module, torch.randn(13, 8))
        self.get_manager.assert_not_called()
        self.start.assert_not_called()
        self.commit.assert_not_called()

    def test_forward_exception_exits_saved_tensor_context(self):
        module = DenseTestMLP()
        with self.assertRaisesRegex(RuntimeError, 'forward failed'):
            dense.dense_mlp_forward_wrapper(Mock(side_effect=RuntimeError('forward failed')))(module, torch.randn(13, 8))
        self.assertFalse(self.manager.inside_context)
        self.commit.assert_not_called()

    def test_runtime_rejects_unmodeled_adaptive_policy(self):
        with patch.object(dense.runtime, 'ADAPTIVE_OFFLOAD_ENABLED', True):
            with self.assertRaisesRegex(RuntimeError, 'MEGATRON_ADAPTIVE_OFFLOAD=0'):
                dense.dense_mlp_forward_wrapper(Mock())(DenseTestMLP(), torch.randn(13, 8))
        self.get_manager.assert_not_called()

    def test_layer_init_only_marks_selected_dense_mlp(self):
        from megatron.core.transformer.identity_op import IdentityOp
        from megatron.core.transformer.moe.moe_layer import BaseMoELayer

        def initialize(layer, config, mlp):
            layer.config = config
            layer.mlp = mlp
            layer.input_layernorm = Mock(spec=IdentityOp)
            layer.pre_mlp_layernorm = Mock(spec=IdentityOp)

        wrapped = transformer_layer_init_wrapper(initialize)
        for modules, is_moe, enabled in ((['dense_mlp'], False, True), ([], False, False), (['dense_mlp'], True, False)):
            with self.subTest(modules=modules, is_moe=is_moe):
                layer = SimpleNamespace()
                mlp = Mock(spec=BaseMoELayer) if is_moe else DenseTestMLP()
                config = SimpleNamespace(fine_grained_activation_offloading=True, offload_modules=modules)
                wrapped(layer, config, mlp)
                self.assertEqual(mlp._offload_dense_mlp, enabled)

    def test_feature_rejects_unsupported_adaptive_and_checkpointing(self):
        feature = AdaptiveOffloadFeature()
        args = SimpleNamespace(fine_grained_activation_offloading=True, transformer_impl='transformer_engine', offload_modules=['dense_mlp'], recompute_granularity=None, recompute_activation_function=False)
        with patch.dict(os.environ, {'MEGATRON_ADAPTIVE_OFFLOAD': '1'}):
            with self.assertRaisesRegex(ValueError, 'MEGATRON_ADAPTIVE_OFFLOAD=0'):
                feature.validate_args(args)
        with patch.dict(os.environ, {'MEGATRON_ADAPTIVE_OFFLOAD': '0'}):
            for granularity, activation in (('full', False), ('selective', False), (None, True)):
                args.recompute_granularity = granularity
                args.recompute_activation_function = activation
                with self.assertRaisesRegex(ValueError, 'checkpointing'):
                    feature.validate_args(args)

    def test_feature_accepts_fixed_offload_and_preserves_existing_modes(self):
        feature = AdaptiveOffloadFeature()
        args = SimpleNamespace(fine_grained_activation_offloading=True, transformer_impl='transformer_engine', offload_modules=['dense_mlp'])
        with patch.dict(os.environ, {'MEGATRON_ADAPTIVE_OFFLOAD': '0'}):
            feature.validate_args(args)
        args.offload_modules = ['expert_fc1', 'moe_act']
        args.recompute_granularity = 'full'
        with patch.dict(os.environ, {'MEGATRON_ADAPTIVE_OFFLOAD': '1'}):
            feature.validate_args(args)
