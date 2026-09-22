import copy
import json
import os
import tempfile
import unittest
from dataclasses import replace
from contextlib import ExitStack
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch.utils.checkpoint import checkpoint

from mindspeed.core.pipeline_parallel.adaptive_offload import auto_activation_memory as automatic
from mindspeed.core.pipeline_parallel.adaptive_offload import fine_grained_activation_offload as runtime
from mindspeed.core.pipeline_parallel.adaptive_offload.unified_offload_optimizer import (
    GroupProfile, MemoryBudget, ModuleProfile, ScheduleConfig, UnifiedOffloadOptimizer,
)


class Recorder:
    def __init__(self):
        self.saved = {}
        self.counter = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def on_save_for_backward(self, tensor):
        self.counter += 1
        self.saved[self.counter] = tensor.detach().clone()
        return (self.counter,)

    def on_get_saved_tensor(self, tag):
        return self.saved.pop(tag[0])


class ToyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 24)
        self.fc2 = torch.nn.Linear(12, 8)
        self.bias = torch.nn.Parameter(torch.randn(8))

    def forward(self, inputs, *, scale, metadata=7):
        gate, value = self.fc1(inputs).chunk(2, dim=-1)
        return {'output': self.fc2(torch.nn.functional.silu(gate) * value) * scale,
                'bias': self.bias, 'metadata': metadata}


class TestAutomaticActivationMemory(unittest.TestCase):
    def cache_fixture(self):
        from mindspeed.core.pipeline_parallel.adaptive_offload import adaptive_memory_profiler as profiling

        profiler = profiling.AdaptiveMemoryProfiler()
        profiler._is_stall_profile_rank = True
        profiler._auto_module_specs = {'layer1.dense_mlp': {'layer': 1, 'kind': 'dense_mlp', 'order': 0, 'is_moe': False}}
        profiler._memory_telemetry.update(baseline_peak_bytes=100, capacity_bytes=1000, samples=4)
        profiler._offload_group_stats['layer1.dense_mlp'] = profiling.OffloadGroupStats(
            group_name='layer1.dense_mlp', total_offload_bytes=10, offload_bytes_sample_count=4)
        profiler._get_or_create_layer(1, False).modules['layer1.dense_mlp'] = profiling.LayerModuleStats(
            compute_time_ms=1, sample_count=4, recompute_sample_count=4, recompute_time_ms=1, input_bytes=5)
        candidate = {'schema': 2, 'signature': 'fixture', 'module_specs': profiler._auto_module_specs,
                     'payload': profiler._profile_payload(),
                     'input_signatures': {'layer1.dense_mlp': [[[4, 8], 'torch.float32', 'cpu']]},
                     'reference_validation': {'samples': [{'iteration': 6, 'peak_bytes': 100}, {'iteration': 7, 'peak_bytes': 100}]}}
        return profiler, candidate

    def test_cache_decodes_before_collective_availability(self):
        profiler, candidate = self.cache_fixture()
        payload, groups, layers, pcie, signatures, reference = automatic._decode_profile_cache(candidate, 'fixture', profiler)
        self.assertEqual(groups['layer1.dense_mlp'].total_offload_bytes, 10)
        self.assertEqual(layers[1].modules['layer1.dense_mlp'].recompute_sample_count, 4)
        self.assertEqual(payload['memory']['samples'], 4)
        self.assertEqual(len(reference['samples']), 2)

    def test_invalid_cache_is_rejected_before_it_mutates_profiler(self):
        for invalid in ('nan', 'unknown_field', 'missing_replay', 'missing_reference', 'missing_signature', 'wrong_schema'):
            with self.subTest(invalid=invalid):
                profiler, candidate = self.cache_fixture()
                if invalid == 'nan':
                    candidate['payload']['memory']['baseline_peak_bytes'] = float('nan')
                elif invalid == 'unknown_field':
                    candidate['payload']['profile']['groups']['layer1.dense_mlp']['bad_field'] = 1
                elif invalid == 'missing_replay':
                    candidate['payload']['profile']['layers'][1]['modules']['layer1.dense_mlp']['recompute_sample_count'] = 0
                elif invalid == 'missing_reference':
                    candidate['reference_validation'] = None
                elif invalid == 'missing_signature':
                    candidate['input_signatures'] = {}
                else:
                    candidate['schema'] = 1
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    automatic._decode_profile_cache(candidate, 'fixture', profiler)
                self.assertFalse(profiler._profiling_done)

    def test_cache_hit_remeasures_capacity_without_restoring_a_plan(self):
        profiler, candidate = self.cache_fixture()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(automatic.Path, 'cwd', return_value=Path(directory)), \
             patch.object(automatic, 'profile_fingerprint', return_value='fixture'), \
             patch.object(profiler, '_init_pp_rank_profiling'), \
             patch.object(profiler, '_sample_memory_capacity') as measure, \
             patch.dict(os.environ, {'AUTO_ACTIVATION_MEMORY': '1'}):
            path = Path(directory) / '.mindspeed/activation_profiles/fixture.rank0.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(candidate))
            automatic.initialize_profile_cache(profiler, [], SimpleNamespace())
            measure.assert_called_once()
            self.assertEqual(profiler._memory_telemetry['capacity_bytes'], 0)
            self.assertTrue(profiler._auto_cache_loaded)
            self.assertTrue(profiler._auto_baseline_validated)
            self.assertFalse(profiler._optimization_applied)
            self.assertIsNone(profiler._plan)

    def test_bad_cache_resumes_cold_profile_relative_to_checkpoint_iteration(self):
        profiler, candidate = self.cache_fixture()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(automatic.Path, 'cwd', return_value=Path(directory)), \
             patch.object(automatic, 'profile_fingerprint', return_value='fixture'), \
             patch.object(profiler, '_init_pp_rank_profiling'), \
             patch.dict(os.environ, {'AUTO_ACTIVATION_MEMORY': '1'}):
            path = Path(directory) / '.mindspeed/activation_profiles/fixture.rank0.json'
            path.parent.mkdir(parents=True)
            path.write_text('{broken')
            automatic.initialize_profile_cache(profiler, [], SimpleNamespace(iteration=50))
            self.assertEqual(profiler._warmup_skip_iters, 52)
            self.assertFalse(profiler._profiling_done)

    def test_reprofiling_recalibrates_recompute_reference_and_cache(self):
        profiler, candidate = self.cache_fixture()
        profiler._plan = SimpleNamespace()
        profiler._auto_baseline_validated = True
        profiler._auto_cache_saved = True
        profiler._auto_reference_validation = candidate['reference_validation']
        with patch.dict(os.environ, {'AUTO_ACTIVATION_MEMORY': '1'}):
            profiler._start_reoptimization(72)
        self.assertIsNone(profiler._plan)
        self.assertFalse(profiler._auto_baseline_validated)
        self.assertFalse(profiler._auto_cache_saved)
        self.assertFalse(hasattr(profiler, '_auto_reference_validation'))
        self.assertFalse(profiler._layer_stats)
        self.assertEqual(profiler._memory_telemetry['baseline_peak_bytes'], 0)

    def test_unoffloadable_activations_and_immediate_last_layer_are_budgeted(self):
        solver = self.solver()
        solver.schedule = replace(solver.schedule, baseline_recompute=True)
        solver.groups = {name: replace(group, resident_mb=20) for name, group in solver.groups.items()}
        actions = dict.fromkeys(solver.groups, 'OFFLOAD')
        self.assertEqual(solver.evaluate(actions).predicted_peak_mb, 240)
        solver.schedule = replace(solver.schedule, immediate_backward_keep=True)
        self.assertEqual(solver.evaluate(actions).predicted_peak_mb, 340)
        self.assertEqual(solver.evaluate(dict.fromkeys(solver.groups, 'RECOMPUTE')).predicted_peak_mb, 100)

    def test_optimizer_floor_and_activation_phase_are_not_added_together(self):
        solver = self.solver()
        solver.schedule = replace(solver.schedule, baseline_recompute=True)
        solver.memory = replace(solver.memory, activation_baseline_peak_mb=40)
        self.assertEqual(solver.evaluate(dict.fromkeys(solver.groups, 'RECOMPUTE')).predicted_peak_mb, 100)
        self.assertEqual(solver.evaluate({'layer0.dense_mlp': 'KEEP', 'layer1.dense_mlp': 'RECOMPUTE'}).predicted_peak_mb, 140)

    def test_post_profile_reference_preserves_separate_phase_peak(self):
        profiler = SimpleNamespace(_current_iter=5, _memory_telemetry={'baseline_peak_bytes': 100},
                                   _iteration_peak_bytes=100, _activation_phase_peak_bytes=50,
                                   _pp_rank=0, is_profiling_done=lambda: True)
        with patch.dict(os.environ, {'AUTO_ACTIVATION_MEMORY': '1'}):
            automatic.advance_reference_validation(profiler)
            profiler._current_iter = 6
            automatic.advance_reference_validation(profiler)
            profiler._current_iter = 7
            automatic.advance_reference_validation(profiler)
        self.assertEqual(profiler._memory_telemetry['baseline_peak_bytes'], 100)
        self.assertEqual(profiler._memory_telemetry['activation_baseline_peak_bytes'], 50)

    def test_phase_peak_is_reset_at_each_training_iteration(self):
        profiler, candidate = self.cache_fixture()
        profiler._profiling_done = True
        profiler._activation_phase_peak_bytes = 999
        with patch.object(torch.cuda, 'is_available', return_value=False):
            profiler.on_iteration_start(9)
        self.assertEqual(profiler._activation_phase_peak_bytes, 0)

    def test_previous_device_peak_is_reset_before_new_iteration_capture(self):
        profiler, candidate = self.cache_fixture()
        profiler._profiling_done = True
        events = []
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'reset_peak_memory_stats', side_effect=lambda: events.append('reset')), \
             patch.object(profiler, '_capture_memory_peak', side_effect=lambda **kwargs: events.append('capture')), \
             patch.object(profiler, '_sample_memory_capacity'):
            profiler.on_iteration_start(9)
        self.assertEqual(events, ['reset', 'capture'])

    def test_fingerprint_changes_with_moe_routing_topk(self):
        model = [ToyModule()]
        with patch.object(torch.cuda, 'is_available', return_value=False):
            first = automatic.profile_fingerprint(model, SimpleNamespace(moe_router_topk=2))
            second = automatic.profile_fingerprint(model, SimpleNamespace(moe_router_topk=8))
        self.assertNotEqual(first, second)

    def test_structurally_corrupt_cache_is_a_cold_miss(self):
        profiler, candidate = self.cache_fixture()
        candidate['payload']['profile']['groups'] = []
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(automatic.Path, 'cwd', return_value=Path(directory)), \
             patch.object(automatic, 'profile_fingerprint', return_value='fixture'), \
             patch.object(profiler, '_init_pp_rank_profiling'), \
             patch.dict(os.environ, {'AUTO_ACTIVATION_MEMORY': '1'}):
            path = Path(directory) / '.mindspeed/activation_profiles/fixture.rank0.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(candidate))
            automatic.initialize_profile_cache(profiler, [], SimpleNamespace(iteration=50))
            self.assertEqual(profiler._warmup_skip_iters, 52)
            self.assertFalse(profiler._profiling_done)

    def test_small_tensor_measurements_do_not_change_offload_filter(self):
        profiler, candidate = self.cache_fixture()
        checker = Mock(return_value=False)
        manager = SimpleNamespace(cur_forward_chunk=lambda: SimpleNamespace(tensor_need_offloading_checker=checker))
        with patch.object(runtime.PipelineOffloadManager, 'get_instance', return_value=manager):
            module = ToyModule()
            tensor = torch.ones(4, 8, requires_grad=True)
            with automatic.observe_saved_activations(module, profiler, 'layer1.dense_mlp'):
                output = module.fc1(tensor).square().sum()
            output.backward()
        self.assertGreater(profiler._offload_group_stats['layer1.dense_mlp'].non_offloadable_bytes, 0)
        self.assertTrue(checker.called)

    def solver(self, count=2, budget=230, max_exact=10000):
        groups = {f'layer{index}.dense_mlp': GroupProfile(100, 5, 10, 10, 10) for index in range(count)}
        modules = {name: ModuleProfile(1 if index == 0 else 100, 5, 20) for index, name in enumerate(groups)}
        return UnifiedOffloadOptimizer(groups, modules, MemoryBudget(100, budget, 0, activation_margin=1),
                                       ScheduleConfig(tuple((key,) for key in groups), 1, adjacent_prefetch=0,
                                                      cross_layer_prefetch=False, serialized_recompute=True,
                                                      transfer_factor=1, recompute_factor=1),
                                       module_groups={key: (key,) for key in groups}, nested_groups={},
                                       max_exact_candidates=max_exact)

    def test_instances_of_same_kind_choose_distinct_actions(self):
        plan = self.solver().solve()
        self.assertEqual(plan.decisions, {'layer0.dense_mlp': 'RECOMPUTE', 'layer1.dense_mlp': 'KEEP'})
        self.assertTrue(plan.search_complete)
        self.assertLessEqual(plan.predicted_peak_mb, plan.memory_limit_mb)

    def test_serial_replay_workspace_is_not_summed_across_layers(self):
        solver = self.solver()
        plan = solver.evaluate(dict.fromkeys(solver.groups, 'RECOMPUTE'))
        self.assertEqual(plan.predicted_peak_mb, 130)

    def test_recompute_baseline_does_not_double_charge_replay_workspace(self):
        solver = self.solver()
        solver.schedule = replace(solver.schedule, baseline_recompute=True)
        plan = solver.evaluate(dict.fromkeys(solver.groups, 'RECOMPUTE'))
        self.assertEqual(plan.predicted_peak_mb, 100)
        plan = solver.evaluate({'layer0.dense_mlp': 'OFFLOAD', 'layer1.dense_mlp': 'RECOMPUTE'})
        self.assertEqual(plan.predicted_peak_mb, 200)

    def test_bounded_d2h_charges_actual_serial_transfer_cost(self):
        solver = self.solver()
        solver.schedule = replace(solver.schedule, synchronous_d2h=True)
        plan = solver.evaluate(dict.fromkeys(solver.groups, 'OFFLOAD'))
        self.assertEqual(plan.offload_tail_ms, 20)

    def test_reference_validation_requires_two_distinct_post_profile_steps(self):
        profiler = SimpleNamespace(_current_iter=5, _memory_telemetry={'baseline_peak_bytes': 100},
                                   _iteration_peak_bytes=100, _pp_rank=0, is_profiling_done=lambda: True)
        with patch.dict(os.environ, {'AUTO_ACTIVATION_MEMORY': '1'}):
            automatic.advance_reference_validation(profiler)
            self.assertFalse(getattr(profiler, '_auto_baseline_validated', False))
            profiler._current_iter = 6
            profiler._iteration_peak_bytes = 80
            automatic.advance_reference_validation(profiler)
            automatic.advance_reference_validation(profiler)
            self.assertEqual(len(profiler._auto_reference_validation['samples']), 1)
            profiler._current_iter = 7
            profiler._iteration_peak_bytes = 90
            automatic.advance_reference_validation(profiler)
            self.assertTrue(profiler._auto_baseline_validated)
            self.assertEqual(profiler._memory_telemetry['baseline_peak_bytes'], 90)

    def test_bounded_search_discloses_incomplete_global_search(self):
        plan = self.solver(count=16, budget=500, max_exact=10).solve()
        self.assertEqual(len(plan.decisions), 16)
        self.assertFalse(plan.search_complete)
        self.assertLess(plan.candidates, 10000)
        self.assertLessEqual(plan.predicted_peak_mb, 500)

    def test_one_switch_owns_legacy_optimization_configuration(self):
        args = SimpleNamespace(auto_activation_memory=True, offload_modules=['expert_fc1'])
        with patch.dict(os.environ, {'MEGATRON_LAST_LAYER_NO_OFFLOAD': '1'}, clear=True):
            automatic.configure(args)
            self.assertTrue(args.fine_grained_activation_offloading)
            self.assertEqual(args.offload_modules, [])
            self.assertEqual(os.environ['MEGATRON_ADAPTIVE_OFFLOAD'], '1')
            self.assertEqual(os.environ['MEGATRON_LAST_LAYER_NO_OFFLOAD'], '0')
            self.assertEqual(os.environ['PYTORCH_NPU_ALLOC_CONF'], 'expandable_segments:True')

    def test_manual_checkpoint_and_optimizer_swap_are_not_silently_mixed(self):
        for field, value in (('recompute_granularity', 'full'), ('recompute_activation_function', True),
                             ('swap_optimizer', True), ('optimizer_cpu_offload', True)):
            with self.subTest(field=field), patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    automatic.configure(SimpleNamespace(auto_activation_memory=True, **{field: value}))

    def test_disabled_configuration_is_untouched(self):
        args = SimpleNamespace(offload_modules=['expert_fc1'])
        with patch.dict(os.environ, {}, clear=True):
            automatic.configure(args)
            self.assertEqual(vars(args), {'offload_modules': ['expert_fc1']})

    def test_tensor_argument_layout_does_not_capture_tensor_in_template(self):
        tensor = torch.randn(2, requires_grad=True)
        values, positions, template, structure = automatic._tensor_layout((tensor,), {'nested': (tensor, None), 'scale': 2})
        self.assertEqual(len(values), 2)
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in template))
        args, kwargs = automatic._restore(values, positions, template, structure)
        self.assertIs(args[0], tensor)
        self.assertIs(kwargs['nested'][0], tensor)

    def test_all_actions_preserve_outputs_and_all_gradients(self):
        for action in ('KEEP', 'OFFLOAD', 'RECOMPUTE'):
            with self.subTest(action=action), ExitStack() as stack:
                torch.manual_seed(7)
                reference = ToyModule()
                candidate = copy.deepcopy(reference)
                original_input = torch.randn(4, 8, requires_grad=True)
                original_scale = torch.randn(4, 1, requires_grad=True)
                actual_input = original_input.detach().clone().requires_grad_(True)
                actual_scale = original_scale.detach().clone().requires_grad_(True)
                expected = reference(original_input, scale=original_scale)
                (expected['output'] + expected['bias']).square().sum().backward()
                recorder = Recorder()
                profiler = SimpleNamespace(_auto_module_specs={}, _plan=SimpleNamespace(decisions={'layer1.dense_mlp': action}),
                                           is_optimization_applied=lambda: True, is_profiling_active=lambda: False)
                stack.enter_context(patch.object(runtime.PipelineOffloadManager, 'get_instance', return_value=recorder))
                stack.enter_context(patch.object(runtime, 'fine_grained_offloading_group_start', side_effect=lambda tensor, **kwargs: tensor))
                stack.enter_context(patch.object(runtime, 'fine_grained_offloading_group_commit', side_effect=lambda *tensors, **kwargs: tensors))
                stack.enter_context(patch.object(runtime, 'get_fine_grained_offloading_context', return_value=recorder))
                stack.enter_context(patch('megatron.core.tensor_parallel.checkpoint', side_effect=lambda function, distribute, *inputs: checkpoint(function, *inputs, use_reentrant=True)))
                automatic.wrap_module(candidate, 1, 'dense_mlp', 0, False, profiler)
                actual = candidate(actual_input, scale=actual_scale)
                (actual['output'] + actual['bias']).square().sum().backward()
                self.assertEqual(actual['metadata'], 7)
                torch.testing.assert_close(actual['output'], expected['output'], rtol=0, atol=0)
                torch.testing.assert_close(actual_input.grad, original_input.grad, rtol=0, atol=0)
                torch.testing.assert_close(actual_scale.grad, original_scale.grad, rtol=0, atol=0)
                for actual_parameter, expected_parameter in zip(candidate.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)
                self.assertFalse(recorder.saved)

    def test_no_grad_passthrough_and_changed_shape_rejection(self):
        module = ToyModule()
        profiler = SimpleNamespace(_auto_module_specs={}, _plan=SimpleNamespace(decisions={'layer1.dense_mlp': 'KEEP'}),
                                   is_optimization_applied=lambda: True, is_profiling_active=lambda: False)
        automatic.wrap_module(module, 1, 'dense_mlp', 0, False, profiler)
        with torch.no_grad():
            module(torch.randn(5, 8), scale=torch.ones(5, 1))
        module(torch.randn(4, 8), scale=torch.ones(4, 1))
        with self.assertRaisesRegex(RuntimeError, 'signature changed'):
            module(torch.randn(5, 8), scale=torch.ones(5, 1))

    def test_preflight_rejects_before_optimizer_allocation(self):
        factory = Mock()
        config = SimpleNamespace(optimizer='adam', use_distributed_optimizer=False)
        with patch.object(torch.cuda, 'mem_get_info', return_value=(1024 ** 3, 1024 ** 3)), \
             patch.object(torch.cuda, 'memory_reserved', return_value=0), \
             patch.object(torch.cuda, 'memory_allocated', return_value=1), \
             patch.dict(os.environ, {'AUTO_ACTIVATION_MEMORY': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'resident parameters'):
                automatic.optimizer_capacity_preflight_wrapper(factory)(config, [ToyModule()])
        factory.assert_not_called()
