import copy
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as functional

from megatron.core.tensor_parallel import random as tensor_random
from megatron.core.transformer.mlp import MLP
from mindspeed.core.pipeline_parallel.adaptive_offload.dense_activation_boundary import (
    EXECUTOR, ChildBoundaryObservation, make_dense_activation_boundary,
    merge_child_measurements, solver_child_profiles, validate_child_profiles, needs_child_profile, _owned_output,
)
from mindspeed.core.pipeline_parallel.adaptive_offload.dense_mlp_wrapper import (
    _storage_key, dense_mlp_forward_wrapper,
)


PARENT = 'layer1.dense_mlp'
CHILD = PARENT + '.activation'
SPECS = {PARENT: {'checkpoint_children': [CHILD]}}


class TupleLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=False, dtype=torch.float64)
        self.bias = torch.nn.Parameter(torch.randn(output_size, dtype=torch.float64)) if bias else None
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        return self.linear(hidden), self.bias


def create_mlp(activation=functional.silu, gated=True, bias=False):
    model = MLP.__new__(MLP)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(activation_func=activation, gated_linear_unit=gated,
                                   bias_activation_fusion=False, add_bias_linear=bias,
                                   activation_func_fp8_input_store=False, fp8=None)
    model.activation_func = activation
    model.linear_fc1 = TupleLinear(8, 24 if gated else 12, bias)
    model.linear_fc2 = TupleLinear(12, 8)
    return model


def profile(sample_count=2):
    return {'parent_group': PARENT, 'executor': EXECUTOR,
            'layout': [[2, 12], 'torch.float64', [12, 1], 192],
            'sample_count': sample_count, 'retained_bytes': 512,
            'workspace_bytes': 384, 'output_bytes': 192, 'recompute_ms': 0.4}


class TestDenseActivationBoundary(unittest.TestCase):
    def test_native_core_facade_preserves_forward_and_gradients(self):
        for activation, gated, bias in ((functional.silu, True, False),
                                        (functional.silu, True, True),
                                        (functional.gelu, False, True),
                                        (functional.gelu, True, False)):
            with self.subTest(activation=activation, gated=gated, bias=bias):
                model = create_mlp(activation, gated, bias)
                reference = copy.deepcopy(model)
                hidden = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
                expected_hidden = hidden.detach().clone().requires_grad_()
                result = make_dense_activation_boundary(model).forward(hidden)
                expected = reference(expected_hidden)
                self.assertTrue(torch.equal(result[0], expected[0]))
                self.assertIs(result[1], expected[1])
                result[0].square().sum().backward()
                expected[0].square().sum().backward()
                self.assertTrue(torch.equal(hidden.grad, expected_hidden.grad))
                for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
                    self.assertTrue(torch.equal(actual_parameter.grad, expected_parameter.grad))

    def test_four_live_checkpoints_do_not_replay_linears_or_change_rng(self):
        model = create_mlp(bias=True)
        reference = copy.deepcopy(model)
        inputs = [torch.randn(2, 8, dtype=torch.float64, requires_grad=True) for _ in range(4)]
        reference_inputs = [hidden.detach().clone().requires_grad_() for hidden in inputs]
        expected = [reference(hidden)[0] for hidden in reference_inputs]
        for output in expected:
            output.square().sum().backward()
        boundary = make_dense_activation_boundary(model)
        rng = torch.get_rng_state().clone()
        with patch.object(tensor_random, '_get_all_rng_states', side_effect=lambda: (torch.get_rng_state(),)), \
                patch.object(tensor_random, '_set_all_rng_states', side_effect=torch.set_rng_state), \
                patch.object(tensor_random, '_fork_rng', side_effect=lambda: torch.random.fork_rng(devices=[])):
            results = [boundary.forward(hidden, selective=True)[0] for hidden in inputs]
            self.assertTrue(all(torch.equal(actual, expected_output) for actual, expected_output in zip(results, expected)))
            for output in results:
                output.square().sum().backward()
        self.assertEqual(model.linear_fc1.calls, 4)
        self.assertEqual(model.linear_fc2.calls, 4)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(hidden.grad, expected_hidden.grad)
                            for hidden, expected_hidden in zip(inputs, reference_inputs)))
        self.assertTrue(all(torch.equal(actual.grad, expected_parameter.grad)
                            for actual, expected_parameter in zip(model.parameters(), reference.parameters())))

    def test_parameter_aliases_and_custom_forward_are_not_silently_supported(self):
        model = create_mlp()
        model.forward = lambda hidden: hidden
        self.assertIsNone(make_dense_activation_boundary(model))
        model = create_mlp()
        model.config.activation_func_fp8_input_store = True
        self.assertIsNone(make_dense_activation_boundary(model))

    def test_known_dense_offload_wrapper_is_unwrapped_without_global_mutation(self):
        model = create_mlp()
        model.forward = types.MethodType(dense_mlp_forward_wrapper(MLP.forward), model)
        original = model.forward
        self.assertIsNotNone(make_dense_activation_boundary(model))
        self.assertIs(model.forward, original)

    def test_discard_alias_is_rejected_before_storage_resize(self):
        model = create_mlp(gated=False)
        boundary = make_dense_activation_boundary(model)
        boundary.activation = lambda projected, bias, scale: projected
        hidden = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        saved = hidden.detach().clone()
        with patch.object(tensor_random, '_get_all_rng_states', return_value=(torch.get_rng_state(),)):
            with self.assertRaisesRegex(RuntimeError, 'aliased or non-owned'):
                boundary.forward(hidden, selective=True)
        self.assertTrue(torch.equal(hidden, saved))

    def test_low_free_memory_skips_copy_probe_without_allocating(self):
        observation = ChildBoundaryObservation(SimpleNamespace(_current_memory_limit_mb=lambda: 4096), PARENT)
        output = torch.ones(2, 12, dtype=torch.float64)
        with patch.object(torch.cuda, 'mem_get_info', return_value=(0, 8 * 2**30)), \
                patch.object(torch.cuda, 'current_stream'), \
                patch.object(torch.cuda, 'memory_stats', return_value={}), \
                patch.object(torch.cuda, 'memory_allocated', return_value=1024), \
                patch.object(torch, 'empty_like', side_effect=AssertionError('unsafe probe allocation')) as allocation:
            observation.probe_copy(output, ())
        self.assertEqual(allocation.call_count, 0)
        self.assertIsNone(observation.complete())

    def test_allocator_limit_also_blocks_probe(self):
        observation = ChildBoundaryObservation(SimpleNamespace(_current_memory_limit_mb=lambda: 0), PARENT)
        with patch.object(torch.cuda, 'mem_get_info', return_value=(8 * 2**30, 8 * 2**30)), \
                patch.object(torch.cuda, 'current_stream'), \
                patch.object(torch.cuda, 'memory_stats', return_value={}), \
                patch.object(torch.cuda, 'memory_allocated', return_value=1024), \
                patch.object(torch, 'empty_like', side_effect=AssertionError('unsafe probe allocation')) as allocation:
            observation.probe_copy(torch.ones(2, 12), ())
        self.assertEqual(allocation.call_count, 0)

    def _run_cached_copy_probe(self, stats, free=0, limit_mb=4096):
        observation = ChildBoundaryObservation(SimpleNamespace(_current_memory_limit_mb=lambda: limit_mb, _reoptimize_count=1), PARENT)
        current = stats.get('allocated_bytes.all.current', 0)
        before = dict(stats, **{'allocated_bytes.all.allocated': current,
                               'allocated_bytes.all.freed': 0})
        after = dict(before, **{'allocated_bytes.all.allocated': current + 96,
                               'allocated_bytes.all.current': current + 96})
        with patch.object(torch.cuda, 'mem_get_info', return_value=(free, 8 * 2**30)), \
                patch.object(torch.cuda, 'current_stream'), \
                patch.object(torch.cuda, 'memory_stats', side_effect=[stats, before, after]), \
                patch.object(torch.cuda, 'memory_allocated', return_value=512 * 2**20), \
                patch.object(torch.cuda, 'Event'), \
                patch.object(torch, 'empty_like', wraps=torch.empty_like) as allocation:
            observation.probe_copy(torch.ones(2, 12), ())
        return observation, allocation.call_count

    def test_copy_probe_uses_reusable_cache(self):
        stats = {'allocated_bytes.all.current': 512 * 2**20, 'active_bytes.all.current': 512 * 2**20,
                 'reserved_bytes.all.current': 3 * 2**30, 'inactive_split_bytes.all.current': 0}
        observation, allocations = self._run_cached_copy_probe(stats)
        self.assertEqual(allocations, 1)
        self.assertIsNotNone(observation.copy_events)
        self.assertTrue(observation.copy_probe_budget['accepted'])
        self.assertEqual(observation.copy_probe_budget['reusable_bytes'], 2560 * 2**20)
        self.assertEqual(observation.copy_probe_budget['free_bytes'], 0)

    def test_copy_probe_excludes_pending_and_inactive_split(self):
        for active, inactive in ((1792 * 2**20, 0), (512 * 2**20, 1280 * 2**20)):
            with self.subTest(active=active, inactive=inactive):
                stats = {'allocated_bytes.all.current': 512 * 2**20, 'active_bytes.all.current': active,
                         'reserved_bytes.all.current': 2 * 2**30, 'inactive_split_bytes.all.current': inactive}
                observation, allocations = self._run_cached_copy_probe(stats)
                self.assertEqual(allocations, 0)
                self.assertEqual(observation.copy_probe_budget['reusable_bytes'], 256 * 2**20)

    def test_copy_probe_missing_or_invalid_stats_get_no_cache_credit(self):
        for stats in ({}, {'allocated_bytes.all.current': 512 * 2**20, 'active_bytes.all.current': 0,
                          'reserved_bytes.all.current': 4 * 2**30, 'inactive_split_bytes.all.current': 0}):
            with self.subTest(stats=stats):
                observation, allocations = self._run_cached_copy_probe(stats)
                self.assertEqual(allocations, 0)
                self.assertFalse(observation.copy_probe_budget['allocator_stats_valid'])
                self.assertEqual(observation.copy_probe_budget['reusable_bytes'], 0)

    def test_copy_probe_preserves_physical_reserve(self):
        stats = {'allocated_bytes.all.current': 512 * 2**20, 'active_bytes.all.current': 512 * 2**20,
                 'reserved_bytes.all.current': 1280 * 2**20, 'inactive_split_bytes.all.current': 0}
        observation, allocations = self._run_cached_copy_probe(stats, free=128 * 2**20)
        self.assertEqual(allocations, 0)
        self.assertEqual(observation.copy_probe_budget['reserve_bytes'], 2**30)

    def test_copy_probe_preserves_planning_footprint_limit(self):
        stats = {'allocated_bytes.all.current': 512 * 2**20, 'active_bytes.all.current': 1024 * 2**20,
                 'reserved_bytes.all.current': 3 * 2**30, 'inactive_split_bytes.all.current': 0}
        observation, allocations = self._run_cached_copy_probe(stats, limit_mb=1024)
        self.assertEqual(allocations, 0)
        self.assertEqual(observation.copy_probe_budget['footprint_bytes'], 2**30)

    def test_storage_union_excludes_activation_internals_and_released_output(self):
        observation = ChildBoundaryObservation(None, PARENT)
        hidden = torch.ones(2, 8, dtype=torch.float64)
        projected = torch.ones(2, 24, dtype=torch.float64)
        activated = torch.ones(2, 12, dtype=torch.float64)
        internal = torch.ones(2, 12, dtype=torch.float64)
        observation.record_saved(hidden)
        observation.inside_activation = True
        observation.record_saved(internal)
        observation.inside_activation = False
        observation.record_saved(activated)
        observation.input_storages[_storage_key(projected)] = projected.untyped_storage().nbytes()
        observation.input_storages[_storage_key(hidden)] = hidden.untyped_storage().nbytes()
        observation.output_key = _storage_key(activated)
        observation.output_bytes = activated.untyped_storage().nbytes()
        observation.activation_workspace_bytes = 192
        observation.copy_workspace_bytes = 192
        observation.layout = profile()['layout']
        timer = SimpleNamespace(elapsed_time=lambda end: 0.2)
        observation.activation_events = timer, None
        observation.copy_events = timer, None
        result = observation.complete()
        self.assertEqual(result['retained_bytes'], 512)
        self.assertEqual(result['workspace_bytes'], 384)
        self.assertEqual(result['recompute_ms'], 0.4)

    def test_copy_without_a_saved_fc2_input_is_not_a_profile(self):
        observation = ChildBoundaryObservation(None, PARENT)
        observation.activation_events = observation.copy_events = (None, None)
        observation.output_key = ('absent', 0)
        self.assertIsNone(observation.complete())

    def test_merge_uses_worst_cost_and_memory_not_sum_of_shared_inputs(self):
        first, second = profile(), profile(3)
        second.update(retained_bytes=768, workspace_bytes=512, recompute_ms=0.7)
        combined = merge_child_measurements([first, second])
        self.assertEqual(combined['retained_bytes'], 768)
        self.assertEqual(combined['workspace_bytes'], 512)
        self.assertEqual(combined['recompute_ms'], 0.7)
        self.assertEqual(combined['sample_count'], 5)
        self.assertEqual(merge_child_measurements([first, second], replicas=True)['sample_count'], 2)
        self.assertIsNone(merge_child_measurements([first, None], replicas=True))

    def test_layout_change_is_rejected_not_merged(self):
        first, second = profile(), profile()
        second['layout'][0][0] = 3
        with self.assertRaisesRegex(ValueError, 'changed layout'):
            merge_child_measurements([first, second])

    def test_only_measured_and_bound_children_reach_solver(self):
        self.assertEqual(solver_child_profiles({CHILD: profile(1)}, SPECS, {PARENT}), {})
        result = solver_child_profiles({CHILD: profile()}, SPECS, {PARENT})
        self.assertEqual(result[CHILD].retained_mb, 512 / 2**20)
        self.assertEqual(result[CHILD].workspace_mb, 384 / 2**20)
        with self.assertRaisesRegex(ValueError, 'unbound'):
            solver_child_profiles({CHILD: profile()}, {PARENT: {}}, {PARENT})

    def test_invalid_cached_measurements_are_rejected(self):
        for field, value in (('sample_count', True), ('sample_count', -1),
                             ('workspace_bytes', -1), ('output_bytes', 0),
                             ('recompute_ms', float('nan')), ('recompute_ms', float('inf')),
                             ('executor', 'unknown'), ('parent_group', 'unknown')):
            measurement = profile()
            measurement[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_child_profiles({CHILD: measurement}, SPECS)

    def test_child_only_deferred_events_are_drained(self):
        from mindspeed.core.pipeline_parallel.adaptive_offload.adaptive_memory_profiler import AdaptiveMemoryProfiler

        profiler = AdaptiveMemoryProfiler(num_profile_iters=100, measure_pcie=False)
        profiler._finish_memory_sample = lambda: None
        profiler._pending_child_profiles = [SimpleNamespace(parent_group=PARENT, complete=lambda: profile(1))]
        with patch.object(torch.cuda, 'synchronize') as synchronize:
            profiler.on_iteration_end(0)
        self.assertEqual(synchronize.call_count, 1)
        self.assertEqual(profiler._pending_child_profiles, [])
        self.assertEqual(profiler._auto_child_profiles[CHILD]['sample_count'], 1)

    def test_child_probe_sampling_stops_after_two_valid_samples(self):
        profiler = SimpleNamespace(_auto_child_profiles={CHILD: profile()}, _pending_child_profiles=[])
        self.assertFalse(needs_child_profile(profiler, PARENT))
        profiler._auto_child_profiles.clear()
        profiler._pending_child_profiles = [SimpleNamespace(parent_group=PARENT) for _ in range(2)]
        self.assertFalse(needs_child_profile(profiler, PARENT))
        profiler._pending_child_profiles = [SimpleNamespace(parent_group='other')]
        self.assertTrue(needs_child_profile(profiler, PARENT))

    def test_full_span_shape_view_is_accepted_without_input_alias(self):
        inputs = torch.ones(4, 2, 16)
        fresh = torch.ones(8, 8)
        output = fresh.view(4, 2, 8)
        self.assertIsNotNone(output._base)
        self.assertTrue(_owned_output(output, (inputs,)))

    def test_partial_offset_transposed_and_input_views_remain_rejected(self):
        inputs = torch.ones(4, 4)
        allocation = torch.ones(32)
        for output, arguments in ((allocation[:16], (inputs,)),
                                  (allocation[16:], (inputs,)),
                                  (torch.ones(4, 4).transpose(0, 1), (inputs,)),
                                  (inputs.view(2, 8), (inputs,))):
            self.assertFalse(_owned_output(output, arguments))

    def test_replaced_activation_callable_is_not_assumed_pure(self):
        model = create_mlp()
        model.activation_func = lambda hidden: hidden
        self.assertIsNone(make_dense_activation_boundary(model))

    def test_three_dimensional_fused_view_preserves_four_microbatch_gradients(self):
        from megatron.core.fusions import fused_bias_swiglu as fusion

        def activation(projected, *arguments):
            gate, value = projected.chunk(2, dim=-1)
            return functional.silu(gate) * value

        model = create_mlp()
        model.config.bias_activation_fusion = True
        reference = copy.deepcopy(model)
        inputs = [torch.randn(2, 2, 8, dtype=torch.float64, requires_grad=True) for _ in range(4)]
        expected_inputs = [hidden.detach().clone().requires_grad_() for hidden in inputs]
        with patch.object(fusion, 'SwiGLUFunction', SimpleNamespace(apply=activation)), \
                patch.object(tensor_random, '_get_all_rng_states', side_effect=lambda: (torch.get_rng_state(),)), \
                patch.object(tensor_random, '_set_all_rng_states', side_effect=torch.set_rng_state), \
                patch.object(tensor_random, '_fork_rng', side_effect=lambda: torch.random.fork_rng(devices=[])):
            expected = [reference(hidden)[0] for hidden in expected_inputs]
            for output in expected:
                output.square().sum().backward()
            boundary = make_dense_activation_boundary(model)
            results = [boundary.forward(hidden, selective=True)[0] for hidden in inputs]
            for output in results:
                output.square().sum().backward()
        self.assertTrue(all(torch.equal(actual, expected_output) for actual, expected_output in zip(results, expected)))
        self.assertTrue(all(torch.equal(hidden.grad, expected_hidden.grad) for hidden, expected_hidden in zip(inputs, expected_inputs)))
        self.assertTrue(all(torch.equal(actual.grad, expected_parameter.grad) for actual, expected_parameter in zip(model.parameters(), reference.parameters())))
        self.assertEqual(model.linear_fc1.calls, 4)
        self.assertEqual(model.linear_fc2.calls, 4)


if __name__ == '__main__':
    unittest.main()
