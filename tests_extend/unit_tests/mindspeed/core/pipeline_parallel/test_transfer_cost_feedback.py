import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mindspeed.core.pipeline_parallel.adaptive_offload.adaptive_memory_profiler import AdaptiveMemoryProfiler
from mindspeed.core.pipeline_parallel.adaptive_offload.transfer_cost_feedback import TransferCostFeedback
from mindspeed.core.pipeline_parallel.adaptive_offload.unified_offload_optimizer import GroupProfile


class FakeModel:
    def __init__(self, groups=None):
        self.groups = groups or {'module': GroupProfile(1, 10, 10, 1, 1)}
        self.memory = SimpleNamespace(limit_mb=10)

    def evaluate(self, decisions):
        action = decisions['module']
        cost = {'KEEP': 0, 'RECOMPUTE': 5, 'OFFLOAD': self.groups['module'].d2h_ms + self.groups['module'].h2d_ms}[action]
        return SimpleNamespace(decisions=dict(decisions), predicted_overhead_ms=cost,
                               predicted_peak_mb=20 if action == 'KEEP' else 5)


class TransferFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.plan = self.model.evaluate({'module': 'OFFLOAD'})
        self.stats = {'module': SimpleNamespace(d2h_sample_count=0, h2d_sample_count=0,
                                               d2h_time_ms=0.0, h2d_time_ms=0.0)}
        self.feedback = TransferCostFeedback([self.model], self.model, self.plan, self.stats, 0, 1.0)
        self.patcher = patch.object(TransferCostFeedback, '_repriced_model', side_effect=lambda model, groups: FakeModel(groups))
        self.reprice = self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def sample(self, count, duration=10.0):
        self.stats['module'].d2h_sample_count = count
        self.stats['module'].h2d_sample_count = count
        self.stats['module'].d2h_time_ms = duration
        self.stats['module'].h2d_time_ms = duration

    def two_windows(self, remaining=32):
        self.sample(8)
        self.assertFalse(self.feedback.ready(self.stats, 1, remaining))
        self.sample(16)
        return self.feedback.ready(self.stats, 9, remaining)

    def test_two_independent_windows_required(self):
        self.assertTrue(self.two_windows())
        self.assertEqual(self.feedback.last_probe['window_count'], 2)

    def test_same_counts_do_not_probe(self):
        self.sample(8)
        self.assertFalse(self.feedback.ready(self.stats, 1, 32))
        self.assertFalse(self.feedback.ready(self.stats, 9, 32))
        self.reprice.assert_not_called()

    def test_same_iteration_does_not_count_as_two_windows(self):
        self.sample(8)
        self.feedback.ready(self.stats, 1, 32)
        self.sample(16)
        self.assertFalse(self.feedback.ready(self.stats, 1, 32))
        self.reprice.assert_not_called()

    def test_infeasible_keep_is_not_a_witness(self):
        self.assertTrue(self.two_windows())
        self.assertEqual(self.feedback.last_probe['witness_decisions'], {'module': 'RECOMPUTE'})

    def test_no_plan_or_profile_mutation(self):
        self.two_windows()
        self.assertEqual(self.plan.decisions, {'module': 'OFFLOAD'})
        self.assertEqual(self.model.groups['module'].d2h_ms, 1)
        self.assertEqual(self.stats['module'].d2h_sample_count, 16)

    def test_request_is_one_shot_until_new_plan(self):
        self.assertTrue(self.two_windows())
        self.sample(24)
        self.assertFalse(self.feedback.ready(self.stats, 17, 32))

    def test_replan_cost_must_amortize(self):
        self.feedback.solve_time_ms = 1000
        self.assertFalse(self.two_windows())
        self.assertEqual(self.feedback.last_probe['reason'], 'insufficient_amortized_gain')

    def test_near_training_end_does_not_amortize(self):
        self.feedback.solve_time_ms = 20
        self.assertFalse(self.two_windows(remaining=1))

    def test_no_remaining_iterations(self):
        self.sample(8)
        self.assertFalse(self.feedback.ready(self.stats, 1, 0))
        self.reprice.assert_not_called()

    def test_invalid_transfer_time_rejected(self):
        for duration in (-1.0, math.nan, math.inf):
            self.sample(8, duration)
            self.assertFalse(self.feedback.ready(self.stats, 1, 32))
        self.reprice.assert_not_called()

    def test_missing_or_partial_measurement_rejected(self):
        self.assertFalse(self.feedback.ready({}, 1, 32))
        self.sample(8)
        self.stats['module'].h2d_sample_count = 0
        self.assertFalse(self.feedback.ready(self.stats, 1, 32))
        self.reprice.assert_not_called()

    def test_counter_regression_rejected(self):
        self.sample(8)
        feedback = TransferCostFeedback([self.model], self.model, self.plan, self.stats, 0, 1)
        self.sample(7)
        self.assertFalse(feedback.ready(self.stats, 1, 32))

    def test_unchanged_exposed_cost_does_not_search(self):
        self.sample(8, 1)
        self.feedback.ready(self.stats, 1, 32)
        self.sample(16, 1)
        self.assertFalse(self.feedback.ready(self.stats, 9, 32))
        self.assertEqual(self.feedback.last_probe['reason'], 'exposed_cost_stable')
        self.assertEqual(self.reprice.call_count, 1)

    def test_counter_regression_after_first_window(self):
        self.sample(8)
        self.feedback.ready(self.stats, 1, 32)
        self.sample(4)
        self.assertFalse(self.feedback.ready(self.stats, 9, 32))
        self.assertEqual(self.feedback.window_count, 1)
        self.reprice.assert_not_called()

    def test_iteration_regression_does_not_create_a_window(self):
        self.sample(8)
        self.feedback.ready(self.stats, 9, 32)
        self.sample(16)
        self.assertFalse(self.feedback.ready(self.stats, 8, 32))
        self.assertEqual(self.feedback.window_count, 1)
        self.reprice.assert_not_called()

    def test_no_offload_has_no_probe(self):
        plan = self.model.evaluate({'module': 'RECOMPUTE'})
        feedback = TransferCostFeedback([self.model], self.model, plan, self.stats, 0, 1)
        self.sample(8)
        self.assertFalse(feedback.ready(self.stats, 1, 32))
        self.reprice.assert_not_called()

    def test_invalid_solve_cost_rejected(self):
        for cost in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                TransferCostFeedback([self.model], self.model, self.plan, self.stats, 0, cost)


class ProfilerTransferFeedbackTests(unittest.TestCase):
    def make_profiler(self):
        profiler = AdaptiveMemoryProfiler.__new__(AdaptiveMemoryProfiler)
        profiler._profiling_done = profiler._optimization_applied = True
        profiler._plan = SimpleNamespace(decisions={'module': 'OFFLOAD'})
        profiler._auto_transfer_feedback = SimpleNamespace(ready=Mock(return_value=True))
        profiler._reoptimize_interval = 64
        profiler._last_profile_iter = 0
        profiler._auto_training_end_iteration = 80
        profiler._offload_group_stats = {}
        profiler._memory_pressure = False
        profiler._memory_guard_interval = 1
        profiler._host_cost_feedback_keys = set()
        profiler._transport_host_costs = {}
        profiler._host_cost_feedback_requests = 0
        profiler._transfer_cost_feedback_requests = 0
        return profiler

    def test_horizon_clamps_to_refresh(self):
        profiler = self.make_profiler()
        with patch.dict(os.environ, AUTO_ACTIVATION_MEMORY='1'):
            self.assertTrue(profiler._transfer_cost_feedback_ready(60))
        profiler._auto_transfer_feedback.ready.assert_called_once_with({}, 60, 4)

    def test_horizon_clamps_to_training_end(self):
        profiler = self.make_profiler()
        profiler._auto_training_end_iteration = 12
        with patch.dict(os.environ, AUTO_ACTIVATION_MEMORY='1'):
            profiler._transfer_cost_feedback_ready(9)
        profiler._auto_transfer_feedback.ready.assert_called_once_with({}, 9, 3)

    def test_feedback_uses_existing_replan_state_transition(self):
        profiler = self.make_profiler()
        with patch.dict(os.environ, AUTO_ACTIVATION_MEMORY='1'), patch('torch.distributed.is_initialized', return_value=False):
            profiler.synchronize_memory_guard(9)
        self.assertFalse(profiler._optimization_applied)
        self.assertTrue(profiler._profiling_done)
        self.assertEqual(profiler._transfer_cost_feedback_requests, 1)
        self.assertEqual(profiler._last_profile_iter, 0)

    def test_memory_guard_keeps_priority(self):
        profiler = self.make_profiler()
        profiler._memory_pressure = True
        with patch.dict(os.environ, AUTO_ACTIVATION_MEMORY='1'), patch('torch.distributed.is_initialized', return_value=False):
            profiler.synchronize_memory_guard(9)
        profiler._auto_transfer_feedback.ready.assert_not_called()
        self.assertFalse(profiler._optimization_applied)
        self.assertEqual(profiler._transfer_cost_feedback_requests, 0)

    def test_host_feedback_does_not_duplicate_transfer_probe(self):
        profiler = self.make_profiler()
        profiler._transport_host_costs = {'new': {'sample_count': 2}}
        with patch.dict(os.environ, AUTO_ACTIVATION_MEMORY='1'), patch('torch.distributed.is_initialized', return_value=False):
            profiler.synchronize_memory_guard(9)
        profiler._auto_transfer_feedback.ready.assert_not_called()
        self.assertEqual(profiler._host_cost_feedback_requests, 1)

    def test_disabled_auto_does_not_probe(self):
        profiler = self.make_profiler()
        with patch.dict(os.environ, AUTO_ACTIVATION_MEMORY='0'):
            self.assertFalse(profiler._transfer_cost_feedback_ready(9))
        profiler._auto_transfer_feedback.ready.assert_not_called()
