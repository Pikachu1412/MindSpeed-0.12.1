import contextlib
import copy
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mindspeed.core.pipeline_parallel.adaptive_offload import adaptive_memory_profiler as profiling
from mindspeed.core.pipeline_parallel.adaptive_offload.transport_host_profile import TransportHostMeter, update_host_costs


class TestPolicyFeedback(unittest.TestCase):
    def setUp(self):
        for active_patch in (patch.dict(os.environ, {"MEGATRON_ADAPTIVE_OFFLOAD": "1", "AUTO_ACTIVATION_MEMORY": "1",
                                                   "ADAPTIVE_MEM_BUDGET_MB": "auto", "ADAPTIVE_MEM_REOPTIMIZE_INTERVAL": "64"}),
                             patch.object(torch.cuda, "is_available", return_value=False),
                             patch.object(torch.distributed, "is_initialized", return_value=False)):
            active_patch.start()
            self.addCleanup(active_patch.stop)

    def profiler(self, initial_iteration=8):
        profiler = profiling.AdaptiveMemoryProfiler()
        profiler._is_stall_profile_rank = True
        profiler._pp_rank = 0
        profiler._profiling_done = True
        profiler._optimization_applied = True
        profiler._memory_guard_interval = 1
        profiler._memory_telemetry.update(baseline_peak_bytes=100 * 2**20, capacity_bytes=1000 * 2**20,
                                          samples=4, max_inflight_microbatches=4)
        profiler._plan = SimpleNamespace(decisions={"module": "OFFLOAD"}, execution={"transport_active": True},
                                         memory_limit_mb=950, predicted_peak_mb=500)
        profiler._run_optimization = Mock()
        profiler._start_reoptimization = Mock()
        self.apply(profiler, initial_iteration)
        return profiler

    def apply(self, profiler, iteration):
        profiler._current_iter = iteration
        with patch("mindspeed.core.pipeline_parallel.adaptive_offload.auto_activation_memory.save_profile_cache"), \
                contextlib.redirect_stdout(io.StringIO()):
            profiler.apply_optimization_results()

    def sample(self, profiler, key="asynchronous:1:1"):
        meter = TransportHostMeter(active=True)
        meter.metrics["begin"].update(calls=8, wall_ms=4.0, cpu_ms=2.0)
        meter.metrics["attach"].update(calls=2, wall_ms=2.0, cpu_ms=1.0)
        update_host_costs(profiler, key.split(":")[0], meter, modules=8, offloads=2, profile_key=key)

    def test_guard_replan_does_not_delay_full_profile_deadline(self):
        profiler = self.profiler()
        profiler._memory_pressure = True
        profiler.synchronize_memory_guard(37)
        self.assertFalse(profiler._optimization_applied)
        self.apply(profiler, 37)
        profiler.synchronize_memory_guard(72)
        profiler._start_reoptimization.assert_called_once_with(72)

    def test_repeated_guard_replans_preserve_profile_age(self):
        profiler = self.profiler()
        for iteration in (20, 40, 60):
            profiler._memory_pressure = True
            profiler.synchronize_memory_guard(iteration)
            self.apply(profiler, iteration)
        profiler.synchronize_memory_guard(72)
        profiler._start_reoptimization.assert_called_once_with(72)

    def test_fresh_profile_application_sets_new_deadline(self):
        profiler = self.profiler()
        profiler._profile_refresh_pending = True
        self.apply(profiler, 78)
        profiler.synchronize_memory_guard(141)
        profiler._start_reoptimization.assert_not_called()
        profiler.synchronize_memory_guard(142)
        profiler._start_reoptimization.assert_called_once_with(142)

    def test_first_application_anchors_cached_profile_deadline(self):
        profiler = self.profiler()
        self.assertEqual(getattr(profiler, "_last_profile_iter", -1), 8)
        profiler.synchronize_memory_guard(71)
        profiler._start_reoptimization.assert_not_called()

    def test_legacy_deadline_still_uses_last_optimization(self):
        profiler = self.profiler()
        with patch.dict(os.environ, {"AUTO_ACTIVATION_MEMORY": "0"}):
            self.apply(profiler, 37)
            profiler.synchronize_memory_guard(72)
            profiler._start_reoptimization.assert_not_called()
            profiler.synchronize_memory_guard(101)
            profiler._start_reoptimization.assert_called_once_with(101)

    def test_profile_completion_marks_fresh_data(self):
        profiler = self.profiler()
        profiler._profiling_done = False
        profiler._warmup_skip_iters = 0
        profiler._num_profile_iters = 1
        with patch.object(profiler, "_log_summary"), contextlib.redirect_stdout(io.StringIO()):
            profiler.on_iteration_end(0)
        self.assertTrue(profiler._profiling_done)
        self.assertTrue(getattr(profiler, "_profile_refresh_pending", False))

    def test_two_completed_host_samples_trigger_feedback(self):
        profiler = self.profiler()
        self.sample(profiler)
        profiler.synchronize_memory_guard(9)
        self.assertTrue(profiler._optimization_applied)
        self.sample(profiler)
        profiler.synchronize_memory_guard(17)
        self.assertFalse(profiler._optimization_applied)
        self.assertFalse(profiler._memory_pressure)
        self.assertEqual(profiler._host_cost_feedback_requests, 1)

    def test_feedback_is_bounded_once_per_measured_layout(self):
        profiler = self.profiler()
        self.sample(profiler)
        self.sample(profiler)
        profiler.synchronize_memory_guard(17)
        self.assertFalse(profiler._optimization_applied)
        self.apply(profiler, 17)
        for iteration in (25, 33, 41):
            self.sample(profiler)
            profiler.synchronize_memory_guard(iteration)
            self.assertTrue(profiler._optimization_applied)
            self.assertEqual(profiler._host_cost_feedback_requests, 1)
        profiler.synchronize_memory_guard(72)
        profiler._start_reoptimization.assert_called_once_with(72)

    def test_another_transport_layout_can_trigger_feedback(self):
        profiler = self.profiler()
        for unused in range(2):
            self.sample(profiler)
        self.apply(profiler, 17)
        for unused in range(2):
            self.sample(profiler, "synchronous:0:1")
        profiler.synchronize_memory_guard(25)
        self.assertFalse(profiler._optimization_applied)

    def test_no_offload_and_legacy_skip_host_feedback(self):
        for action, enabled in (("KEEP", "1"), ("OFFLOAD", "0")):
            with self.subTest(action=action, enabled=enabled):
                profiler = self.profiler()
                profiler._plan.decisions["module"] = action
                for unused in range(2):
                    self.sample(profiler)
                with patch.dict(os.environ, {"AUTO_ACTIVATION_MEMORY": enabled}):
                    profiler.synchronize_memory_guard(17)
                self.assertTrue(profiler._optimization_applied)

    def test_feedback_uses_existing_single_collective(self):
        profiler = self.profiler()
        self.sample(profiler)
        self.sample(profiler)
        flag = SimpleNamespace(item=Mock(return_value=1))
        with patch.object(torch.distributed, "is_available", return_value=True), \
                patch.object(torch.distributed, "is_initialized", return_value=True), \
                patch.object(torch, "tensor", return_value=flag) as tensor, \
                patch.object(torch.distributed, "all_reduce") as collective:
            profiler.synchronize_memory_guard(17)
        tensor.assert_called_once_with([1], device="cuda", dtype=torch.int32)
        collective.assert_called_once_with(flag, op=torch.distributed.ReduceOp.MAX)
        flag.item.assert_called_once()
        self.assertFalse(profiler._optimization_applied)

    def test_peer_request_still_invalidates_local_plan(self):
        profiler = self.profiler()
        flag = SimpleNamespace(item=Mock(return_value=1))
        with patch.object(torch.distributed, "is_available", return_value=True), \
                patch.object(torch.distributed, "is_initialized", return_value=True), \
                patch.object(torch, "tensor", return_value=flag), patch.object(torch.distributed, "all_reduce"):
            profiler.synchronize_memory_guard(17)
        self.assertFalse(profiler._optimization_applied)

    def test_failed_solver_does_not_consume_feedback_or_reset_profile_age(self):
        profiler = self.profiler()
        self.sample(profiler)
        self.sample(profiler)
        before = set(getattr(profiler, "_host_cost_feedback_keys", set()))
        profiler._run_optimization.side_effect = ValueError("invalid fixture plan")
        with self.assertRaisesRegex(RuntimeError, "invalid fixture plan"):
            self.apply(profiler, 17)
        self.assertEqual(getattr(profiler, "_host_cost_feedback_keys", set()), before)
        self.assertEqual(getattr(profiler, "_last_profile_iter", -1), 8)

    def test_local_feedback_is_not_repeated_after_conservative_replica_merge(self):
        profiler = self.profiler()
        self.sample(profiler)
        self.sample(profiler)
        peer = copy.deepcopy(profiler._profile_payload())
        peer["profile"]["transport_host"]["asynchronous:1:1"]["sample_count"] = 1

        def gather(output, payload):
            if payload is None or isinstance(payload, str):
                output[:] = [None, payload]
            else:
                output[:] = [payload, peer]

        with patch.object(torch.distributed, "is_available", return_value=True), \
                patch.object(torch.distributed, "is_initialized", return_value=True), \
                patch.object(torch.distributed, "get_world_size", return_value=2), \
                patch.object(torch.distributed, "all_gather_object", side_effect=gather):
            self.apply(profiler, 17)
        self.assertEqual(profiler._transport_host_costs["asynchronous:1:1"]["sample_count"], 1)
        self.sample(profiler)
        profiler.synchronize_memory_guard(25)
        self.assertTrue(profiler._optimization_applied)
        self.assertIn("asynchronous:1:1", getattr(profiler, "_host_cost_feedback_keys", set()))

    def test_completed_profile_waits_for_baseline_calibration_without_restarting(self):
        profiler = self.profiler(initial_iteration=7)
        profiler._num_profile_iters = 4
        profiler._start_reoptimization = profiling.AdaptiveMemoryProfiler._start_reoptimization.__get__(profiler)

        def memory_sample():
            profiler._memory_telemetry["samples"] += 1

        with patch.object(profiler, "_finish_memory_sample", side_effect=memory_sample), \
                patch.object(profiler, "_log_summary"), contextlib.redirect_stdout(io.StringIO()):
            for iteration in range(71, 77):
                profiler._current_iter = iteration
                profiler.synchronize_memory_guard(iteration)
                profiler.on_iteration_end(iteration)
        self.assertEqual(profiler._reoptimize_count, 1)
        self.assertTrue(profiler._profiling_done)
        self.assertTrue(profiler._profile_refresh_pending)
        self.assertFalse(profiler._optimization_applied)
        self.assertEqual(profiler._last_profile_iter, 7)
        self.apply(profiler, 77)
        self.assertEqual(profiler._last_profile_iter, 77)
        self.assertFalse(profiler._profile_refresh_pending)

    def test_cached_profile_is_applied_before_considering_periodic_refresh(self):
        profiler = self.profiler()
        profiler._optimization_applied = False
        profiler._last_profile_iter = -1
        profiler._last_optimize_iter = -1
        profiler.synchronize_memory_guard(100)
        profiler._start_reoptimization.assert_not_called()

    def test_pending_profile_does_not_bypass_collective_safety_request(self):
        profiler = self.profiler()
        profiler._optimization_applied = False
        profiler._profile_refresh_pending = True
        profiler._memory_pressure = True
        profiler.synchronize_memory_guard(100)
        self.assertFalse(profiler._memory_pressure)
        self.assertFalse(profiler._optimization_applied)
        self.assertTrue(profiler._profile_refresh_pending)
        profiler._start_reoptimization.assert_not_called()

    def test_legacy_pending_profile_keeps_original_refresh_behavior(self):
        profiler = self.profiler()
        profiler._optimization_applied = False
        with patch.dict(os.environ, {"AUTO_ACTIVATION_MEMORY": "0"}):
            profiler.synchronize_memory_guard(100)
        profiler._start_reoptimization.assert_called_once_with(100)


if __name__ == "__main__":
    unittest.main()
