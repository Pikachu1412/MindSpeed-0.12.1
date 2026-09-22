import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

from mindspeed.core.pipeline_parallel.adaptive_offload.adaptive_memory_profiler import AdaptiveMemoryProfiler, _memory_budget_from_telemetry


class TestCapacityPlanningWindow(unittest.TestCase):
    def profiler(self, samples=4):
        profiler = AdaptiveMemoryProfiler()
        profiler._warmup_skip_iters = 2
        profiler._num_profile_iters = samples
        return profiler

    def observe(self, profiler, iteration, capacity):
        profiler._current_iter = iteration
        profiler._record_capacity_observation(capacity)

    def test_warmup_initialization_capacity_is_excluded(self):
        profiler = self.profiler()
        self.observe(profiler, 1, 9000)
        self.assertNotIn("planning_capacity_bytes", profiler._memory_telemetry)
        self.observe(profiler, 2, 1000)
        self.assertEqual(profiler._memory_telemetry["planning_capacity_bytes"], 1000)
        self.assertEqual(profiler._memory_telemetry["capacity_window_iterations"], 1)

    def test_minimum_and_variation_include_every_sample_within_iteration(self):
        profiler = self.profiler()
        for capacity in (1000, 997, 999, 1002):
            self.observe(profiler, 2, capacity)
        self.assertEqual(profiler._memory_telemetry["planning_capacity_bytes"], 992)
        self.assertEqual(profiler._memory_telemetry["capacity_variation_bytes"], 5)
        self.assertEqual(profiler._memory_telemetry["capacity_window_iterations"], 1)

    def test_window_expires_old_iterations_including_large_initial_transient(self):
        profiler = self.profiler(samples=2)
        for iteration, capacity in ((2, 1000), (3, 990), (4, 988)):
            self.observe(profiler, iteration, capacity)
        self.assertEqual(profiler._memory_telemetry["planning_capacity_bytes"], 986)
        self.assertEqual(set(profiler._capacity_observations), {3, 4})

    def test_stable_capacity_has_no_extra_penalty(self):
        profiler = self.profiler()
        for iteration in range(2, 9):
            self.observe(profiler, iteration, 1000)
        self.assertEqual(profiler._memory_telemetry["planning_capacity_bytes"], 1000)
        self.assertEqual(profiler._memory_telemetry["capacity_variation_bytes"], 0)
        self.assertEqual(profiler._memory_telemetry["capacity_window_iterations"], 4)

    def test_planning_reduces_capacity_without_reducing_original_reserve(self):
        telemetry = {"baseline_peak_bytes": 100 * 2**20, "capacity_bytes": 1000 * 2**20,
                     "planning_capacity_bytes": 990 * 2**20}
        with patch.dict("os.environ", {"ADAPTIVE_MEM_RESERVE_MB": "10", "ADAPTIVE_MEM_RESERVE_FRACTION": "0.05"}):
            live = _memory_budget_from_telemetry(telemetry)
            planning = _memory_budget_from_telemetry(telemetry, planning=True)
        self.assertEqual(live.limit_mb, 950)
        self.assertEqual(planning.limit_mb, 940)
        self.assertEqual(planning.reserve_mb, live.reserve_mb)

    def test_missing_history_preserves_previous_budget(self):
        telemetry = {"baseline_peak_bytes": 100 * 2**20, "capacity_bytes": 10000 * 2**20}
        self.assertEqual(asdict(_memory_budget_from_telemetry(telemetry)), asdict(_memory_budget_from_telemetry(telemetry, planning=True)))

    def test_live_guard_does_not_use_planning_window_or_relax_plan_limit(self):
        profiler = self.profiler()
        profiler._memory_telemetry.update(baseline_peak_bytes=100 * 2**20, capacity_bytes=1000 * 2**20,
                                          planning_capacity_bytes=900 * 2**20)
        profiler._plan = SimpleNamespace(memory_limit_mb=940)
        with patch.dict("os.environ", {"ADAPTIVE_MEM_RESERVE_MB": "10", "ADAPTIVE_MEM_RESERVE_FRACTION": "0.05"}):
            self.assertEqual(profiler._current_memory_limit_mb(), 940)
            profiler._memory_telemetry["capacity_bytes"] = 980 * 2**20
            self.assertEqual(profiler._current_memory_limit_mb(), 931)

    def test_measured_native_window_rejects_previous_boundary_plan(self):
        profiler = self.profiler()
        capacities = (60573.74609375, 60573.28515625, 60573.53125, 60573.7734375)
        for iteration, capacity in enumerate(capacities, 4):
            self.observe(profiler, iteration, int(capacity * 2**20))
        profiler._memory_telemetry.update(baseline_peak_bytes=51564.853515625 * 2**20,
                                          capacity_bytes=capacities[-1] * 2**20)
        with patch.dict("os.environ", {"ADAPTIVE_MEM_RESERVE_MB": "1024", "ADAPTIVE_MEM_RESERVE_FRACTION": "0.05"}):
            previous = _memory_budget_from_telemetry(profiler._memory_telemetry)
            robust = _memory_budget_from_telemetry(profiler._memory_telemetry, planning=True)
        self.assertGreater(previous.limit_mb, 57544.80157928467)
        self.assertLess(robust.limit_mb, 57544.80157928467)
        self.assertLess(robust.limit_mb, 57544.6431640625)

    def test_large_capacity_changes_reset_window_without_disabling_guard(self):
        for before, after, predicted in ((60000, 64000, 56000), (64000, 60000, 58000)):
            with self.subTest(before=before, after=after):
                profiler = self.profiler()
                self.observe(profiler, 2, before * 2**20)
                profiler._memory_telemetry.update(capacity_bytes=before * 2**20, baseline_peak_bytes=54000 * 2**20)
                profiler._current_iter = 3
                profiler._plan = SimpleNamespace(predicted_peak_mb=predicted, memory_limit_mb=before * 0.95)
                with patch.dict("os.environ", {"AUTO_ACTIVATION_MEMORY": "1", "ADAPTIVE_MEM_RESERVE_MB": "1024",
                                               "ADAPTIVE_MEM_RESERVE_FRACTION": "0.05"}), \
                     patch("torch.cuda.is_available", return_value=True), \
                     patch("torch.cuda.mem_get_info", return_value=(after * 2**20, 65536 * 2**20)), \
                     patch("torch.cuda.memory_reserved", return_value=0):
                    profiler._sample_memory_capacity()
                    planning = _memory_budget_from_telemetry(profiler._memory_telemetry, planning=True)
                    self.assertLessEqual(profiler._current_memory_limit_mb(), after * 0.95)
                self.assertEqual(profiler._memory_telemetry["planning_capacity_bytes"], after * 2**20)
                self.assertEqual(profiler._memory_telemetry["capacity_window_iterations"], 1)
                self.assertTrue(profiler._memory_pressure)
                self.assertGreaterEqual(planning.limit_mb, 54000)

    def test_small_capacity_jitter_preserves_window(self):
        profiler = self.profiler()
        self.observe(profiler, 2, 60000 * 2**20)
        profiler._memory_telemetry.update(capacity_bytes=60000 * 2**20, baseline_peak_bytes=54000 * 2**20)
        profiler._current_iter = 3
        with patch.dict("os.environ", {"AUTO_ACTIVATION_MEMORY": "1"}), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(59999 * 2**20, 65536 * 2**20)), \
             patch("torch.cuda.memory_reserved", return_value=0):
            profiler._sample_memory_capacity()
        self.assertEqual(profiler._memory_telemetry["planning_capacity_bytes"], 59998 * 2**20)
        self.assertEqual(profiler._memory_telemetry["capacity_window_iterations"], 2)
        self.assertFalse(profiler._memory_pressure)

    def test_periodic_reprofile_discards_stale_capacity_window(self):
        profiler = self.profiler()
        self.observe(profiler, 2, 1000)
        profiler._is_stall_profile_rank = False
        with patch.dict("os.environ", {"AUTO_ACTIVATION_MEMORY": "1"}):
            profiler._start_reoptimization(72)
        self.assertFalse(profiler._capacity_observations)
        self.assertNotIn("planning_capacity_bytes", profiler._memory_telemetry)
        self.observe(profiler, 72, 1100)
        self.assertEqual(profiler._memory_telemetry["planning_capacity_bytes"], 1100)


if __name__ == "__main__":
    unittest.main()
