import gc
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mindspeed.core.pipeline_parallel.adaptive_offload import adaptive_memory_profiler as profiling
from mindspeed.core.pipeline_parallel.adaptive_offload import fine_grained_activation_offload as runtime
from mindspeed.core.pipeline_parallel.adaptive_offload.auto_activation_memory import observe_saved_activations
from mindspeed.core.pipeline_parallel.adaptive_offload.saved_activation_identity import SavedActivationIdentity
from mindspeed.core.pipeline_parallel.adaptive_offload.unified_offload_optimizer import GroupProfile, MemoryBudget, ScheduleConfig, UnifiedOffloadOptimizer


class TestStorageMemoryModel(unittest.TestCase):
    def optimizer(self, groups, **schedule):
        return UnifiedOffloadOptimizer(groups, {}, MemoryBudget(1000, 10000, 0, activation_margin=1.0),
            ScheduleConfig((tuple(groups),), 2, baseline_recompute=True, unified_transport=True, **schedule),
            module_groups={name: (name,) for name in groups}, nested_groups={})

    def test_distinct_views_share_keep_storage_not_transfer_identity(self):
        tensor = torch.arange(32).reshape(8, 4)
        view = tensor.transpose(0, 1)
        ledger = SavedActivationIdentity()
        ledger.record(0, tensor)
        ledger.record(1, view)
        self.assertEqual(ledger.aliases(), {})
        self.assertEqual(sum(ledger.storage_footprint(2).values()), tensor.untyped_storage().nbytes())
        self.assertEqual(ledger.contiguous_copy_bytes(2), view.numel() * view.element_size())

    def test_same_object_materialization_is_counted_once(self):
        tensor = torch.arange(32).reshape(8, 4).transpose(0, 1)
        ledger = SavedActivationIdentity()
        ledger.record(0, tensor)
        ledger.record(1, tensor)
        self.assertEqual(ledger.aliases(), {1: 0})
        self.assertEqual(ledger.contiguous_copy_bytes(2), tensor.numel() * tensor.element_size())

    def test_cloned_equal_values_are_distinct_storage(self):
        tensor = torch.arange(32)
        other = tensor.clone()
        ledger = SavedActivationIdentity()
        ledger.record(0, tensor)
        ledger.record(1, other)
        self.assertEqual(sum(ledger.storage_footprint(2).values()), 2 * tensor.untyped_storage().nbytes())

    def test_incomplete_witnesses_disable_storage_savings(self):
        tensor = torch.arange(32)
        ledger = SavedActivationIdentity()
        ledger.record(0, tensor)
        self.assertIsNone(ledger.storage_footprint(2))
        del tensor
        gc.collect()
        self.assertIsNone(ledger.storage_footprint(1))
        self.assertIsNone(ledger.contiguous_copy_bytes(1))

    def test_storage_query_failure_is_explicitly_unavailable(self):
        tensor = torch.arange(32)
        ledger = SavedActivationIdentity()
        ledger.record(0, tensor)
        with patch.object(torch.Tensor, "untyped_storage", side_effect=RuntimeError("unavailable")):
            self.assertIsNone(ledger.storage_footprint(1))

    def test_keep_uses_storage_but_reload_uses_copy_bytes(self):
        group = GroupProfile(100, 1, 1, 1, 1, keep_mb=20)
        optimizer = self.optimizer({"module": group}, adjacent_prefetch=1)
        self.assertEqual(optimizer.evaluate({"module": "KEEP"}).predicted_peak_mb, 1040)
        self.assertEqual(optimizer.evaluate({"module": "OFFLOAD"}).predicted_peak_mb, 1300)
        fallback = self.optimizer({"module": replace(group, keep_mb=None)}, adjacent_prefetch=1)
        self.assertEqual(fallback.evaluate({"module": "KEEP"}).predicted_peak_mb, 1200)

    def test_large_backing_storage_can_increase_keep_estimate(self):
        group = GroupProfile(100, 1, 1, 1, 1, keep_mb=140)
        optimizer = self.optimizer({"module": group}, immediate_backward_keep=True, adjacent_prefetch=0)
        self.assertEqual(optimizer.evaluate({"module": "KEEP"}).predicted_peak_mb, 1280)
        self.assertEqual(optimizer.evaluate({"module": "OFFLOAD"}).predicted_peak_mb, 1340)

    def test_byte_credits_use_worst_source_ratio_not_largest_group_source(self):
        groups = {"large": GroupProfile(100, 1, 1, 1, 1, keep_mb=20),
                  "small": GroupProfile(10, 1, 1, 1, 1, keep_mb=5, d2h_storage_ratio=3)}
        optimizer = self.optimizer(groups, adjacent_prefetch=1, d2h_slots=2)
        self.assertEqual(optimizer.evaluate({name: "OFFLOAD" for name in groups}).predicted_peak_mb, 1800)

    def test_invalid_storage_measurements_are_rejected(self):
        group = GroupProfile(100, 1, 1, 1, 1)
        for invalid in (-1, float("nan"), float("inf")):
            with self.subTest(keep_mb=invalid), self.assertRaises(ValueError):
                self.optimizer({"module": replace(group, keep_mb=invalid)})
        for invalid in (0.5, float("nan"), float("inf")):
            with self.subTest(ratio=invalid), self.assertRaises(ValueError):
                self.optimizer({"module": replace(group, d2h_storage_ratio=invalid)})

    def observation(self, checker):
        stats = profiling.OffloadGroupStats()
        profiler = SimpleNamespace(record_group_offload_bytes=Mock(), _get_or_create_offload_group=lambda key: stats)
        manager = SimpleNamespace(cur_forward_chunk=lambda: SimpleNamespace(tensor_need_offloading_checker=checker))
        return stats, profiler, manager

    def test_filtered_small_view_keeps_its_full_backing_without_duplicate_charge(self):
        stats, profiler, manager = self.observation(lambda tensor: tensor.numel() > 1)
        tensor = torch.arange(32, dtype=torch.float32).reshape(8, 4).requires_grad_()
        with patch.object(runtime.PipelineOffloadManager, "get_instance", return_value=manager):
            with observe_saved_activations(torch.nn.Identity(), profiler, "module"):
                result = tensor.square().sum() + tensor[:1, :1].square().sum()
        self.assertEqual(stats.non_offloadable_bytes, 4)
        self.assertEqual(stats.resident_storage_bytes, 128)
        self.assertEqual(stats.keep_storage_bytes, 0)
        self.assertEqual(stats.storage_footprint_sample_count, 1)
        self.assertFalse(stats.storage_footprint_incomplete)
        profiler.record_group_offload_bytes.assert_called_once_with("module", 128, logical_bytes=128)
        result.backward()

    def test_slice_source_and_contiguous_temporary_are_both_budgeted(self):
        stats, profiler, manager = self.observation(lambda tensor: True)
        tensor = torch.arange(32, dtype=torch.float32).reshape(8, 4).requires_grad_()
        with patch.object(runtime.PipelineOffloadManager, "get_instance", return_value=manager):
            with observe_saved_activations(torch.nn.Identity(), profiler, "module"):
                result = tensor[:, :1].square().sum()
        self.assertEqual(stats.keep_storage_bytes, 128)
        self.assertEqual(stats.d2h_storage_ratio, 5)
        profiler.record_group_offload_bytes.assert_called_once_with("module", 32, logical_bytes=32)
        result.backward()

    def test_solver_only_uses_complete_repeated_storage_samples(self):
        for sample_count, incomplete, expected in ((1, False, None), (2, False, 20.0), (4, True, None)):
            with self.subTest(samples=sample_count, incomplete=incomplete):
                profiler = profiling.AdaptiveMemoryProfiler()
                profiler._auto_module_specs = {"layer1.attention": {"layer": 1, "order": 2}}
                profiler._offload_group_stats = {"layer1.attention": profiling.OffloadGroupStats(
                    total_offload_bytes=100 * 2**20, forward_compute_time_ms=1, backward_compute_time_ms=1,
                    d2h_time_ms=1, h2d_time_ms=1, d2h_sample_count=2, h2d_sample_count=2,
                    keep_storage_bytes=20 * 2**20, resident_storage_bytes=8 * 2**20,
                    non_offloadable_bytes=4 * 2**20, storage_footprint_sample_count=sample_count,
                    storage_footprint_incomplete=incomplete, d2h_storage_ratio=3)}
                profiler._memory_telemetry.update(baseline_peak_bytes=100 * 2**20, capacity_bytes=10 * 2**30,
                    activation_baseline_peak_bytes=100 * 2**20, max_inflight_microbatches=1, samples=4)
                captured = []

                def build(groups, *args, **kwargs):
                    captured.append(groups["layer1.attention"])
                    return UnifiedOffloadOptimizer(groups, *args, **kwargs)

                with patch.object(profiling, "UnifiedOffloadOptimizer", side_effect=build):
                    profiler._solve_auto_modules()
                self.assertTrue(captured)
                self.assertTrue(all(group.keep_mb == expected and group.resident_mb == 8 and group.d2h_storage_ratio == 3 for group in captured))


if __name__ == "__main__":
    unittest.main()
