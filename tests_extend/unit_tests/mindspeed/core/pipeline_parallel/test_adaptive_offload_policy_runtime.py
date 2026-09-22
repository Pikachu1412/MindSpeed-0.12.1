import contextlib
import copy
import io
import itertools
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mindspeed.core.pipeline_parallel.adaptive_offload import adaptive_memory_profiler as profiling
from mindspeed.core.pipeline_parallel.adaptive_offload import fine_grained_activation_offload as runtime
from mindspeed.core.pipeline_parallel.adaptive_offload.attention_wrapper import attention_forward_wrapper
from mindspeed.core.pipeline_parallel.adaptive_offload.experts_wrapper import te_grouped_mlp_forward_wrapper
from mindspeed.core.pipeline_parallel.adaptive_offload.transformer_layer_wrapper import (
    _fgao_layer_needs_wrapper, transformer_layer_init_wrapper,
)
from mindspeed.core.pipeline_parallel.adaptive_offload.unified_offload_optimizer import StrategyPlan


class FakeEvent:
    clock = 0

    def __init__(self, timestamp=None, **kwargs):
        self.timestamp = timestamp

    def record(self, *args):
        FakeEvent.clock += 1
        self.timestamp = FakeEvent.clock

    def elapsed_time(self, other):
        return other.timestamp - self.timestamp


class TestAdaptiveOffloadPolicyRuntime(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "MEGATRON_ADAPTIVE_OFFLOAD": "1",
            "ADAPTIVE_MEM_BUDGET_MB": "auto",
            "ADAPTIVE_MEM_RESERVE_MB": "0",
            "ADAPTIVE_MEM_RESERVE_FRACTION": "0",
            "ADAPTIVE_MEM_WARMUP_SKIP_ITERS": "0",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def profiler(self):
        profiler = profiling.AdaptiveMemoryProfiler()
        profiler._is_stall_profile_rank = True
        profiler._diag_print_count = 100
        return profiler

    def seed(self, profiler):
        profiler._memory_telemetry.update(
            baseline_peak_bytes=100 * 1024 ** 2, capacity_bytes=1000 * 1024 ** 2,
            max_inflight_microbatches=2, samples=3,
        )
        profiler._offload_group_stats["attn_norm"] = profiling.OffloadGroupStats(
            group_name="attn_norm", total_offload_bytes=100 * 1024 ** 2,
            forward_compute_time_ms=2, backward_compute_time_ms=4,
            d2h_time_ms=10, h2d_time_ms=10, d2h_sample_count=3, h2d_sample_count=3,
        )
        layer = profiler._get_or_create_layer(1, False)
        layer.modules["attn_norm"] = profiling.LayerModuleStats(
            compute_time_ms=2, peak_memory_delta_bytes=10 * 1024 ** 2,
            input_bytes=5 * 1024 ** 2, sample_count=3,
        )
        profiler._num_layers_on_this_rank = 1

    def test_profile_does_not_continue_between_reoptimization_windows(self):
        profiler = self.profiler()
        profiler._profiling_done = True
        profiler._reoptimize_interval = 10
        self.assertFalse(profiler.is_stall_profiling_active())
        self.assertFalse(profiler.is_profiling_active())

    def test_single_device_profiles_without_distributed_initialization(self):
        profiler = profiling.AdaptiveMemoryProfiler()
        self.assertTrue(profiler.is_stall_profiling_active())

    def test_reprofile_keeps_measurements_for_currently_kept_groups(self):
        profiler = self.profiler()
        self.seed(profiler)
        with contextlib.redirect_stdout(io.StringIO()):
            profiler._start_reoptimization(50)
        self.assertIn("attn_norm", profiler.get_offload_group_stats())

    def test_module_peak_uses_max_not_average(self):
        stats = profiling.LayerModuleStats()
        stats.update(2, 100, 20)
        stats.update(4, 10, 40)
        self.assertEqual((stats.compute_time_ms, stats.peak_memory_delta_bytes, stats.input_bytes), (3, 100, 40))

    def test_checkpoint_replay_is_profiled_separately(self):
        profiler = self.profiler()
        context = Mock(return_value=contextlib.nullcontext())
        with patch.object(profiler, "profile_module", context):
            wrapped = profiler.profile_checkpoint_callable(lambda value: value * 2, 1, "attention")
            self.assertEqual(wrapped(3), 6)
            context.assert_not_called()
            self.assertEqual(wrapped(3), 6)
        self.assertTrue(context.call_args.kwargs["recompute"])

    def test_microbatch_lifetimes_do_not_depend_on_offload_markers(self):
        profiler = self.profiler()
        for _ in range(10):
            token = object()
            profiler.begin_microbatch(token)
            output = torch.ones(2, requires_grad=True).square()
            profiler.track_microbatch_backward(token, output)
            output.sum().backward()
        self.assertFalse(profiler._live_microbatches)
        self.assertEqual(profiler._memory_telemetry["max_inflight_microbatches"], 1)

    def test_module_profiling_defers_sync_and_preserves_iteration_peak(self):
        profiler = self.profiler()
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "Event", FakeEvent), \
             patch.object(torch.cuda, "max_memory_allocated", return_value=500), \
             patch.object(torch.cuda, "memory_allocated", return_value=100), \
             patch.object(torch.cuda, "reset_peak_memory_stats"), \
             patch.object(torch.cuda, "synchronize") as synchronize, \
             patch.object(torch.cuda, "current_stream") as stream:
            with profiler.profile_module(1, "attention", False, torch.ones(5)):
                pass
            self.assertEqual(profiler._iteration_peak_bytes, 500)
            self.assertEqual(profiler._pending_module_events[0][-1], 20)
            synchronize.assert_not_called()
            stream.assert_not_called()

    def test_nested_stalls_are_removed_from_parent_compute(self):
        profiler = self.profiler()
        profiler.enqueue_backward_compute_events("attn_norm", FakeEvent(0), FakeEvent(12), [(FakeEvent(2), FakeEvent(7))])
        with patch.object(torch.cuda, "is_available", return_value=False), patch.object(torch.cuda, "synchronize"):
            profiler.on_iteration_end(0)
        self.assertEqual(profiler.get_offload_group_stats()["attn_norm"].backward_compute_time_ms, 7)

    def test_native_moe_type_is_detected_without_a_private_flag(self):
        from megatron.core.transformer.moe.moe_layer import MoELayer

        for mlp, expected in ((object.__new__(MoELayer), True), (torch.nn.Linear(2, 2), False)):
            layer = SimpleNamespace(
                mlp=mlp, config=SimpleNamespace(fine_grained_activation_offloading=True, offload_modules=[]),
                input_layernorm=torch.nn.Identity(), pre_mlp_layernorm=torch.nn.Identity(),
            )
            transformer_layer_init_wrapper(lambda target: None)(layer)
            self.assertEqual(layer._is_moe_layer, expected)

    def test_layer_moe_metadata_is_promoted_without_downgrading(self):
        profiler = self.profiler()
        profiler._get_or_create_layer(1, False)
        self.assertTrue(profiler._get_or_create_layer(1, True).is_moe)
        self.assertTrue(profiler._get_or_create_layer(1, False).is_moe)

    def test_expert_groups_cannot_silently_disappear_from_dense_metadata(self):
        profiler = self.profiler()
        self.seed(profiler)
        profiler._offload_group_stats["expert_fc1"] = profiling.OffloadGroupStats(
            group_name="expert_fc1", total_offload_bytes=100 * 1024 ** 2,
            forward_compute_time_ms=2, backward_compute_time_ms=4,
            d2h_time_ms=1, h2d_time_ms=1, d2h_sample_count=3, h2d_sample_count=3,
        )
        optimizer = profiling.OffloadRecomputeOptimizer(
            profiler._offload_group_stats, None, layer_stats=profiler._layer_stats,
            memory_telemetry=profiler._memory_telemetry,
        )
        with self.assertRaisesRegex(ValueError, "no profiled layer is marked as MoE"):
            optimizer.compute_decisions()
        profiler._get_or_create_layer(1, True)
        optimizer.compute_decisions()
        self.assertEqual(optimizer.plan.decisions["expert_fc1"], "KEEP")
        self.assertGreater(optimizer.plan.predicted_peak_mb, 400)

    def test_correlated_allocator_footprint_does_not_add_disjoint_peaks(self):
        profiler = self.profiler()
        snapshots = [
            {"allocated_bytes.all.current": 100, "active_bytes.all.current": 100,
             "inactive_split_bytes.all.current": 0, "inactive_split_bytes.all.peak": 80},
            {"allocated_bytes.all.current": 20, "active_bytes.all.current": 20,
             "inactive_split_bytes.all.current": 80, "inactive_split_bytes.all.peak": 80},
            {"allocated_bytes.all.current": 70, "active_bytes.all.current": 70,
             "inactive_split_bytes.all.current": 50, "inactive_split_bytes.all.peak": 50},
        ]
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "max_memory_allocated", side_effect=[100, 20, 70]), \
             patch.object(torch.cuda, "memory_stats", side_effect=snapshots), \
             patch.object(torch.cuda, "reset_peak_memory_stats"):
            profiler._capture_memory_peak(reset=True)
            profiler._capture_memory_peak(reset=True)
            self.assertEqual(profiler._memory_telemetry["allocator_peak_overhead_bytes"], 0)
            profiler._capture_memory_peak(reset=True)
            self.assertEqual(profiler._memory_telemetry["allocator_peak_overhead_bytes"], 20)

    def test_nonreleasable_allocator_peak_survives_counter_resets(self):
        profiler = self.profiler()
        stats = {
            "inactive_split_bytes.all.current": 5,
            "inactive_split_bytes.all.peak": 8,
            "active_bytes.all.current": 30,
            "allocated_bytes.all.current": 25,
            "reserved_bytes.all.peak": 100,
        }
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "max_memory_allocated", return_value=50), \
             patch.object(torch.cuda, "memory_stats", return_value=stats), \
             patch.object(torch.cuda, "reset_peak_memory_stats"):
            profiler._capture_memory_peak(reset=True)
            self.assertEqual(profiler._memory_telemetry["allocator_unavailable_bytes"], 13)
            self.assertEqual(profiler._iteration_reserved_peak_bytes, 100)
            stats.update({"inactive_split_bytes.all.current": 1,
                          "inactive_split_bytes.all.peak": 2,
                          "active_bytes.all.current": 25})
            profiler._capture_memory_peak()
            self.assertEqual(profiler._memory_telemetry["allocator_unavailable_bytes"], 13)

    def test_allocator_pressure_reaches_memory_guard(self):
        profiler = self.profiler()
        self.seed(profiler)
        profiler._plan = StrategyPlan({}, 0, 800, 1000, 0, 0, 0)
        profiler._memory_telemetry["allocator_peak_overhead_bytes"] = 300 * 1024 ** 2
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "mem_get_info", return_value=(1000 * 1024 ** 2, 1000 * 1024 ** 2)), \
             patch.object(torch.cuda, "memory_reserved", return_value=0):
            profiler._sample_memory_capacity()
        self.assertTrue(profiler._memory_pressure)

    def test_iteration_trace_contains_allocator_and_peak_measurements(self):
        profiler = self.profiler()
        self.seed(profiler)
        profiler._pp_rank = 0
        profiler._iteration_peak_bytes = 512
        profiler._iteration_reserved_peak_bytes = 768
        with tempfile.TemporaryDirectory() as directory:
            prefix = os.path.join(directory, "iterations")
            with patch.dict(os.environ, {"ADAPTIVE_MEM_ITERATION_JSONL": prefix}), \
                 patch.object(torch.distributed, "is_initialized", return_value=False):
                profiler._write_iteration_trace()
            with open(prefix + ".pp0.rank0.jsonl") as source:
                record = json.loads(source.readline())
            self.assertEqual(record["peak_allocated_bytes"], 512)
            self.assertEqual(record["peak_reserved_bytes"], 768)
            self.assertIn("allocator_unavailable_bytes", record["memory"])

    def test_memory_underprediction_is_not_added_repeatedly(self):
        profiler = self.profiler()
        self.seed(profiler)
        profiler._plan = StrategyPlan({}, 0, 200, 1000, 0, 0, 0)
        profiler._plan_baseline_bytes = 100 * 1024 ** 2
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "max_memory_allocated", return_value=220 * 1024 ** 2), \
             patch.object(torch.cuda, "mem_get_info", return_value=(900 * 1024 ** 2, 1000 * 1024 ** 2)), \
             patch.object(torch.cuda, "memory_reserved", return_value=100 * 1024 ** 2):
            profiler._finish_memory_sample()
            profiler._finish_memory_sample()
        self.assertEqual(profiler._memory_telemetry["baseline_peak_bytes"], 120 * 1024 ** 2)

    def test_capacity_drop_requests_replanning_before_training(self):
        profiler = self.profiler()
        self.seed(profiler)
        profiler._plan = StrategyPlan({}, 0, 200, 1000, 0, 0, 0)
        profiler._profiling_done = True
        profiler._optimization_applied = True
        profiler._reoptimize_interval = 1
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "max_memory_allocated", return_value=100 * 1024 ** 2), \
             patch.object(torch.cuda, "mem_get_info", return_value=(60 * 1024 ** 2, 1000 * 1024 ** 2)), \
             patch.object(torch.cuda, "memory_reserved", return_value=100 * 1024 ** 2), \
             patch.object(torch.cuda, "reset_peak_memory_stats"):
            profiler.on_iteration_start(20)
        self.assertTrue(profiler.is_profiling_done())
        self.assertTrue(profiler._memory_pressure)
        profiler.synchronize_memory_guard(20)
        self.assertFalse(profiler.is_optimization_applied())

    def test_peer_pressure_precedes_periodic_reprofiling_on_every_rank(self):
        profiler = self.profiler()
        profiler._profiling_done = True
        profiler._optimization_applied = True
        profiler._reoptimize_interval = 1
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
             patch.object(torch.distributed, "all_reduce") as reduce, \
             patch.object(torch, "tensor", return_value=SimpleNamespace(item=lambda: 1)):
            profiler.synchronize_memory_guard(20)
        reduce.assert_called_once()
        self.assertTrue(profiler.is_profiling_done())
        self.assertFalse(profiler.is_optimization_applied())

    def test_memory_pressure_interrupts_an_active_reprofile_window(self):
        profiler = self.profiler()
        profiler._plan = StrategyPlan({}, 0, 200, 1000, 0, 0, 0)
        profiler._profiling_done = False
        profiler._optimization_applied = True
        profiler._memory_pressure = True
        profiler.synchronize_memory_guard(20)
        self.assertTrue(profiler.is_profiling_done())
        self.assertFalse(profiler.is_optimization_applied())

    def test_periodic_reprofile_starts_after_collective_guard(self):
        profiler = self.profiler()
        profiler._profiling_done = True
        profiler._last_optimize_iter = 0
        profiler._reoptimize_interval = 2
        with contextlib.redirect_stdout(io.StringIO()):
            profiler.on_iteration_start(2)
            self.assertTrue(profiler.is_profiling_done())
            profiler.synchronize_memory_guard(2)
        self.assertFalse(profiler.is_profiling_done())

    def test_single_rank_application_marks_applied(self):
        profiler = self.profiler()
        self.seed(profiler)
        with contextlib.redirect_stdout(io.StringIO()):
            kept, recomputed = profiler.apply_optimization_results()
        self.assertEqual(kept, {"attn_norm"})
        self.assertFalse(recomputed)
        self.assertTrue(profiler.is_optimization_applied())

    def test_all_ranks_contribute_worst_case_memory_and_activation_size(self):
        profiler = self.profiler()
        self.seed(profiler)
        profiler._pp_rank = 0
        local = profiler._profile_payload()
        peer = copy.deepcopy(local)
        peer["memory"]["baseline_peak_bytes"] = 150 * 1024 ** 2
        peer["memory"]["capacity_bytes"] = 700 * 1024 ** 2
        peer["memory"]["allocator_unavailable_bytes"] = 11 * 1024 ** 2
        peer["memory"]["allocator_peak_overhead_bytes"] = 7 * 1024 ** 2
        peer["memory"]["max_inflight_microbatches"] = 4
        peer["profile"]["groups"]["attn_norm"]["total_offload_bytes"] = 200 * 1024 ** 2

        def gather(output, value):
            output[:] = [local, peer] if isinstance(value, dict) else [value, None]

        with patch.object(torch.distributed, "is_initialized", return_value=True), \
             patch.object(torch.distributed, "get_world_size", return_value=2), \
             patch.object(torch.distributed, "all_gather_object", side_effect=gather), \
             contextlib.redirect_stdout(io.StringIO()):
            profiler.apply_optimization_results()
        self.assertEqual(profiler._memory_telemetry["capacity_bytes"], 700 * 1024 ** 2)
        self.assertEqual(profiler._memory_telemetry["allocator_unavailable_bytes"], 11 * 1024 ** 2)
        self.assertEqual(profiler._memory_telemetry["allocator_peak_overhead_bytes"], 7 * 1024 ** 2)
        self.assertEqual(profiler._memory_telemetry["max_inflight_microbatches"], 4)
        self.assertNotIn("attn_norm", profiler.get_skip_offload_groups())
        self.assertLessEqual(profiler._plan.predicted_peak_mb, 700)

    def test_peer_failure_rejects_policy_before_manager_mutation(self):
        profiler = self.profiler()
        self.seed(profiler)
        payload = profiler._profile_payload()

        def gather(output, value):
            output[:] = [payload, payload] if isinstance(value, dict) else [value, "peer cannot fit"]

        with patch.object(torch.distributed, "is_initialized", return_value=True), \
             patch.object(torch.distributed, "get_world_size", return_value=2), \
             patch.object(torch.distributed, "all_gather_object", side_effect=gather), \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError, "peer cannot fit"):
            profiler.apply_optimization_results()
        self.assertFalse(profiler.is_optimization_applied())

    def test_json_round_trip_preserves_new_profile_inputs(self):
        profiler = self.profiler()
        self.seed(profiler)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "profile.json")
            profiler.save_to_json(path)
            loaded = profiling.AdaptiveMemoryProfiler.load_from_json(path)
        self.assertEqual(loaded._memory_telemetry, profiler._memory_telemetry)
        self.assertEqual(loaded._layer_stats[1].modules["attn_norm"].input_bytes, 5 * 1024 ** 2)

    def test_shared_layer_predicate_allows_native_keep_path(self):
        layer = SimpleNamespace(offload_attn_norm=True, offload_mlp_norm=True)
        manager = SimpleNamespace(skip_offload_groups={"attn_norm", "mlp_norm"}, recompute_groups=set())
        self.assertFalse(_fgao_layer_needs_wrapper(layer, None, manager))
        manager.recompute_groups = {"qkv_linear"}
        self.assertTrue(_fgao_layer_needs_wrapper(layer, None, manager))

    def test_nested_group_boundaries_match_without_actual_offload(self):
        profiler = Mock()
        profiler.is_stall_profiling_active.return_value = True
        manager = SimpleNamespace(d2h_stream=Mock(), h2d_stream=Mock())
        with patch.object(runtime.PipelineOffloadManager, "get_instance", return_value=manager), \
             patch.object(runtime, "get_adaptive_profiler", return_value=profiler), \
             patch.object(torch.cuda, "Event", FakeEvent), \
             patch.object(torch.cuda, "current_stream", return_value=Mock()):
            handler = runtime.ChunkOffloadHandler(False, 1)
            handler.should_bulk_offload = lambda: False
            outer = handler.on_group_start_forward("attn_norm")
            inner = handler.on_group_start_forward("qkv_linear")
            self.assertEqual(handler.on_group_commit_forward([]), inner)
            self.assertEqual(handler.on_group_commit_forward([]), outer)
        self.assertFalse(handler._groups_to_offload)
        self.assertFalse(handler._fwd_profile_events)
        self.assertEqual([call.args[0] for call in profiler.enqueue_forward_compute_events.call_args_list],
                         ["qkv_linear", "attn_norm"])

    def test_mixed_attention_modes_preserve_outputs_gradients_and_saved_storage(self):
        torch.manual_seed(12)
        weight = torch.randn(8, 8)
        value = torch.randn(3, 8)

        def reference(hidden):
            query, key, data = hidden.sigmoid(), hidden.tanh(), hidden.square()
            return torch.nn.functional.linear((query * key + data).tanh(), weight)

        baseline = value.clone().requires_grad_()
        reference(baseline).sum().backward()
        for actions in itertools.product((False, True), repeat=3):
            groups = ("qkv_linear", "core_attn", "attn_proj")
            kept = {group for group, keep in zip(groups, actions) if keep}
            manager = SimpleNamespace(skip_offload_groups=kept, recompute_groups=set())
            layer = SimpleNamespace(
                offload_qkv_linear=True, offload_core_attention=True, offload_attn_proj=True,
                config=SimpleNamespace(flash_decode=False), training=True, checkpoint_core_attention=False,
                get_query_key_value_tensors=lambda hidden, states: (hidden.sigmoid(), hidden.tanh(), hidden.square()),
                _adjust_key_value_for_inference=lambda context, query, key, data, rotary, *args: (query, key, data, rotary, None),
                core_attention=lambda query, key, data, mask, **kwargs: (query * key + data).tanh(),
                linear_proj=lambda hidden: (torch.nn.functional.linear(hidden, weight), None),
            )

            def offload_context(enabled):
                return torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor.clone(), lambda tensor: tensor) if enabled else contextlib.nullcontext()

            def commit(*tensors, name, forced_released_tensors):
                for tensor in forced_released_tensors:
                    tensor.untyped_storage().resize_(0)
                return tensors

            wrapped = attention_forward_wrapper(lambda model, hidden: (reference(hidden), None))
            hidden = value.clone().requires_grad_()
            with patch.object(runtime.PipelineOffloadManager, "get_instance", return_value=manager), \
                 patch.object(runtime, "get_fine_grained_offloading_context", side_effect=offload_context), \
                 patch.object(runtime, "fine_grained_offloading_group_start", side_effect=lambda tensor, name: tensor), \
                 patch.object(runtime, "fine_grained_offloading_group_commit", side_effect=commit):
                result, _ = wrapped(layer, hidden)
                torch.testing.assert_close(result, reference(value))
                result.sum().backward()
            torch.testing.assert_close(hidden.grad, baseline.grad)

    def test_mixed_expert_modes_preserve_outputs_gradients_and_saved_storage(self):
        torch.manual_seed(44)
        first_weight = torch.randn(8, 8)
        second_weight = torch.randn(8, 8)
        value = torch.randn(3, 8)
        probabilities = torch.rand(3)
        counts = torch.tensor([3])

        def reference(hidden):
            intermediate = torch.nn.functional.linear(hidden, first_weight).sigmoid()
            activated = torch.nn.functional.silu(intermediate) * probabilities.unsqueeze(-1)
            return torch.nn.functional.linear(activated, second_weight)

        baseline = value.clone().requires_grad_()
        reference(baseline).sum().backward()
        for actions in itertools.product((False, True), repeat=2):
            kept = {group for group, keep in zip(("expert_fc1", "moe_act"), actions) if keep}
            manager = SimpleNamespace(skip_offload_groups=kept, recompute_groups=set())
            layer = SimpleNamespace(
                offload_expert_fc1=True, offload_moe_act=True,
                config=SimpleNamespace(fp8=False, moe_apply_probs_on_input=False,
                                       bias_activation_fusion=False, gated_linear_unit=False),
                activation_recompute=False, activation_func=torch.nn.functional.silu,
                linear_fc1=lambda hidden, tokens: (torch.nn.functional.linear(hidden, first_weight).sigmoid(), None),
                linear_fc2=lambda hidden, tokens: (torch.nn.functional.linear(hidden, second_weight), None),
            )

            def offload_context(enabled):
                return torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor.clone(), lambda tensor: tensor) if enabled else contextlib.nullcontext()

            def commit(*tensors, name, forced_released_tensors):
                for tensor in forced_released_tensors:
                    tensor.untyped_storage().resize_(0)
                return tensors

            wrapped = te_grouped_mlp_forward_wrapper(lambda model, hidden, *args: (reference(hidden), None))
            hidden = value.clone().requires_grad_()
            with patch.object(runtime.PipelineOffloadManager, "get_instance", return_value=manager), \
                 patch.object(runtime, "get_fine_grained_offloading_context", side_effect=offload_context), \
                 patch.object(runtime, "fine_grained_offloading_group_start", side_effect=lambda tensor, name: tensor), \
                 patch.object(runtime, "fine_grained_offloading_group_commit", side_effect=commit):
                result, _ = wrapped(layer, hidden, counts, probabilities)
                torch.testing.assert_close(result, reference(value))
                result.sum().backward()
            torch.testing.assert_close(hidden.grad, baseline.grad)


if __name__ == "__main__":
    unittest.main()
