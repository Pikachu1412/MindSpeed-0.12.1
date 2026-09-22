import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mindspeed.core.pipeline_parallel.adaptive_offload import fine_grained_activation_offload as runtime
from mindspeed.core.pipeline_parallel.adaptive_offload.offload_transport_audit import OffloadTransportAudit
from mindspeed.core.pipeline_parallel.adaptive_offload.schedules_wrapper import (
    forward_backward_pipelining_without_interleaving_wrapper,
)


class TestOffloadTransport(unittest.TestCase):
    def test_cpu_affinity_is_opt_in_rank_local_and_applied_once(self):
        with patch.dict(os.environ, {'MEGATRON_OFFLOAD_CPU_AFFINITY': '1-2;5,7-8', 'LOCAL_RANK': '1'}), \
                patch.object(runtime, '_applied_cpu_affinity', None), \
                patch.object(os, 'sched_getaffinity', return_value=set(range(10))), \
                patch.object(os, 'sched_setaffinity') as setter:
            runtime.set_ideal_affinity_for_current_gpu()
            runtime.set_ideal_affinity_for_current_gpu()
            setter.assert_called_once_with(0, {5, 7, 8})

    def test_cpu_affinity_rejects_disallowed_cpu(self):
        with patch.dict(os.environ, {'MEGATRON_OFFLOAD_CPU_AFFINITY': '9', 'LOCAL_RANK': '0'}), \
                patch.object(runtime, '_applied_cpu_affinity', None), \
                patch.object(os, 'sched_getaffinity', return_value={0, 1}), \
                patch.object(os, 'sched_setaffinity') as setter:
            with self.assertRaisesRegex(ValueError, 'outside the allowed'):
                runtime.set_ideal_affinity_for_current_gpu()
            setter.assert_not_called()

    def test_cpu_affinity_disabled_preserves_current_mask(self):
        with patch.dict(os.environ, {'MEGATRON_OFFLOAD_CPU_AFFINITY': ''}), \
                patch.object(os, 'sched_setaffinity') as setter:
            runtime.set_ideal_affinity_for_current_gpu()
            setter.assert_not_called()

    def test_cpu_affinity_rejects_missing_rank_mask_and_bad_range(self):
        with patch.dict(os.environ, {'MEGATRON_OFFLOAD_CPU_AFFINITY': '1-2', 'LOCAL_RANK': '2'}):
            with self.assertRaisesRegex(ValueError, 'one mask per local rank'):
                runtime.set_ideal_affinity_for_current_gpu()
        with self.assertRaisesRegex(ValueError, 'Invalid CPU affinity range'):
            runtime._parse_cpu_affinity('9-3')

    def test_unmodeled_transport_options_reject_adaptive_policy(self):
        with patch.object(runtime, 'ADAPTIVE_OFFLOAD_ENABLED', True), \
                patch.object(runtime, 'LAYER_PREFETCH_DEPTH', 2):
            with self.assertRaisesRegex(RuntimeError, 'ADAPTIVE_OFFLOAD=0'):
                runtime.PipelineOffloadManager()

    def test_event_sync_avoids_unrelated_compute_barrier(self):
        handler = runtime.ChunkOffloadHandler.__new__(runtime.ChunkOffloadHandler)
        handler.h2d_stream = Mock()
        with patch.object(runtime, 'RELOAD_EVENT_SYNC_ENABLED', True):
            handler._prepare_reload_stream()
        handler.h2d_stream.wait_stream.assert_not_called()
        with patch.object(runtime, 'RELOAD_EVENT_SYNC_ENABLED', False), \
                patch.object(torch.cuda, 'current_stream', return_value='compute'):
            handler._prepare_reload_stream()
        handler.h2d_stream.wait_stream.assert_called_once_with('compute')

    def test_reloaded_storage_is_protected_until_compute_finishes(self):
        handler = runtime.ChunkOffloadHandler.__new__(runtime.ChunkOffloadHandler)
        tensor = Mock()
        tag = (4, 0)
        handler._tensor_tag_to_state = {tag: tensor}
        handler._reloaded_tensor_tags = {tag}
        with patch.object(runtime, 'RELOAD_EVENT_SYNC_ENABLED', True), \
                patch.object(torch.cuda, 'current_stream', return_value='compute'):
            self.assertIs(handler.tensor_pop(tag), tensor)
        tensor.record_stream.assert_called_once_with('compute')
        self.assertEqual(handler._reloaded_tensor_tags, set())

    def test_layer_depth_zero_disables_prefetch_without_changing_keep(self):
        handler = runtime.ChunkOffloadHandler.__new__(runtime.ChunkOffloadHandler)
        handler._build_cross_layer_index = Mock()
        with patch.object(runtime, 'LAST_LAYER_NO_OFFLOAD_ENABLED', True), \
                patch.object(runtime, 'LAYER_PREFETCH_DEPTH', 0):
            handler.prefetch_previous_layer(28)
        handler._build_cross_layer_index.assert_not_called()

    def test_layer_prefetch_delay_and_depth(self):
        handler = runtime.ChunkOffloadHandler.__new__(runtime.ChunkOffloadHandler)
        handler._build_cross_layer_index = Mock()
        handler._total_groups_per_layer = 7
        handler._layer_backward_groups = {}
        handler._prefetch_previous_layer_batch = Mock()
        with patch.object(runtime, 'LAST_LAYER_NO_OFFLOAD_ENABLED', True), \
                patch.object(runtime, 'LAYER_PREFETCH_DEPTH', 2), \
                patch.object(runtime, 'LAYER_PREFETCH_DELAY_GROUPS', 1):
            handler.prefetch_previous_layer(28)
            handler._prefetch_previous_layer_batch.assert_not_called()
            handler.prefetch_previous_layer(27)
        self.assertEqual([entry.args[0] for entry in handler._prefetch_previous_layer_batch.call_args_list], [27, 20])

    def test_schedule_uses_model_config_when_config_argument_absent(self):
        config = SimpleNamespace(fine_grained_activation_offloading=True)
        model = SimpleNamespace(config=config)
        manager = Mock()

        def schedule(model, forward_only=False):
            return 'completed'

        with patch.object(runtime.PipelineOffloadManager, 'get_instance', return_value=manager):
            result = forward_backward_pipelining_without_interleaving_wrapper(schedule)(model=[model])
        self.assertEqual(result, 'completed')
        manager.reset.assert_called_once()
        manager.transport_audit.finish.assert_called_once()

    def test_schedule_accepts_positional_model(self):
        model = SimpleNamespace(config=SimpleNamespace(fine_grained_activation_offloading=True))
        manager = Mock()

        def schedule(model, forward_only=False):
            return 7

        with patch.object(runtime.PipelineOffloadManager, 'get_instance', return_value=manager):
            self.assertEqual(forward_backward_pipelining_without_interleaving_wrapper(schedule)([model], False), 7)
        manager.reset.assert_called_once()
        manager.transport_audit.finish.assert_called_once()

    def test_forward_only_does_not_reset_training_offload(self):
        model = SimpleNamespace(config=SimpleNamespace(fine_grained_activation_offloading=True))
        manager = Mock()

        def schedule(model, forward_only=False):
            return None

        with patch.object(runtime.PipelineOffloadManager, 'get_instance', return_value=manager):
            forward_backward_pipelining_without_interleaving_wrapper(schedule)(model=[model], forward_only=True)
        manager.reset.assert_not_called()
        manager.transport_audit.finish.assert_not_called()

    def test_original_small_tensor_and_parameter_filters(self):
        handler = runtime.ChunkOffloadHandler.__new__(runtime.ChunkOffloadHandler)
        handler.min_offloaded_tensor_size = 1024
        self.assertFalse(handler.tensor_need_offloading_checker(torch.ones(1023)))
        self.assertTrue(handler.tensor_need_offloading_checker(torch.ones(1024)))
        self.assertFalse(handler.tensor_need_offloading_checker(torch.nn.Parameter(torch.ones(2048))))
        excluded = torch.ones(2048)
        excluded.offloading_activation = False
        self.assertFalse(handler.tensor_need_offloading_checker(excluded))

    def test_audit_accounts_for_last_layer_keep(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "trace")
            with patch.dict(os.environ, {"MEGATRON_OFFLOAD_TRANSPORT_JSONL": prefix}), \
                    patch.object(torch.distributed, "is_initialized", return_value=False), \
                    patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                    patch.object(torch.cuda, "max_memory_reserved", return_value=0):
                audit = OffloadTransportAudit()
                audit.pinned_memory_pool = SimpleNamespace(stats={'hit_count': 9, 'retained_bytes': 1024})
                audit.start()
                for phase in ("candidate", "d2h", "h2d"):
                    audit.record(phase, "qkv_linear", 0, False, 1024)
                for phase in ("candidate", "kept"):
                    audit.record(phase, "qkv_linear", 1, True, 1024)
                audit.finish()
            result = json.loads(Path(prefix + ".rank0.jsonl").read_text())
            self.assertTrue(result["eligible_activation_accounting_valid"])
            self.assertEqual(result["pinned_memory_pool"]["hit_count"], 9)
            self.assertEqual(result["groups"]["layer1:qkv_linear"]["kept_bytes"], 1024)

    def test_audit_rejects_missing_reload(self):
        with patch.dict(os.environ, {"MEGATRON_OFFLOAD_TRANSPORT_JSONL": ""}), \
                patch.object(torch.distributed, "is_initialized", return_value=False), \
                patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                patch.object(torch.cuda, "max_memory_reserved", return_value=0):
            audit = OffloadTransportAudit()
            audit.enabled = True
            audit.start()
            for phase in ("candidate", "d2h"):
                audit.record(phase, "expert_fc1", 0, False, 2048)
            with self.assertRaisesRegex(RuntimeError, "accounting failed"):
                audit.finish()
