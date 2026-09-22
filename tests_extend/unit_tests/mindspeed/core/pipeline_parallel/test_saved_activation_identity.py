import gc
import unittest
import weakref
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mindspeed.core.pipeline_parallel.adaptive_offload import fine_grained_activation_offload as runtime
from mindspeed.core.pipeline_parallel.adaptive_offload.auto_activation_memory import observe_saved_activations
from mindspeed.core.pipeline_parallel.adaptive_offload.offload_transport_audit import OffloadTransportAudit
from mindspeed.core.pipeline_parallel.adaptive_offload.saved_activation_identity import SavedActivationIdentity


class TestSavedActivationIdentity(unittest.TestCase):
    def test_only_same_object_can_alias(self):
        tensor = torch.randn(8, 4)
        ledger = SavedActivationIdentity()
        ledger.record(0, tensor)
        ledger.record(1, tensor)
        detached = tensor.detach()
        ledger.record(2, detached)
        self.assertEqual(ledger.aliases(), {1: 0})

    def test_save_versions_and_submission_version_must_match(self):
        tensor = torch.randn(8, 4)
        ledger = SavedActivationIdentity()
        ledger.record(0, tensor)
        tensor.add_(1)
        ledger.record(1, tensor)
        ledger.record(2, tensor)
        self.assertEqual(ledger.aliases(), {2: 1})
        tensor.add_(1)
        self.assertEqual(ledger.aliases(), {})

    def test_layout_or_storage_change_disables_alias(self):
        for mutation in (lambda tensor: tensor.transpose_(0, 1),
                         lambda tensor: setattr(tensor, 'data', tensor.clone())):
            with self.subTest(mutation=mutation):
                tensor = torch.randn(8, 4)
                ledger = SavedActivationIdentity()
                ledger.record(0, tensor)
                ledger.record(1, tensor)
                mutation(tensor)
                self.assertEqual(ledger.aliases(), {})

    def test_untracked_inference_tensor_is_not_deduplicated(self):
        with torch.inference_mode():
            tensor = torch.randn(8, 4)
        ledger = SavedActivationIdentity()
        ledger.record(0, tensor)
        ledger.record(1, tensor)
        self.assertEqual(ledger.aliases(), {})

    def test_ledger_does_not_retain_activation(self):
        tensor = torch.randn(8, 4)
        reference = weakref.ref(tensor)
        ledger = SavedActivationIdentity()
        ledger.record(0, tensor)
        ledger.record(1, tensor)
        del tensor
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(ledger.aliases(), {})

    def test_detached_witness_survives_original_object(self):
        tensor = torch.randn(8, 4)
        witnesses = [tensor.detach(), tensor.detach()]
        ledger = SavedActivationIdentity()
        for index, witness in enumerate(witnesses):
            ledger.record(index, tensor, witness=witness)
        del tensor
        gc.collect()
        self.assertEqual(ledger.aliases(), {1: 0})
        witnesses[0].add_(1)
        self.assertEqual(ledger.aliases(), {})


class TestOffloadIdentityLifecycle(unittest.TestCase):
    def setUp(self):
        self.links = ExitStack()
        self.addCleanup(self.links.close)
        self.audit = OffloadTransportAudit()
        self.audit.enabled = True
        self.audit.path = None
        self.stream = Mock()
        self.manager = SimpleNamespace(d2h_stream=self.stream, h2d_stream=self.stream,
                                       transport_audit=self.audit)
        self.links.enter_context(patch.object(runtime.PipelineOffloadManager, 'get_instance', return_value=self.manager))
        self.links.enter_context(patch.object(runtime, 'get_adaptive_profiler', return_value=None))
        self.links.enter_context(patch.object(runtime, 'OFFLOAD_PROFILING', False))
        self.links.enter_context(patch.object(runtime, 'RELOAD_EVENT_SYNC_ENABLED', True))
        self.links.enter_context(patch.object(torch.cuda, 'stream', side_effect=lambda stream: nullcontext()))
        self.links.enter_context(patch.object(torch.cuda, 'current_stream', return_value=self.stream))
        self.links.enter_context(patch.object(torch.cuda, 'Event', side_effect=lambda **kwargs: Mock()))
        self.links.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
        self.links.enter_context(patch.object(torch.cuda, 'max_memory_reserved', return_value=0))
        self.record_stream = self.links.enter_context(patch.object(torch.Tensor, 'record_stream'))
        self.handler = self.make_handler()
        self.manager.cur_forward_chunk = lambda: self.handler

    def make_handler(self):
        handler = runtime.ChunkOffloadHandler(False, 1)
        handler._group_id_to_name[0] = 'attention'
        handler._group_layer_metadata[0] = (0, False)
        handler.layer_index = 0
        handler.is_first_last_layer = lambda: False
        handler.offload = Mock(side_effect=lambda tensor: (tensor.device, tensor.detach().clone()))
        handler.reload = Mock(side_effect=lambda state: state[1].clone())
        return handler

    def test_copy_once_and_consume_every_logical_tag(self):
        for order in ((0, 1, 2), (2, 0, 1)):
            with self.subTest(order=order):
                self.audit.start()
                handler = self.make_handler()
                tensor = torch.randn(8, 4)
                tags = [handler.tensor_push(tensor) for _ in range(3)]
                handler.bulk_offload_group((0, 'attention'))
                self.assertEqual(len(handler._tensor_tag_to_state), 1)
                handler.offload.assert_called_once()
                handler.bulk_reload_group((0, 'attention'))
                handler.reload.assert_called_once()
                for position, index in enumerate(order):
                    recovered = handler.tensor_pop(tags[index])
                    torch.testing.assert_close(recovered, tensor, rtol=0, atol=0)
                    self.assertEqual(len(handler._tensor_tag_to_state), int(position < 2))
                    with self.assertRaises(AssertionError):
                        handler.tensor_pop(tags[index])
                self.assertEqual(handler._tensor_aliases, {})
                self.assertEqual(handler._tensor_consumers, {})
                self.assertEqual(handler._reloaded_tensor_tags, set())
                self.assertEqual(handler._saved_activation_groups, {})
                group = self.audit.groups['layer0:attention']
                self.assertEqual(group['candidate_bytes'], 3 * 128)
                self.assertEqual(group['deduplicated_bytes'], 2 * 128)
                self.assertEqual(group['d2h_bytes'], 128)
                self.assertEqual(group['h2d_bytes'], 128)
                self.audit.finish()

    def test_mutation_before_submission_preserves_independent_copies(self):
        tensor = torch.randn(8, 4)
        tags = [self.handler.tensor_push(tensor) for _ in range(2)]
        tensor.add_(1)
        self.handler.bulk_offload_group((0, 'attention'))
        self.assertEqual(self.handler.offload.call_count, 2)
        self.handler.bulk_reload_group((0, 'attention'))
        for tag in tags:
            self.handler.tensor_pop(tag)
        self.audit.finish()

    def test_shared_cpu_buffer_is_returned_once(self):
        self.manager.pinned_memory_pool = SimpleNamespace(mark_used=Mock())
        self.handler.reload = runtime.ChunkOffloadHandler.reload.__get__(self.handler)
        tensor = torch.randn(8, 4)
        tags = [self.handler.tensor_push(tensor) for _ in range(3)]
        self.handler.bulk_offload_group((0, 'attention'))
        self.handler.bulk_reload_group((0, 'attention'))
        self.manager.pinned_memory_pool.mark_used.assert_called_once()
        for tag in tags:
            self.handler.tensor_pop(tag)
        self.audit.finish()

    def test_every_shared_consumer_stream_is_protected(self):
        for event_sync in (False, True):
            with self.subTest(event_sync=event_sync), patch.object(runtime, 'RELOAD_EVENT_SYNC_ENABLED', event_sync):
                handler = self.make_handler()
                tensor = torch.randn(8, 4)
                tags = [handler.tensor_push(tensor) for _ in range(2)]
                handler.bulk_offload_group((0, 'attention'))
                handler.bulk_reload_group((0, 'attention'))
                streams = [Mock(name='consumer_first'), Mock(name='consumer_second')]
                self.record_stream.reset_mock()
                with patch.object(torch.cuda, 'current_stream', side_effect=streams):
                    handler.tensor_pop(tags[1])
                    handler.tensor_pop(tags[0])
                self.assertEqual([call.args[0] for call in self.record_stream.call_args_list], streams)
                self.assertEqual(handler._reloaded_tensor_tags, set())

    def test_groups_and_microbatches_do_not_share_transfers(self):
        tensor = torch.randn(8, 4)
        for handler in (self.handler, self.make_handler()):
            for group_id in (0, 1):
                handler._offloaded_group_index = group_id
                handler._group_id_to_name[group_id] = 'attention'
                handler._group_layer_metadata[group_id] = (0, False)
                tags = [handler.tensor_push(tensor) for _ in range(2)]
                handler.bulk_offload_group((group_id, 'attention'))
                handler.bulk_reload_group((group_id, 'attention'))
                for tag in reversed(tags):
                    handler.tensor_pop(tag)
            self.assertEqual(handler.offload.call_count, 2)
            self.assertEqual(handler.reload.call_count, 2)
        self.audit.finish()

    def test_small_tensor_and_parameter_filter_unchanged(self):
        self.handler.min_offloaded_tensor_size = 64
        tensors = [torch.ones(8), torch.nn.Parameter(torch.ones(128))]
        tags = [self.handler.tensor_push(tensor) for tensor in tensors for _ in range(2)]
        self.handler.bulk_offload_group((0, 'attention'))
        self.handler.offload.assert_not_called()
        self.assertEqual(self.handler._tensor_aliases, {})
        for tag in tags:
            self.handler.tensor_pop(tag)
        self.assertEqual(self.audit.groups, {})

    def test_full_autograd_output_and_gradients_match(self):
        for dtype in (torch.float32, torch.bfloat16):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    self.audit.start()
                    handler = self.make_handler()
                    tensor = torch.randn(8, 8, dtype=dtype)
                    tensor = (tensor[:, ::2] if strided else tensor).requires_grad_(True)
                    expected = (tensor * tensor).sum()
                    expected.backward()
                    expected_gradient = tensor.grad.clone()
                    tensor.grad = None
                    with torch.autograd.graph.saved_tensors_hooks(handler.tensor_push, handler.tensor_pop):
                        actual = (tensor * tensor).sum()
                    handler.bulk_offload_group((0, 'attention'))
                    handler.bulk_reload_group((0, 'attention'))
                    actual.backward()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(tensor.grad, expected_gradient, rtol=0, atol=0)
                    self.assertEqual(handler.offload.call_count, 1)
                    self.assertEqual(handler._tensor_tag_to_state, {})
                    self.audit.finish()

    def test_observer_counts_physical_and_logical_bytes_separately(self):
        profiler = SimpleNamespace(record_group_offload_bytes=Mock(),
                                   _get_or_create_offload_group=lambda key: SimpleNamespace(non_offloadable_bytes=0))
        tensor = torch.randn(8, 4, requires_grad=True)
        with observe_saved_activations(torch.nn.Identity(), profiler, 'attention'):
            output = tensor * tensor
        profiler.record_group_offload_bytes.assert_called_once_with('attention', 128, logical_bytes=256)
        output.sum().backward()
        torch.testing.assert_close(tensor.grad, 2 * tensor, rtol=0, atol=0)

    def test_audit_rejects_missing_alias_or_reload_bytes(self):
        for phase in ('deduplicated', 'h2d'):
            with self.subTest(phase=phase):
                self.audit.start()
                for name, count in (('candidate', 128), ('candidate', 128), ('deduplicated', 128), ('d2h', 128), ('h2d', 128)):
                    self.audit.record(name, 'attention', 0, False, count)
                self.audit.finish()
                self.audit.groups['layer0:attention'][phase + '_bytes'] = 0
                with self.assertRaises(RuntimeError):
                    self.audit.finish()
