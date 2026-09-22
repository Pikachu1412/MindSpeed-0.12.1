import unittest
from collections import Counter
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mindspeed.core.pipeline_parallel.adaptive_offload import fine_grained_activation_offload as runtime
from mindspeed.core.pipeline_parallel.adaptive_offload.activation_transfer_scheduler import (
    ActivationTransferScheduler, ByteCredits, ModuleTransfer,
)


class TestOffloadGroupEvents(unittest.TestCase):
    def setUp(self):
        self.links = ExitStack()
        self.addCleanup(self.links.close)
        self.trace = []
        self.events = []
        self.handler = runtime.ChunkOffloadHandler.__new__(runtime.ChunkOffloadHandler)
        self.handler.d2h_stream = Mock(name='d2h')
        self.handler.h2d_stream = Mock(name='h2d')
        self.handler._tensor_tag_to_state = {}
        self.handler._group_layer_metadata = {4: (1, False), 9: (2, False)}
        self.handler._offload_events = {}
        self.handler._offload_events_by_id = {}
        self.handler._reload_events = {}
        self.handler._reload_events_by_id = {}
        self.handler._reload_issued = set()
        self.handler._reloaded_tensor_tags = set()
        self.handler.is_first_last_layer = Mock(return_value=False)
        self.handler.tensor_need_offloading_checker = lambda tensor: tensor.eligible
        self.handler.offload = Mock(side_effect=self.offload)
        self.handler.reload = Mock(side_effect=self.reload)
        self.manager = SimpleNamespace(transport_audit=Mock(), d2h_stream=self.handler.d2h_stream,
                                       h2d_stream=self.handler.h2d_stream)
        self.links.enter_context(patch.object(runtime.PipelineOffloadManager, 'get_instance', return_value=self.manager))
        self.links.enter_context(patch.object(runtime, 'get_adaptive_profiler', return_value=None))
        self.links.enter_context(patch.object(runtime, 'OFFLOAD_PROFILING', False))
        self.links.enter_context(patch.object(runtime, 'RELOAD_EVENT_SYNC_ENABLED', True))
        self.links.enter_context(patch.object(torch.cuda, 'stream', side_effect=lambda stream: nullcontext()))
        self.links.enter_context(patch.object(torch.cuda, 'current_stream', return_value=self.handler.h2d_stream))
        self.links.enter_context(patch.object(torch.cuda, 'Event', side_effect=self.make_event))

    def tensor(self, eligible=True):
        return SimpleNamespace(shape=(8,), dtype=torch.float32, eligible=eligible,
                               numel=lambda: 8, element_size=lambda: 4, record_stream=Mock())

    def offload(self, tensor):
        self.trace.append('d2h_copy')
        return ('device', tensor)

    def reload(self, state):
        self.trace.append('h2d_copy')
        return self.tensor()

    def make_event(self, **kwargs):
        event = Mock()
        event.record.side_effect = lambda stream: self.trace.append('completion')
        self.events.append(event)
        return event

    def fill(self, group=4, count=3, eligible=True):
        tensors = [self.tensor(eligible) for _ in range(count)]
        self.handler._tensor_tag_to_state.update({(group, index): tensor for index, tensor in enumerate(tensors)})
        return tensors

    def test_offload_records_one_completion_after_all_copies(self):
        tensors = self.fill()
        self.fill(group=9, count=1)
        self.handler.bulk_offload_group((4, 'mlp'))
        self.assertEqual(self.trace, ['d2h_copy'] * 3 + ['completion'])
        self.assertEqual(len(self.events), 1)
        self.events[0].record.assert_called_once_with(self.handler.d2h_stream)
        self.assertIs(self.handler._offload_events_by_id[4], self.events[0])
        for tensor in tensors:
            tensor.record_stream.assert_called_once_with(self.handler.d2h_stream)
        self.assertNotIsInstance(self.handler._tensor_tag_to_state[(9, 0)], tuple)

    def test_reload_waits_once_and_keeps_d2h_completion_immutable(self):
        self.fill()
        self.handler.bulk_offload_group((4, 'mlp'))
        d2h_done = self.events[0]
        self.trace.clear()
        self.assertTrue(self.handler.bulk_reload_group((4, 'mlp')))
        self.assertEqual(self.trace, ['h2d_copy'] * 3 + ['completion'])
        self.handler.h2d_stream.wait_event.assert_called_once_with(d2h_done)
        d2h_done.record.assert_called_once_with(self.handler.d2h_stream)
        self.assertEqual(len(self.events), 2)
        self.assertIs(self.handler._reload_events_by_id[4], self.events[1])
        self.assertIsNot(self.handler._reload_events_by_id[4], d2h_done)
        self.assertEqual(self.handler._reloaded_tensor_tags, {(4, 0), (4, 1), (4, 2)})

    def test_duplicate_reload_does_not_repeat_copies_or_events(self):
        self.fill(count=1)
        self.handler.bulk_offload_group((4, 'mlp'))
        self.handler.bulk_reload_group((4, 'mlp'))
        self.assertTrue(self.handler.bulk_reload_group((4, 'mlp')))
        self.handler.reload.assert_called_once()
        self.assertEqual(len(self.events), 2)
        self.handler.h2d_stream.wait_event.assert_called_once()

    def test_same_name_groups_wait_for_their_own_completion(self):
        self.fill(count=1)
        self.fill(group=9, count=1)
        self.handler.bulk_offload_group((4, 'mlp'))
        first_done = self.events[-1]
        self.handler.bulk_offload_group((9, 'mlp'))
        second_done = self.events[-1]
        self.handler.bulk_reload_group((4, 'mlp'))
        self.handler.bulk_reload_group((9, 'mlp'))
        self.assertEqual([call.args[0] for call in self.handler.h2d_stream.wait_event.call_args_list],
                         [first_done, second_done])
        first_done.record.assert_called_once()
        second_done.record.assert_called_once()

    def test_group_without_eligible_tensors_allocates_no_events(self):
        self.fill(eligible=False)
        self.handler.bulk_offload_group((4, 'mlp'))
        self.assertTrue(self.handler.bulk_reload_group((4, 'mlp')))
        self.assertEqual(self.events, [])
        self.handler.offload.assert_not_called()
        self.handler.reload.assert_not_called()
        self.handler.h2d_stream.wait_event.assert_not_called()

    def scheduler(self, measured):
        scheduler = ActivationTransferScheduler()
        scheduler.manager = self.manager
        scheduler.torch = torch
        scheduler.measure_transfers = measured
        scheduler.d2h = ByteCredits(1024)
        scheduler.h2d = ByteCredits(1024)
        scheduler.largest = 128
        scheduler.counts = Counter()
        scheduler.trace = Mock()
        scheduler.event = Mock(side_effect=lambda *args, **kwargs: Mock())
        return scheduler

    def test_unmeasured_scheduler_reuses_runtime_completions(self):
        self.fill()
        self.handler.bulk_offload_group((4, 'mlp'))
        scheduler = self.scheduler(False)
        ticket = ModuleTransfer(self.handler, 'mlp', 'OFFLOAD', 0, group_id=4)
        ticket.transfer_start = None
        scheduler.commit(ticket)
        self.assertIs(ticket.d2h_done, self.handler._offload_events_by_id[4])
        self.assertTrue(scheduler.issue(ticket, 'demand', demand=True))
        self.assertIs(ticket.h2d_done, self.handler._reload_events_by_id[4])
        scheduler.event.assert_not_called()
        self.handler.h2d_stream.wait_event.assert_called_once_with(ticket.d2h_done)
        self.assertEqual(scheduler.d2h.used, 96)
        self.assertEqual(scheduler.h2d.used, 96)

    def test_measured_scheduler_keeps_separate_timing_completions(self):
        self.fill(count=1)
        self.handler.bulk_offload_group((4, 'mlp'))
        scheduler = self.scheduler(True)
        ticket = ModuleTransfer(self.handler, 'mlp', 'OFFLOAD', 0, group_id=4)
        ticket.transfer_start = None
        scheduler.commit(ticket)
        self.assertIsNot(ticket.d2h_done, self.handler._offload_events_by_id[4])
        self.assertTrue(scheduler.issue(ticket, 'demand', demand=True))
        self.assertIsNot(ticket.h2d_done, self.handler._reload_events_by_id[4])
        self.assertEqual(scheduler.event.call_count, 3)
        self.assertTrue(all(call.kwargs['timing'] for call in scheduler.event.call_args_list))
