import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mindspeed.core.pipeline_parallel.adaptive_offload.dense_activation_boundary import (
    ChildBoundaryObservation, _allocation_counters, _allocation_volume, _storage_key,
)


PARENT = 'layer1.dense_mlp'


def counters(allocated, freed, current, peak=2**40):
    return {'allocated_bytes.all.allocated': allocated, 'allocated_bytes.all.freed': freed,
            'allocated_bytes.all.current': current, 'allocated_bytes.all.peak': peak}


class Event:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def record(self):
        self.events.append(self.name + '.record')

    def synchronize(self):
        self.events.append(self.name + '.wait')

    def elapsed_time(self, end):
        return 0.2


class TestWorkspaceAllocation(unittest.TestCase):
    def test_allocation_volume_counts_allocations_despite_larger_frees(self):
        self.assertEqual(_allocation_volume((10000, 2000, 8000), (11000, 7000, 4000)), 1000)

    def test_allocation_volume_includes_serial_temporary_allocations(self):
        self.assertEqual(_allocation_volume((1000, 0, 1000), (1400, 300, 1100)), 400)

    def test_counter_regression_and_inconsistent_current_are_rejected(self):
        before = (1000, 800, 200)
        for after in (None, (500, 0, 700), (1100, 700, 400), (1100, 900, 201)):
            with self.subTest(after=after):
                self.assertIsNone(_allocation_volume(before, after))
        self.assertIsNone(_allocation_volume(None, before))

    def test_missing_negative_noninteger_and_failed_counters_are_rejected(self):
        for stats in ({}, [], counters(-1, 0, 0), counters(True, 0, 0), counters(1.0, 0, 0)):
            with self.subTest(stats=stats), patch.object(torch.cuda, 'memory_stats', return_value=stats):
                self.assertIsNone(_allocation_counters())
        for exception in (AttributeError('unsupported'), RuntimeError('unavailable')):
            with patch.object(torch.cuda, 'memory_stats', side_effect=exception):
                self.assertIsNone(_allocation_counters())

    def test_activation_waits_at_sample_boundaries_without_resetting_peaks(self):
        events = []
        observation = ChildBoundaryObservation(None, PARENT)
        input_tensor = torch.ones(2, 12)
        stream = SimpleNamespace(synchronize=lambda: events.append('stream.wait'))
        before, after = counters(1000, 0, 1000), counters(1256, 512, 744)

        def memory_stats():
            events.append('stats')
            return before if events.count('stats') == 1 else after

        def operation(value):
            events.append('operation')
            return value + 1

        with patch.object(torch.cuda, 'current_stream', return_value=stream), \
                patch.object(torch.cuda, 'memory_stats', side_effect=memory_stats), \
                patch.object(torch.cuda, 'Event', side_effect=[Event('start', events), Event('end', events)]), \
                patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('global synchronization')), \
                patch.object(torch.cuda, 'reset_peak_memory_stats', side_effect=AssertionError('peak reset')), \
                patch.object(torch.cuda, 'reset_accumulated_memory_stats', side_effect=AssertionError('counter reset')), \
                patch.object(torch.cuda, 'max_memory_allocated', side_effect=AssertionError('historical peak')):
            output = observation.activation(operation, (input_tensor,))
        self.assertTrue(torch.equal(output, input_tensor + 1))
        self.assertEqual(observation.activation_workspace_bytes, 256)
        self.assertEqual(observation.activation_allocation_counters, ((1000, 0, 1000), (1256, 512, 744)))
        self.assertEqual(events, ['stream.wait', 'stats', 'start.record', 'operation',
                                  'end.record', 'end.wait', 'stats'])
        self.assertFalse(observation.inside_activation)

    def test_activation_exception_preserves_error_and_leaves_no_valid_sample(self):
        observation = ChildBoundaryObservation(None, PARENT)

        def operation(value):
            raise ValueError('original operation failed')

        with patch.object(torch.cuda, 'current_stream'), patch.object(torch.cuda, 'Event'), \
                patch.object(torch.cuda, 'memory_stats', return_value=counters(1000, 0, 1000)) as stats:
            with self.assertRaisesRegex(ValueError, 'original operation failed'):
                observation.activation(operation, (torch.ones(2, 12),))
        self.assertEqual(stats.call_count, 1)
        self.assertIsNone(observation.activation_workspace_bytes)
        self.assertIsNone(observation.activation_events)
        self.assertFalse(observation.inside_activation)

    def test_copy_workspace_records_allocator_size_not_only_storage_size(self):
        current = 512 * 2**20
        before = counters(current, 0, current)
        budget = dict(before, **{'active_bytes.all.current': current, 'reserved_bytes.all.current': 3 * 2**30,
                                'inactive_split_bytes.all.current': 0})
        after = counters(current + 128, 0, current + 128)
        profiler = SimpleNamespace(_current_memory_limit_mb=lambda: 4096, _reoptimize_count=0)
        observation = ChildBoundaryObservation(profiler, PARENT)
        output = torch.ones(2, 12)
        with patch.object(torch.cuda, 'current_stream'), patch.object(torch.cuda, 'Event'), \
                patch.object(torch.cuda, 'mem_get_info', return_value=(4 * 2**30, 8 * 2**30)), \
                patch.object(torch.cuda, 'memory_allocated', return_value=current), \
                patch.object(torch.cuda, 'memory_stats', side_effect=[budget, before, after]):
            observation.probe_copy(output, ())
        self.assertEqual(observation.output_bytes, 96)
        self.assertEqual(observation.copy_workspace_bytes, 128)
        self.assertEqual(observation.copy_allocation_counters, ((current, 0, current), (current + 128, 0, current + 128)))

    def test_missing_copy_counters_skip_allocation(self):
        current = 512 * 2**20
        budget = {'allocated_bytes.all.current': current, 'active_bytes.all.current': current,
                  'reserved_bytes.all.current': 3 * 2**30, 'inactive_split_bytes.all.current': 0}
        observation = ChildBoundaryObservation(SimpleNamespace(_current_memory_limit_mb=lambda: 4096), PARENT)
        with patch.object(torch.cuda, 'current_stream'), \
                patch.object(torch.cuda, 'mem_get_info', return_value=(4 * 2**30, 8 * 2**30)), \
                patch.object(torch.cuda, 'memory_allocated', return_value=current), \
                patch.object(torch.cuda, 'memory_stats', side_effect=[budget, {}]), \
                patch.object(torch, 'empty_like', side_effect=AssertionError('unmeasured allocation')) as allocation:
            observation.probe_copy(torch.ones(2, 12), ())
        self.assertEqual(allocation.call_count, 0)
        self.assertFalse(observation.copy_probe_budget['accepted'])
        self.assertEqual(observation.copy_probe_budget['reason'], 'missing_allocation_counters')

    def test_complete_rejects_incomplete_or_undersized_workspace(self):
        output = torch.ones(2, 12)
        for activation, copy in ((None, 96), (96, None), (0, 96), (95, 96), (96, 95)):
            with self.subTest(activation=activation, copy=copy):
                observation = ChildBoundaryObservation(None, PARENT)
                observation.output_key = _storage_key(output)
                observation.output_bytes = output.untyped_storage().nbytes()
                observation.outside_storages[observation.output_key] = observation.output_bytes
                observation.activation_events = observation.copy_events = (Event('timer', []), None)
                observation.activation_workspace_bytes = activation
                observation.copy_workspace_bytes = copy
                self.assertIsNone(observation.complete())
