import unittest
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from mindspeed.core.pipeline_parallel.adaptive_offload.activation_transfer_scheduler import (
    ActivationTransferScheduler, ByteCredits, ModuleTransfer,
)
from mindspeed.core.pipeline_parallel.adaptive_offload.unified_offload_optimizer import (
    GroupProfile, MemoryBudget, ModuleProfile, ScheduleConfig, UnifiedOffloadOptimizer,
)


class Completion:
    def __init__(self, ready=False):
        self.ready = ready
        self.wait_count = 0

    def query(self):
        return self.ready

    def synchronize(self):
        self.wait_count += 1
        self.ready = True


class TestByteCredits(unittest.TestCase):
    def test_sources_live_until_actual_completion(self):
        budget = ByteCredits(100)
        event = Completion()
        budget.reserve('first', 60, event)
        self.assertFalse(budget.reserve('second', 50))
        self.assertEqual(budget.used, 60)
        event.ready = True
        self.assertTrue(budget.reserve('second', 50))
        self.assertEqual(budget.used, 50)

    def test_backpressure_waits_only_required_event(self):
        budget = ByteCredits(100)
        first, second = Completion(), Completion()
        budget.reserve('first', 40, first)
        budget.reserve('second', 40, second)
        self.assertTrue(budget.reserve('third', 50, wait=True))
        self.assertEqual(first.wait_count, 1)
        self.assertEqual(second.wait_count, 0)
        self.assertEqual(budget.used, 90)
        self.assertLessEqual(budget.peak, budget.limit)

    def test_prefetch_reserves_space_for_demand(self):
        budget = ByteCredits(100)
        budget.reserve('future', 40, reserve=50)
        self.assertFalse(budget.reserve('later', 20, reserve=50))
        self.assertTrue(budget.reserve('demand', 50, wait=True))

    def test_unconsumed_prefetch_cannot_be_freed_by_copy_completion(self):
        budget = ByteCredits(100)
        owner = object()
        budget.reserve(owner, 100)
        self.assertFalse(budget.reserve(object(), 1, wait=True))
        consumed = Completion()
        budget.release_after(owner, consumed)
        self.assertTrue(budget.reserve(object(), 1, wait=True))
        self.assertEqual(consumed.wait_count, 1)

    def test_oversized_transfer_fails_closed(self):
        with self.assertRaises(RuntimeError):
            ByteCredits(10).reserve(object(), 11)

    def test_invalid_reservations(self):
        with self.assertRaises(ValueError):
            ByteCredits(-1)
        with self.assertRaises(ValueError):
            ByteCredits(10).reserve(object(), -1)


class TestTransportCostModel(unittest.TestCase):
    def solver(self, schedule):
        groups = {'cover': GroupProfile(10, 12, 20, 6, 6),
                  'transfer': GroupProfile(20, 1, 2, 6, 6)}
        modules = {name: ModuleProfile(8, 1, 20) for name in groups}
        return UnifiedOffloadOptimizer(groups, modules, MemoryBudget(100, 10000, 0, activation_margin=1), schedule,
                                       module_groups={name: (name,) for name in groups}, nested_groups={})

    def schedule(self, **kwargs):
        return ScheduleConfig(layers=(('cover', 'transfer'),), inflight_microbatches=1,
                              cross_layer_prefetch=False, baseline_recompute=True,
                              recompute_factor=1, transfer_factor=1, **kwargs)

    def test_keep_boundary_can_hide_future_reload(self):
        schedule = self.schedule(adjacent_prefetch=1)
        decisions = {'cover': 'KEEP', 'transfer': 'OFFLOAD'}
        previous = self.solver(schedule).evaluate(decisions)
        unified = self.solver(replace(schedule, unified_transport=True)).evaluate(decisions)
        self.assertEqual(previous.reload_stall_ms, 6)
        self.assertEqual(unified.reload_stall_ms, 0)

    def test_recompute_boundary_can_hide_future_reload(self):
        schedule = self.schedule(adjacent_prefetch=1, unified_transport=True)
        plan = self.solver(schedule).evaluate({'cover': 'RECOMPUTE', 'transfer': 'OFFLOAD'})
        self.assertEqual(plan.reload_stall_ms, 0)
        self.assertEqual(plan.recompute_ms, 8)

    def test_source_and_prefetch_storage_counted_separately(self):
        schedule = self.schedule(adjacent_prefetch=1, unified_transport=True, d2h_slots=2)
        plan = self.solver(schedule).evaluate({'cover': 'OFFLOAD', 'transfer': 'OFFLOAD'})
        self.assertEqual(plan.predicted_peak_mb, 180)

    def test_more_source_credit_cannot_increase_modeled_copy_stall(self):
        schedule = self.schedule(adjacent_prefetch=0, unified_transport=True)
        decisions = {'cover': 'OFFLOAD', 'transfer': 'OFFLOAD'}
        small = self.solver(schedule).evaluate(decisions)
        large = self.solver(replace(schedule, d2h_slots=2)).evaluate(decisions)
        self.assertLessEqual(large.offload_tail_ms, small.offload_tail_ms)

    def test_invalid_source_credit_configuration(self):
        with self.assertRaises(ValueError):
            self.solver(self.schedule(d2h_slots=0))


class TestPrefetchDeadline(unittest.TestCase):
    def scheduler(self):
        scheduler = ActivationTransferScheduler()
        chunk = object()
        ticket = ModuleTransfer(chunk, 'target', 'OFFLOAD', 0, size=10)
        scheduler.tickets = {chunk: [ticket]}
        scheduler.depth = 1
        scheduler.counts = Counter()
        scheduler.stats = lambda key: SimpleNamespace(h2d_time_ms=5, h2d_sample_count=1, backward_compute_time_ms=2)
        scheduler.issue = Mock(return_value=True)
        return scheduler, chunk

    def test_long_compute_window_must_issue_before_next_callback(self):
        scheduler, chunk = self.scheduler()
        scheduler.prefetch(chunk, 20, 'backward_KEEP', chunk, next_boundary=20)
        scheduler.issue.assert_called_once()

    def test_far_future_transfer_waits_for_later_module_boundary(self):
        scheduler, chunk = self.scheduler()
        scheduler.prefetch(chunk, 40, 'pp_forward', chunk, next_boundary=10)
        scheduler.issue.assert_not_called()


class TestDemandAdmission(unittest.TestCase):
    def scheduler(self, current_started):
        scheduler = ActivationTransferScheduler()
        chunk = SimpleNamespace(bulk_reload_group=Mock())
        scheduler.manager = SimpleNamespace(h2d_stream=SimpleNamespace(wait_event=Mock()))
        scheduler.h2d = ByteCredits(200)
        scheduler.largest = 100
        scheduler.counts = Counter()
        scheduler.event = Mock(return_value=Completion())
        scheduler.trace = Mock()
        current = ModuleTransfer(chunk, 'current', 'OFFLOAD', 0, size=100,
                                 backward_started=current_started)
        scheduler.h2d.reserve(current, 100)
        target = ModuleTransfer(chunk, 'target', 'OFFLOAD', 1, size=100, d2h_done=Completion())
        return scheduler, target

    def test_current_demand_is_not_reserved_twice(self):
        scheduler, target = self.scheduler(True)
        self.assertTrue(scheduler.issue(target, 'backward_OFFLOAD'))
        self.assertEqual(scheduler.h2d.used, 200)
        self.assertEqual(scheduler.counts['h2d_backward_OFFLOAD'], 1)

    def test_future_only_residency_preserves_demand_headroom(self):
        scheduler, target = self.scheduler(False)
        self.assertFalse(scheduler.issue(target, 'pp_forward'))
        self.assertEqual(scheduler.h2d.used, 100)
