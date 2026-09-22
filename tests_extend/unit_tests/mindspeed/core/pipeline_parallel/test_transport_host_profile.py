import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mindspeed.core.pipeline_parallel.adaptive_offload import activation_transfer_scheduler as runtime
from mindspeed.core.pipeline_parallel.adaptive_offload.transport_host_profile import (
    TransportHostMeter, estimate_host_costs, update_host_costs,
)
from mindspeed.core.pipeline_parallel.adaptive_offload.unified_offload_optimizer import (
    GroupProfile, MemoryBudget, ScheduleConfig, UnifiedOffloadOptimizer,
)


class TestHostMeter(unittest.TestCase):
    def test_nested_regions_exclude_children(self):
        meter = TransportHostMeter(True)
        with patch('time.perf_counter_ns', side_effect=[0, 10, 40, 100]), patch(
            'time.thread_time_ns', side_effect=[0, 5, 15, 50]
        ):
            with meter.region('outer'):
                with meter.region('inner'):
                    pass
        self.assertAlmostEqual(meter.metrics['outer']['wall_ms'], 70 / 1e6)
        self.assertAlmostEqual(meter.metrics['outer']['cpu_ms'], 40 / 1e6)
        self.assertAlmostEqual(meter.metrics['inner']['cpu_ms'], 10 / 1e6)
        self.assertFalse(meter.stack)

    def test_exception_releases_frame(self):
        meter = TransportHostMeter(True)
        with self.assertRaises(ValueError), meter.region('failure'):
            raise ValueError('expected')
        self.assertFalse(meter.stack)
        self.assertEqual(meter.metrics['failure']['calls'], 1)

    def test_native_and_device_wait_are_not_double_charged(self):
        meter = TransportHostMeter(True)
        for name, cost in {'begin': 4, 'issue': 6, 'wait': 100, 'native_reload': 200}.items():
            meter.metrics[name]['cpu_ms'] = cost
        profiler = SimpleNamespace()
        update_host_costs(profiler, 'asynchronous', meter, modules=4, offloads=2, flush_cpu_ms=2)
        sample = profiler._transport_host_costs['asynchronous']
        self.assertEqual(sample['module_cpu_ms'], 1)
        self.assertEqual(sample['offload_cpu_ms'], 4)
        self.assertEqual(sample['sample_count'], 1)

    def test_optional_sampling_cost_is_amortized(self):
        meter = TransportHostMeter(True)
        meter.metrics['optional_event']['cpu_ms'] = 8
        meter.metrics['event']['cpu_ms'] = 2
        profiler = SimpleNamespace()
        update_host_costs(profiler, 'synchronous', meter, modules=0, offloads=1, sampling_divisor=8)
        self.assertEqual(profiler._transport_host_costs['synchronous']['offload_cpu_ms'], 3)

    def test_no_samples_do_not_invent_measurements(self):
        profiler = SimpleNamespace()
        update_host_costs(profiler, 'asynchronous', TransportHostMeter(False), 1, 1)
        update_host_costs(profiler, 'synchronous', TransportHostMeter(True), 1, 0)
        self.assertFalse(hasattr(profiler, '_transport_host_costs'))

    def test_mode_samples_are_independent_and_smoothed(self):
        profiler = SimpleNamespace()
        meter = TransportHostMeter(True)
        meter.metrics['issue']['cpu_ms'] = 4
        update_host_costs(profiler, 'asynchronous', meter, 1, 1)
        meter.metrics['issue']['cpu_ms'] = 8
        update_host_costs(profiler, 'asynchronous', meter, 1, 1)
        update_host_costs(profiler, 'synchronous', meter, 1, 1)
        self.assertEqual(profiler._transport_host_costs['asynchronous']['offload_cpu_ms'], 5)
        self.assertEqual(profiler._transport_host_costs['synchronous']['offload_cpu_ms'], 8)


class TestHostCostEstimates(unittest.TestCase):
    def test_cold_start_is_explicitly_unmeasured(self):
        costs, source, references = estimate_host_costs({}, 'asynchronous', 2, 2)
        self.assertEqual(costs, {'module_cpu_ms': 0.0, 'offload_cpu_ms': 0.0, 'sample_count': 0})
        self.assertEqual(source, 'unmeasured_lower_bound')
        self.assertEqual(references, [])

    def test_exact_measurement_takes_precedence(self):
        measured = {'module_cpu_ms': 1.0, 'offload_cpu_ms': 2.0, 'sample_count': 8}
        profiles = {'asynchronous:2:2': measured,
                    'asynchronous:0:1': {'module_cpu_ms': 10.0, 'offload_cpu_ms': 20.0, 'sample_count': 4}}
        costs, source, references = estimate_host_costs(profiles, 'asynchronous', 2, 2)
        self.assertEqual(costs, measured)
        self.assertIsNot(costs, measured)
        self.assertEqual(source, 'observed_exclusive_cpu')
        self.assertEqual(references, ['asynchronous:2:2'])

    def test_unmeasured_layout_uses_same_mode_maxima_without_mutation(self):
        profiles = {'asynchronous:2:2': {'module_cpu_ms': 1.0, 'offload_cpu_ms': 4.0, 'sample_count': 8},
                    'asynchronous:0:1': {'module_cpu_ms': 2.0, 'offload_cpu_ms': 3.0, 'sample_count': 4}}
        before = {name: dict(values) for name, values in profiles.items()}
        costs, source, references = estimate_host_costs(profiles, 'asynchronous', 1, 2)
        self.assertEqual(costs, {'module_cpu_ms': 2.0, 'offload_cpu_ms': 4.0, 'sample_count': 0})
        self.assertEqual(source, 'same_mode_conservative_estimate')
        self.assertEqual(references, ['asynchronous:0:1', 'asynchronous:2:2'])
        self.assertEqual(profiles, before)

    def test_async_reference_estimates_sync_offload_work(self):
        profiles = {'asynchronous:2:2': {'module_cpu_ms': 1.0, 'offload_cpu_ms': 4.0, 'sample_count': 8}}
        costs, source, references = estimate_host_costs(profiles, 'synchronous', 1, 1)
        self.assertEqual(costs, {'module_cpu_ms': 0.0, 'offload_cpu_ms': 5.0, 'sample_count': 0})
        self.assertEqual(source, 'cross_mode_conservative_estimate')
        self.assertEqual(references, ['asynchronous:2:2'])

    def test_sync_reference_does_not_make_async_callbacks_free(self):
        profiles = {'synchronous:1:1': {'module_cpu_ms': 0.0, 'offload_cpu_ms': 4.0, 'sample_count': 8}}
        costs, source, references = estimate_host_costs(profiles, 'asynchronous', 2, 2)
        self.assertEqual(costs, {'module_cpu_ms': 4.0, 'offload_cpu_ms': 4.0, 'sample_count': 0})
        self.assertEqual(source, 'cross_mode_conservative_estimate')
        self.assertEqual(references, ['synchronous:1:1'])

    def test_zero_sample_entries_are_not_measurements(self):
        profiles = {'asynchronous:2:2': {'module_cpu_ms': 100.0, 'offload_cpu_ms': 100.0, 'sample_count': 0}}
        costs, source, references = estimate_host_costs(profiles, 'asynchronous', 2, 2)
        self.assertEqual(costs['offload_cpu_ms'], 0)
        self.assertEqual(source, 'unmeasured_lower_bound')
        self.assertEqual(references, [])


class TestTransportFastPath(unittest.TestCase):
    def test_no_offload_does_not_allocate_events_or_attach_hooks(self):
        scheduler = runtime.ActivationTransferScheduler()
        profiler = SimpleNamespace(_current_iter=1, is_optimization_applied=lambda: True,
                                   _plan=SimpleNamespace(decisions={'first': 'KEEP', 'second': 'RECOMPUTE'},
                                                         execution={'transport_version': 1, 'adjacent_prefetch': 0,
                                                                    'd2h_slots': 1}))
        with patch.dict(os.environ, {'ADAPTIVE_MEM_PROFILE_JSON': ''}), patch('torch.cuda.Event') as event, patch(
            'torch.cuda.current_stream'
        ) as stream:
            scheduler.start(profiler, SimpleNamespace())
            self.assertFalse(scheduler.active)
            self.assertTrue(scheduler.managed)
            self.assertIsNone(scheduler.begin(object(), 'first', 'KEEP'))
            tensor = Mock()
            self.assertIs(scheduler.attach(None, tensor), tensor)
            tensor.register_hook.assert_not_called()
            scheduler.finish()
            event.assert_not_called()
            stream.assert_not_called()
            self.assertFalse(scheduler.managed)

    def test_credit_queries_are_deferred_until_pressure(self):
        budget = runtime.ByteCredits(100)
        completion = Mock()
        completion.query.return_value = True
        budget.reserve(object(), 20, completion)
        budget.reserve(object(), 20)
        completion.query.assert_not_called()
        self.assertTrue(budget.ensure(70))
        completion.query.assert_called_once()
        self.assertEqual(budget.used, 20)

    def test_synchronous_boundary_starts_reload_without_output_hook(self):
        scheduler = runtime.ActivationTransferScheduler()
        handler = object()
        ticket = runtime.ModuleTransfer(handler, 'module', 'OFFLOAD', 0, group_id=7)
        scheduler.active = True
        scheduler.mode = 'synchronous'
        scheduler.audit = False
        scheduler.groups[(handler, 7)] = ticket
        scheduler.backward_begin = Mock()
        original = Mock()
        with patch.object(runtime, '_scheduler', scheduler):
            runtime.group_backward_begin_wrapper(original)(handler, 'module', 7)
        scheduler.backward_begin.assert_called_once_with(ticket)
        original.assert_called_once_with(handler, 'module', 7)


class TestHostCostModel(unittest.TestCase):
    def evaluate(self, schedule, action='OFFLOAD'):
        return UnifiedOffloadOptimizer(
            {'module': GroupProfile(10, 5, 5, 2, 2)}, {}, MemoryBudget(100, 1000, 0), schedule,
            module_groups={'module': ('module',)}, nested_groups={},
        ).evaluate({'module': action})

    def schedule(self, **kwargs):
        return ScheduleConfig(layers=(('module',),), inflight_microbatches=1,
                              unified_transport=True, baseline_recompute=True,
                              adjacent_prefetch=0, cross_layer_prefetch=False, **kwargs)

    def test_offload_cost_includes_measured_host_work(self):
        plan = self.evaluate(self.schedule(host_module_ms=3, host_offload_ms=4))
        self.assertEqual(plan.host_overhead_ms, 7)
        self.assertAlmostEqual(plan.predicted_overhead_ms,
                               plan.recompute_ms + plan.reload_stall_ms + plan.offload_tail_ms + 7)

    def test_keep_fast_path_has_no_transport_cost(self):
        plan = self.evaluate(self.schedule(host_module_ms=3, host_offload_ms=4), 'KEEP')
        self.assertEqual(plan.host_overhead_ms, 0)

    def test_sync_skips_module_callback_cost(self):
        plan = self.evaluate(self.schedule(synchronous_d2h=True, host_module_ms=3, host_offload_ms=4))
        self.assertEqual(plan.host_overhead_ms, 4)

    def test_host_cost_can_reverse_transport_preference(self):
        asynchronous = self.evaluate(self.schedule(host_module_ms=10, host_offload_ms=1))
        synchronous = self.evaluate(self.schedule(synchronous_d2h=True, host_offload_ms=1))
        self.assertLess(synchronous.predicted_overhead_ms, asynchronous.predicted_overhead_ms)

    def test_invalid_host_cost_rejected(self):
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(self.schedule(host_offload_ms=value))
