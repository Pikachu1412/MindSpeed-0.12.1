import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mindspeed.core.pipeline_parallel.adaptive_offload import bandwidth_calibration as calibration
from mindspeed.core.pipeline_parallel.adaptive_offload import auto_activation_memory as automatic
from mindspeed.core.pipeline_parallel.adaptive_offload.adaptive_memory_profiler import AdaptiveMemoryProfiler, PCIeBandwidthStats


class SharedCollectives:
    def __init__(self, size):
        self.size = size
        self.sync = threading.Barrier(size, timeout=5)
        self.values = [None] * size
        self.reductions = [None] * size
        self.barriers = [0] * size
        self.gathers = [0] * size
        self.reduce_calls = [0] * size

    def adapter(self, rank):
        def gather(outputs, value):
            self.gathers[rank] += 1
            self.values[rank] = value.clone()
            self.sync.wait()
            for destination, source in zip(outputs, self.values):
                destination.copy_(source)
            self.sync.wait()

        def barrier():
            self.barriers[rank] += 1
            self.sync.wait()

        def reduce(value, op):
            assert op == 'MAX'
            self.reduce_calls[rank] += 1
            self.reductions[rank] = value.item()
            self.sync.wait()
            value.fill_(max(self.reductions))
            self.sync.wait()

        distributed = SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: rank,
            get_world_size=lambda: self.size, all_gather=gather, barrier=barrier,
            all_reduce=reduce, ReduceOp=SimpleNamespace(MAX='MAX'))
        return SimpleNamespace(distributed=distributed, cuda=SimpleNamespace(is_available=lambda: False),
            tensor=torch.tensor, empty_like=torch.empty_like, int32=torch.int32, int64=torch.int64)


class ScheduleTests(unittest.TestCase):
    def test_one_host(self):
        self.assertEqual(calibration.calibration_schedule([(1, 2)] * 4, [True] * 4), ((0,), (1,), (2,), (3,)))

    def test_multiple_hosts(self):
        self.assertEqual(calibration.calibration_schedule([(1, 2), (1, 2), (3, 4), (3, 4)], [True] * 4), ((0, 2), (1, 3)))

    def test_uneven_hosts(self):
        self.assertEqual(calibration.calibration_schedule([(1, 2)] * 3 + [(3, 4)], [True] * 4), ((0, 3), (1,), (2,)))

    def test_disabled_ranks(self):
        self.assertEqual(calibration.calibration_schedule([(1, 2)] * 4, [True, False, False, True]), ((0,), (3,)))

    def test_all_disabled(self):
        self.assertEqual(calibration.calibration_schedule([(1, 2)] * 4, [False] * 4), ())

    def test_unknown_identity_uses_global_order(self):
        self.assertEqual(calibration.calibration_schedule([(1, 2), (0, 0), (3, 4)], [True] * 3), ((0,), (1,), (2,)))

    def test_unknown_disabled_identity_is_conservative(self):
        self.assertEqual(calibration.calibration_schedule([(1, 2), (0, 0), (3, 4)], [True, False, True]), ((0,), (2,)))

    def test_invalid_participants(self):
        for identities, flags in (([], []), ([(1, 2)], []), ([(1, 2)], [1]), ([(-1, 2)], [True]),
                                  ([(1,)], [True]), ([(True, 1)], [True]), ([(1 << 63, 2)], [True])):
            with self.subTest(identities=identities, flags=flags), self.assertRaises(ValueError):
                calibration.calibration_schedule(identities, flags)

    def test_kernel_identity_is_stable(self):
        with patch.object(Path, 'read_text', return_value='11111111-1111-4111-8111-111111111111'):
            identity = calibration._node_identity()
            self.assertEqual(identity, calibration._node_identity())
            self.assertNotEqual(identity, (0, 0))

    def test_kernel_identity_missing(self):
        with patch.object(Path, 'read_text', side_effect=PermissionError):
            self.assertEqual(calibration._node_identity(), (0, 0))

    def test_kernel_identity_invalid(self):
        with patch.object(Path, 'read_text', return_value='invalid'):
            self.assertEqual(calibration._node_identity(), (0, 0))


class CollectiveTests(unittest.TestCase):
    def run_ranks(self, identities, enabled, failing_rank=None):
        size = len(identities)
        group = SharedCollectives(size)
        local = threading.local()
        results = [None] * size
        failures = [None] * size
        intervals = []
        lock = threading.Lock()

        def run(rank):
            local.rank = rank

            def measure():
                started = time.monotonic_ns()
                time.sleep(0.01)
                with lock:
                    intervals.append((rank, started, time.monotonic_ns()))
                if failing_rank == rank:
                    raise ValueError('injected measurement failure')
                return rank + 100

            try:
                results[rank] = calibration.coordinate_bandwidth_calibration(measure, enabled[rank], group.adapter(rank))
            except BaseException as exception:
                failures[rank] = exception

        workers = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(size)]
        with patch.object(calibration, '_node_identity', side_effect=lambda: identities[local.rank]):
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=10)
            alive = [worker for worker in workers if worker.is_alive()]
            if alive:
                group.sync.abort()
                for worker in alive:
                    worker.join(timeout=2)
            self.assertFalse(alive, 'Collective test failed to terminate')
        return results, failures, group, sorted(intervals, key=lambda item: item[1])

    def test_real_callback_intervals_serialized_per_host(self):
        results, failures, group, intervals = self.run_ranks([(1, 2)] * 4, [True] * 4)
        self.assertEqual(failures, [None] * 4)
        self.assertEqual([item[0] for item in intervals], list(range(4)))
        self.assertTrue(all(left[2] <= right[1] for left, right in zip(intervals, intervals[1:])))
        self.assertEqual(group.barriers, [5] * 4)
        self.assertEqual(group.reduce_calls, [1] * 4)
        self.assertEqual([result[0] for result in results], list(range(100, 104)))

    def test_different_hosts_share_slots(self):
        results, failures, group, intervals = self.run_ranks([(1, 2), (1, 2), (3, 4), (3, 4)], [True] * 4)
        self.assertEqual(failures, [None] * 4)
        self.assertTrue(all(result[1]['phases'] == [[0, 2], [1, 3]] for result in results))
        self.assertEqual(group.barriers, [3] * 4)
        self.assertEqual(len(intervals), 4)

    def test_disabled_ranks_still_participate(self):
        results, failures, group, intervals = self.run_ranks([(1, 2)] * 4, [False, True, False, True])
        self.assertEqual(failures, [None] * 4)
        self.assertEqual([item[0] for item in intervals], [1, 3])
        self.assertEqual(group.gathers, [1] * 4)
        self.assertEqual(group.barriers, [3] * 4)
        self.assertIsNone(results[0][0])
        self.assertIsNone(results[0][1]['call_started_ns'])

    def test_all_disabled_avoids_barriers(self):
        results, failures, group, intervals = self.run_ranks([(1, 2)] * 4, [False] * 4)
        self.assertEqual(failures, [None] * 4)
        self.assertEqual(group.gathers, [1] * 4)
        self.assertEqual(group.barriers, [0] * 4)
        self.assertEqual(group.reduce_calls, [0] * 4)
        self.assertFalse(intervals)
        self.assertTrue(all(result[1]['phases'] == [] for result in results))

    def test_callback_failure_reaches_every_rank(self):
        results, failures, group, intervals = self.run_ranks([(1, 2)] * 4, [True] * 4, failing_rank=1)
        self.assertTrue(all(isinstance(failure, RuntimeError) for failure in failures))
        self.assertTrue(all('callback failed' in str(failure) for failure in failures))
        self.assertEqual(len(intervals), 4)
        self.assertEqual(group.reduce_calls, [1] * 4)
        self.assertIsInstance(failures[1].__cause__, ValueError)

    def test_missing_host_identity_does_not_deadlock(self):
        results, failures, group, intervals = self.run_ranks([(1, 2), (0, 0), (3, 4)], [True] * 3)
        self.assertEqual(failures, [None] * 3)
        self.assertEqual([item[0] for item in intervals], [0, 1, 2])
        self.assertTrue(all(result[1]['coordination'] == 'global_fallback' for result in results))

    def test_single_process_does_not_create_tensors(self):
        adapter = SimpleNamespace(distributed=SimpleNamespace(is_initialized=lambda: False), tensor=Mock())
        callback = Mock(return_value='sample')
        result, metadata = calibration.coordinate_bandwidth_calibration(callback, True, adapter)
        self.assertEqual(result, 'sample')
        self.assertEqual(metadata['phases'], [[0]])
        callback.assert_called_once_with()
        adapter.tensor.assert_not_called()

    def test_single_disabled_process_does_not_measure(self):
        adapter = SimpleNamespace(distributed=SimpleNamespace(is_initialized=lambda: False))
        callback = Mock()
        result, metadata = calibration.coordinate_bandwidth_calibration(callback, False, adapter)
        self.assertIsNone(result)
        self.assertFalse(metadata['phases'])
        callback.assert_not_called()


class ProfilerIntegrationTests(unittest.TestCase):
    def profiler(self):
        profiler = AdaptiveMemoryProfiler()
        profiler._capture_memory_peak = Mock()
        profiler._sample_memory_capacity = Mock()
        profiler._refresh_is_profile_rank = Mock()
        profiler._is_profile_rank = True
        profiler._is_stall_profile_rank = True
        profiler._pp_rank_initialized = True
        profiler._measure_pcie = True
        profiler._auto_calibration_iteration = 5
        profiler._warmup_skip_iters = 5
        return profiler

    def test_only_original_first_profile_iteration_coordinates(self):
        profiler = self.profiler()
        sample = PCIeBandwidthStats(d2h_gbps=20, h2d_gbps=21, measured=True)
        coordinator = Mock(return_value=(sample, {'phases': [[0]]}))
        with patch.object(calibration, 'coordinate_bandwidth_calibration', coordinator), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                patch.object(profiler, '_measure_pcie_bandwidth_safe') as legacy:
            profiler.on_iteration_start(4)
            coordinator.assert_not_called()
            profiler.on_iteration_start(5)
            profiler.on_iteration_start(6)
        coordinator.assert_called_once()
        self.assertTrue(coordinator.call_args.args[1])
        self.assertIs(profiler._pcie_stats, sample)
        self.assertIsNone(profiler._auto_calibration_iteration)
        self.assertTrue(profiler._pcie_measured)
        self.assertEqual(profiler._auto_calibration_telemetry['iteration'], 5)
        legacy.assert_not_called()

    def test_profile_done_still_participates_without_sampling(self):
        profiler = self.profiler()
        profiler._profiling_done = True
        coordinator = Mock(return_value=(None, {'phases': []}))
        with patch.object(calibration, 'coordinate_bandwidth_calibration', coordinator):
            profiler.on_iteration_start(5)
        coordinator.assert_called_once()
        self.assertFalse(coordinator.call_args.args[1])

    def test_disabled_measurement_still_participates(self):
        profiler = self.profiler()
        profiler._measure_pcie = False
        coordinator = Mock(return_value=(None, {'phases': []}))
        with patch.object(calibration, 'coordinate_bandwidth_calibration', coordinator):
            profiler.on_iteration_start(5)
        self.assertFalse(coordinator.call_args.args[1])
        self.assertFalse(profiler._pcie_measured)

    def test_warm_cache_no_coordination(self):
        profiler = self.profiler()
        profiler._auto_calibration_iteration = None
        profiler._profiling_done = True
        with patch.object(calibration, 'coordinate_bandwidth_calibration') as coordinator:
            profiler.on_iteration_start(5)
            profiler.on_iteration_start(6)
        coordinator.assert_not_called()


class CacheSchedulingTests(unittest.TestCase):
    def run_cache(self, available, remote_miss=False):
        profiler = SimpleNamespace(_init_pp_rank_profiling=Mock(), _auto_calibration_iteration=999,
                                   _auto_module_specs={}, _sample_memory_capacity=Mock())
        args = SimpleNamespace(iteration=7, train_iters=80)
        value = {'profile': {}, 'memory': {'capacity_bytes': 100}}
        decoded = (value, {}, {}, SimpleNamespace(measured=True), {}, {'samples': [1, 2]})
        original_tensor = torch.tensor
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            if available:
                path = root / '.mindspeed/activation_profiles/signature.rank0.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'cache_fixture': True}))
            with patch.object(automatic, 'enabled', return_value=True), \
                    patch.object(automatic, 'profile_fingerprint', return_value='signature'), \
                    patch.object(automatic, '_decode_profile_cache', return_value=decoded), \
                    patch.object(automatic.Path, 'cwd', return_value=root), \
                    patch.object(torch.distributed, 'is_initialized', return_value=remote_miss), \
                    patch.object(torch.distributed, 'get_rank', return_value=0), \
                    patch.object(torch.distributed, 'all_reduce', side_effect=lambda value, op: value.fill_(0)), \
                    patch.object(torch, 'tensor', side_effect=lambda value, **kwargs: original_tensor(value, dtype=kwargs['dtype'])):
                automatic.initialize_profile_cache(profiler, [], args)
        return profiler

    def test_cold_cache_uses_resumed_iteration_anchor(self):
        profiler = self.run_cache(False)
        self.assertEqual(profiler._auto_calibration_iteration, 9)
        self.assertEqual(profiler._warmup_skip_iters, 9)

    def test_valid_warm_cache_clears_pending_calibration(self):
        profiler = self.run_cache(True)
        self.assertIsNone(profiler._auto_calibration_iteration)
        self.assertTrue(profiler._auto_cache_loaded)
        self.assertTrue(profiler._profiling_done)

    def test_peer_cache_miss_coordinates_local_warm_rank(self):
        profiler = self.run_cache(True, remote_miss=True)
        self.assertEqual(profiler._auto_calibration_iteration, 9)
        self.assertFalse(getattr(profiler, '_auto_cache_loaded', False))


if __name__ == '__main__':
    unittest.main()
