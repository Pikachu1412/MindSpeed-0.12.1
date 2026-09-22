import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mindspeed.core.pipeline_parallel.adaptive_offload.adaptive_memory_profiler import measure_pcie_bandwidth


class TestProfilerRng(unittest.TestCase):
    def calibration_fixture(self, failure=None):
        generator = torch.Generator(device="cpu").manual_seed(321)
        original_randn, original_empty = torch.randn, torch.empty
        devices = []

        def get_state(device):
            devices.append(device)
            return generator.get_state()

        def set_state(state, device):
            self.assertEqual(device, 3)
            generator.set_state(state)

        def random_tensor(*args, **kwargs):
            self.assertEqual(kwargs.pop("device"), "cuda")
            original_randn(3)
            result = original_randn(*args, generator=generator, **kwargs)
            if failure == "random_allocation":
                raise RuntimeError("intentional random allocation failure")
            return result

        def empty_tensor(*args, **kwargs):
            self.assertTrue(kwargs.pop("pin_memory"))
            if failure == "host_allocation":
                raise RuntimeError("intentional host allocation failure")
            return original_empty(*args, **kwargs)

        def event(**kwargs):
            self.assertEqual(kwargs, {"enable_timing": True})
            record = Mock(side_effect=RuntimeError("intentional transfer failure")) if failure == "transfer" else Mock()
            return SimpleNamespace(record=record, elapsed_time=lambda other: 1.0)

        backend = SimpleNamespace(is_available=lambda: True, current_device=lambda: 3,
                                  get_rng_state=get_state, set_rng_state=set_state,
                                  Stream=object, Event=event, synchronize=Mock(),
                                  stream=lambda stream: contextlib.nullcontext())
        return generator, backend, random_tensor, empty_tensor, devices

    def run_calibration(self, failure=None):
        generator, backend, random_tensor, empty_tensor, devices = self.calibration_fixture(failure)
        cpu_before, device_before = torch.get_rng_state().clone(), generator.get_state().clone()
        with patch.object(torch, "cuda", backend), patch.object(torch, "randn", random_tensor), \
             patch.object(torch, "empty", empty_tensor):
            if failure is None:
                result = measure_pcie_bandwidth(tensor_size_mb=1, num_warmup=1, num_trials=2)
                self.assertTrue(result.measured)
                self.assertAlmostEqual(result.d2h_gbps, 1000 / 1024)
                self.assertEqual(result.d2h_gbps, result.h2d_gbps)
                self.assertEqual(backend.synchronize.call_count, 6)
            else:
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    measure_pcie_bandwidth(tensor_size_mb=1, num_warmup=1, num_trials=2)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_before))
        self.assertTrue(torch.equal(generator.get_state(), device_before))
        self.assertEqual(devices, [3])

    def test_calibration_preserves_cpu_and_current_device_rng(self):
        self.run_calibration()

    def test_calibration_preserves_rng_when_random_allocation_raises(self):
        self.run_calibration("random_allocation")

    def test_calibration_preserves_rng_when_host_allocation_raises(self):
        self.run_calibration("host_allocation")

    def test_calibration_preserves_rng_when_transfer_raises(self):
        self.run_calibration("transfer")

    def test_unavailable_backend_does_not_access_rng_or_allocate(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.cuda, "get_rng_state") as state, patch.object(torch, "randn") as allocate:
            result = measure_pcie_bandwidth()
        self.assertFalse(result.measured)
        state.assert_not_called()
        allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
