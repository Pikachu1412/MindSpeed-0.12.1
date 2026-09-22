import gc
import unittest
import weakref

import torch

from mindspeed.core.pipeline_parallel.adaptive_offload.microbatch_liveness import MicrobatchLiveness


class OutputTensorLifetimeTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.gc_enabled = gc.isenabled()
        gc.disable()
        self.counts = []
        self.tracker = MicrobatchLiveness(self.counts.append)
        self.tracker.begin('sample')

    def tearDown(self):
        if self.gc_enabled:
            gc.enable()
        gc.collect()

    def test_completed_backward_releases_output_without_gc(self):
        tensor = torch.ones(8, requires_grad=True) * 2
        reference = weakref.ref(tensor)
        self.tracker.track('sample', tensor)
        tensor.sum().backward()
        self.assertFalse(self.tracker.live)
        del tensor
        self.assertIsNone(reference())

    def test_abandoned_forward_releases_output_and_liveness_without_gc(self):
        tensor = torch.ones(8, requires_grad=True) * 2
        reference = weakref.ref(tensor)
        self.tracker.track('sample', tensor)
        self.assertEqual(self.tracker.live, {'sample'})
        del tensor
        self.assertIsNone(reference())
        self.assertFalse(self.tracker.live)

    def test_view_keeps_legitimate_output_lifetime(self):
        tensor = torch.ones(8, requires_grad=True) * 2
        reference = weakref.ref(tensor)
        self.tracker.track('sample', tensor)
        view = tensor.view(2, 4)
        del tensor
        self.assertIsNotNone(reference())
        self.assertEqual(self.tracker.live, {'sample'})
        del view
        self.assertIsNone(reference())
        self.assertFalse(self.tracker.live)

    def test_retain_graph_preserves_tracker_until_final_backward(self):
        tensor = torch.ones(8, requires_grad=True) * 2
        reference = weakref.ref(tensor)
        self.tracker.track('sample', tensor)
        tensor.sum().backward(retain_graph=True)
        self.assertEqual(self.tracker.live, {'sample'})
        tensor.sum().backward()
        self.assertFalse(self.tracker.live)
        del tensor
        self.assertIsNone(reference())

    def test_nested_order_and_duplicate_tensor_identity(self):
        first = torch.ones(8, requires_grad=True) * 2
        second = torch.ones(8, requires_grad=True) * 3
        third = torch.ones(8, requires_grad=True) * 4
        structure = {'first': [first, (second, first)], 'second': {'third': third}}
        references = [weakref.ref(tensor) for tensor in (first, second, third)]
        self.assertIs(self.tracker.track('sample', structure), structure)
        self.assertEqual(len(self.tracker.records['sample']['outputs']), 3)
        self.assertEqual([next(iter(tensor._backward_hooks.values())).index for tensor in (first, second, third)], [0, 1, 2])
        (first.sum() + second.sum() + third.sum()).backward()
        del structure, first, second, third
        self.assertTrue(all(reference() is None for reference in references))
        self.assertFalse(self.tracker.live)

    def test_output_container_is_unchanged(self):
        tensor = torch.ones(8, requires_grad=True) * 2
        nested = [tensor, None, {'scalar': 3}]
        output = (nested, 'text')
        self.assertIs(self.tracker.track('sample', output), output)
        self.assertIs(output[0], nested)
        self.assertIs(output[0][0], tensor)
        self.assertEqual(output[1], 'text')
        tensor.sum().backward()
        self.assertFalse(self.tracker.live)

    def test_nondifferentiable_output_is_not_retained(self):
        tensor = torch.ones(8)
        reference = weakref.ref(tensor)
        self.tracker.track('sample', {'output': tensor, 'metadata': None})
        self.assertFalse(self.tracker.live)
        del tensor
        self.assertIsNone(reference())

    def test_unused_second_output_keeps_only_its_legitimate_lifetime(self):
        first = torch.ones(8, requires_grad=True) * 2
        second = torch.ones(8, requires_grad=True) * 3
        first_reference, second_reference = weakref.ref(first), weakref.ref(second)
        self.tracker.track('sample', (first, second))
        first.sum().backward()
        del first
        self.assertIsNone(first_reference())
        self.assertEqual(self.tracker.live, {'sample'})
        del second
        self.assertIsNone(second_reference())
        self.assertFalse(self.tracker.live)


if __name__ == '__main__':
    unittest.main()
