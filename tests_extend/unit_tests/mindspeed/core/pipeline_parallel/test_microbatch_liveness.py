import gc
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from mindspeed.core.pipeline_parallel.adaptive_offload.microbatch_liveness import MicrobatchLiveness


class TestMicrobatchLiveness(unittest.TestCase):
    def setUp(self):
        self.observations = []
        self.tracker = MicrobatchLiveness(self.observations.append)

    def attach(self, output):
        token = object()
        self.tracker.begin(token)
        self.assertIs(self.tracker.track(token, output), output)
        return token

    def test_token_remains_live_during_parameter_backward(self):
        tensor = torch.randn(8, requires_grad=True)
        weight = torch.randn(8, requires_grad=True)
        output = (tensor * weight).sin()
        token = self.attach(output)
        seen = []
        weight.register_hook(lambda gradient: seen.append(token in self.tracker.live))
        output.sum().backward()
        self.assertEqual(seen, [True])
        self.assertFalse(self.tracker.live)
        self.assertEqual(max(self.observations), 1)
        self.assertEqual(self.tracker.snapshot()['backward_tasks_completed'], 1)

    def test_fifo_microbatches_count_current_once(self):
        outputs = [torch.randn(8, requires_grad=True).sin() for _ in range(3)]
        tokens = [self.attach(output) for output in outputs]
        self.assertEqual(self.tracker.live, set(tokens))
        for index, output in enumerate(outputs):
            output.sum().backward()
            self.assertEqual(len(self.tracker.live), 2 - index)
        self.assertEqual(max(self.observations), 3)

    def test_deleting_python_output_does_not_release_a_live_graph(self):
        tensor = torch.randn(8, requires_grad=True)
        output = tensor.sin()
        token = self.attach(output)
        loss = output.square().sum()
        del output
        gc.collect()
        self.assertIn(token, self.tracker.live)
        loss.backward()
        self.assertFalse(self.tracker.live)

    def test_unused_output_stays_charged_until_its_node_is_destroyed(self):
        tensor = torch.randn(8, requires_grad=True)
        unused = tensor.cos()
        used = tensor.sin()
        output = {'unused': unused, 'used': used}
        token = self.attach(output)
        used.sum().backward()
        self.assertIn(token, self.tracker.live)
        self.assertEqual(self.tracker.snapshot()['retained_after_backward'], 1)
        del output, unused
        gc.collect()
        self.assertFalse(self.tracker.live)

    def test_multiple_and_duplicate_outputs_share_one_completion(self):
        tensor = torch.randn(8, requires_grad=True)
        first, second = tensor.sin(), tensor.cos()
        self.attach((first, second, first))
        (first.sum() + second.sum()).backward()
        self.assertFalse(self.tracker.live)
        self.assertEqual(self.tracker.snapshot()['backward_tasks_completed'], 1)

    def test_retain_graph_stays_charged_until_final_backward(self):
        tensor = torch.randn(8, requires_grad=True)
        output = tensor.sin()
        token = self.attach(output)
        loss = output.sum()
        loss.backward(retain_graph=True)
        self.assertIn(token, self.tracker.live)
        self.assertEqual(self.tracker.snapshot()['retained_after_backward'], 1)
        loss.backward()
        self.assertFalse(self.tracker.live)
        torch.testing.assert_close(tensor.grad, 2 * tensor.cos())

    def test_discarded_graph_releases_without_backward(self):
        output = torch.randn(8, requires_grad=True).sin()
        self.attach(output)
        del output
        gc.collect()
        self.assertFalse(self.tracker.live)

    def test_iteration_boundary_does_not_erase_retained_graph(self):
        output = torch.randn(8, requires_grad=True).sin()
        token = self.attach(output)
        output.sum().backward(retain_graph=True)
        self.tracker.start_iteration()
        self.assertIn(token, self.tracker.live)
        self.assertEqual(self.tracker.snapshot()['carried_graphs'], 1)
        output.sum().backward()
        self.assertFalse(self.tracker.live)

    def test_reentrant_checkpoint_preserves_outer_lifetime(self):
        tensor = torch.randn(8, requires_grad=True)
        token = object()
        self.tracker.begin(token)
        replay_seen = []

        def forward(value):
            if torch.is_grad_enabled():
                replay_seen.append(token in self.tracker.live)
            return value.sin()

        output = checkpoint(forward, tensor, use_reentrant=True)
        self.tracker.track(token, output)
        output.sum().backward()
        self.assertEqual(replay_seen, [True])
        self.assertFalse(self.tracker.live)
        torch.testing.assert_close(tensor.grad, tensor.cos())

    def test_no_gradient_or_failed_forward_releases_token(self):
        for output in (None, {'value': torch.ones(8)}, (1, 'value')):
            with self.subTest(output=output):
                self.attach(output)
                self.assertFalse(self.tracker.live)

    def test_unfinished_backward_is_not_reset_at_new_iteration(self):
        output = torch.randn(8, requires_grad=True).sin()
        token = self.attach(output)
        callbacks = []
        with patch.object(torch._C, '_current_graph_task_id', return_value=71), \
                patch.object(torch._C._autograd, '_get_current_graph_task_keep_graph', return_value=False), \
                patch.object(torch.autograd.Variable, '_execution_engine', SimpleNamespace(queue_callback=callbacks.append)):
            self.tracker.backward_started(token, 0)
        with self.assertRaisesRegex(RuntimeError, 'unfinished autograd'):
            self.tracker.start_iteration()
        callbacks[0]()
        self.assertFalse(self.tracker.live)

    def test_nested_backward_retires_inner_before_outer(self):
        tracker = self.tracker
        outer_token = object()
        observations = []

        class NestedBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor):
                return tensor.clone()

            @staticmethod
            def backward(ctx, gradient):
                with torch.enable_grad():
                    inner_token = object()
                    tracker.begin(inner_token)
                    inner = torch.ones(8, requires_grad=True).sin()
                    tracker.track(inner_token, inner)
                    observations.append((outer_token in tracker.live, inner_token in tracker.live))
                    inner.sum().backward()
                    observations.append((outer_token in tracker.live, inner_token in tracker.live))
                return gradient

        self.tracker.begin(outer_token)
        output = NestedBackward.apply(torch.ones(8, requires_grad=True))
        self.tracker.track(outer_token, output)
        output.sum().backward()
        self.assertEqual(observations, [(True, True), (True, False)])
        self.assertEqual(self.tracker.snapshot()['peak_live'], 2)
        self.assertFalse(self.tracker.live)

    def test_backward_error_cannot_masquerade_as_completion(self):
        class BrokenBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor):
                return tensor.clone()

            @staticmethod
            def backward(ctx, gradient):
                raise RuntimeError('intentional backward failure')

        output = BrokenBackward.apply(torch.ones(8, requires_grad=True))
        token = self.attach(output)
        with self.assertRaisesRegex(RuntimeError, 'intentional backward failure'):
            output.sum().backward()
        self.assertIn(token, self.tracker.live)
        with self.assertRaisesRegex(RuntimeError, 'unfinished autograd'):
            self.tracker.start_iteration()
