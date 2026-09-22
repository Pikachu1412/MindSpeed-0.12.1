import gc
import unittest
import weakref
from types import SimpleNamespace

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from mindspeed.core.megatron_basic.requirements_basic import (
    linear_with_grad_accum_backward_wrapper,
    linear_with_grad_accum_forward_wrapper,
)


def linear_forward(context, inputs, weight):
    context.save_for_backward(inputs, weight)
    return inputs.matmul(weight.t())


def linear_backward(context, gradient):
    inputs, weight = context.saved_tensors
    return gradient.matmul(weight), gradient.t().matmul(inputs)


def run_linear(wrapped, retain_first):
    forward = linear_with_grad_accum_forward_wrapper(linear_forward) if wrapped else linear_forward
    backward = linear_with_grad_accum_backward_wrapper(linear_backward) if wrapped else linear_backward
    function = type('LinearFixture', (torch.autograd.Function,), {
        'forward': staticmethod(forward), 'backward': staticmethod(backward),
    })
    generator = torch.Generator().manual_seed(1234)
    inputs = torch.randn((5, 8), generator=generator, requires_grad=True)
    activation = torch.sigmoid(inputs)
    activation_ref = weakref.ref(activation)
    storage_ref = StorageWeakRef(activation.untyped_storage())
    weight = torch.nn.Parameter(torch.randn((4, 8), generator=generator))
    weight.main_grad = torch.zeros_like(weight)
    output = function.apply(activation, weight)
    gradient = torch.ones_like(output)
    if retain_first:
        output.backward(gradient, retain_graph=True)
    output.backward(gradient)
    input_gradient, weight_gradient = inputs.grad.clone(), weight.grad.clone()
    del output, activation, inputs, weight, gradient
    return input_gradient, weight_gradient, activation_ref() is None, storage_ref.expired()


class LinearContextLifetimeTest(unittest.TestCase):
    def test_proxy_preserves_parameter_identity_and_releases_activation(self):
        settings = gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks)
        metadata = {}

        def backend(context, gradient):
            metadata['weight_id'] = id(context.saved_tensors[1])
            metadata['main_grad_id'] = id(context.saved_tensors[1].main_grad)
            context.visits += 1
            return gradient

        activation = torch.ones((8, 8))
        activation_ref = weakref.ref(activation)
        storage_ref = StorageWeakRef(activation.untyped_storage())
        weight = torch.nn.Parameter(torch.ones((4, 8)))
        weight.main_grad = object()
        identity = id(weight), id(weight.main_grad)
        context = SimpleNamespace(weight=weight, saved_tensors=(activation,), visits=0)
        gradient = object()
        self.assertIs(linear_with_grad_accum_backward_wrapper(backend)(context, gradient), gradient)
        self.assertEqual(context.visits, 1)
        self.assertEqual((metadata['weight_id'], metadata['main_grad_id']), identity)
        del context, activation, weight
        self.assertIsNone(activation_ref())
        self.assertTrue(storage_ref.expired())
        self.assertEqual(settings, (gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks)))

    def check_gradients_and_lifetime(self, retain_first):
        reference = run_linear(False, retain_first)
        candidate = run_linear(True, retain_first)
        self.assertTrue(torch.equal(reference[0], candidate[0]))
        self.assertTrue(torch.equal(reference[1], candidate[1]))
        self.assertEqual(candidate[2:], (True, True))

    def test_gradients_match_without_forced_collection(self):
        self.check_gradients_and_lifetime(False)

    def test_retained_graph_gradients_match(self):
        self.check_gradients_and_lifetime(True)

    def test_backward_exception_is_preserved(self):
        marker = ValueError('sentinel')

        def raises(context, gradient):
            raise marker

        context = SimpleNamespace(weight=object(), saved_tensors=(object(),))
        with self.assertRaises(ValueError) as captured:
            linear_with_grad_accum_backward_wrapper(raises)(context, object())
        self.assertIs(captured.exception, marker)

    def test_unpatched_context_falls_through(self):
        expected = SimpleNamespace()
        gradient = object()

        def fallback(context, value):
            self.assertIs(context, expected)
            self.assertIs(value, gradient)
            return value

        self.assertIs(linear_with_grad_accum_backward_wrapper(fallback)(expected, gradient), gradient)
