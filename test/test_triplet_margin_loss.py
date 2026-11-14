"""Triplet margin loss across the reduction modes and both distance options.

The loss joins three operands, measures the anchor-to-positive and
anchor-to-negative distances along the last axis, and answers
``max(0, d(a, p) - d(a, n) + margin)`` per position.  Coverage here:

* the native op with every reduction mode, order, eps and the swap option;
* the module path, which composes the same formula from primitive ops;
* parameter validation (dimension count, reduction enum, margin bound);
* shape inference on the meta device;
* gradients for all three differentiable operands.
"""
import os
import sys
import unittest

import torch
import torch.nn.functional as torch_F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.testing._internal.reference import assert_reference_close, from_reference, reference_devices


def _reduction_enum(reduction):
    return {"none": 0, "mean": 1, "sum": 2}[reduction]


def _operands(shape, seed, dtype=torch.float32):
    """Three reference operands drawn from one seed."""
    torch.manual_seed(seed)
    return [torch.randn(*shape, dtype=dtype) for _ in range(3)]


def _triple(shape, seed, device, dtype, requires_grad=False):
    return [from_reference(t, device, requires_grad=requires_grad)
            for t in _operands(shape, seed, dtype)]


class TestNativeForward(unittest.TestCase):
    """The native op answers the same value as the composed form."""

    def _run(self, shape, reduction, margin, p, eps, swap, dev, dtype, seed):
        ref_operands = _operands(shape, seed, dtype)
        ref = torch.triplet_margin_loss(
            *ref_operands, margin=margin, p=p, eps=eps, swap=swap,
            reduction=_reduction_enum(reduction))
        got = tp.triplet_margin_loss(
            *[from_reference(t, dev) for t in ref_operands], margin, float(p),
            eps, swap, _reduction_enum(reduction))
        assert_reference_close(
            got, ref,
            msg=f"triplet shape={shape} red={reduction} p={p} eps={eps} "
                f"swap={swap} ({dev}, {dtype})")

    def test_reductions(self):
        for dev in reference_devices():
            for reduction in ("none", "mean", "sum"):
                for shape in ((16,), (4, 5), (2, 3, 4)):
                    self._run(shape, reduction, 1.0, 2.0, 1e-6, False, dev,
                              torch.float32, 3)

    def test_orders_and_swap(self):
        for dev in reference_devices():
            for p in (1.0, 2.0, 3.0, 1.5, float("inf")):
                for swap in (False, True):
                    self._run((5, 6), "mean", 0.7, p, 1e-6, swap, dev,
                              torch.float32, 4)

    def test_eps_shifts_the_distance(self):
        # A zero difference is not a zero distance: the shift joins every lane
        # of the difference before the magnitude is taken.
        for dev in reference_devices():
            for eps in (0.0, 1e-6, 0.25):
                self._run((3, 4), "mean", 1.0, 2.0, eps, False, dev,
                          torch.float32, 5)

    def test_margin_and_dtype(self):
        for dev in reference_devices():
            for dtype in (torch.float32, torch.float64):
                for margin in (0.1, 1.0, 4.0):
                    self._run((7, 5), "sum", margin, 2.0, 1e-6, False, dev,
                              dtype, 6)

    def test_non_contiguous_operands(self):
        for dev in reference_devices():
            torch.manual_seed(7)
            base = [torch.randn(6, 5) for _ in range(3)]
            operands = [from_reference(t.t(), dev) for t in base]
            ref = torch.triplet_margin_loss(
                *[t.t() for t in base], margin=0.5, p=2.0, eps=1e-6,
                swap=True, reduction=0)
            got = tp.triplet_margin_loss(*operands, 0.5, 2.0, 1e-6, True, 0)
            assert_reference_close(got, ref, msg=f"non-contiguous ({dev})")


class TestFunctionalForward(unittest.TestCase):
    """The module path takes its reduction by name."""

    def test_reductions(self):
        for dev in reference_devices():
            for reduction in ("none", "mean", "sum"):
                for swap in (False, True):
                    ref_operands = _operands((4, 6), 21)
                    ref = torch_F.triplet_margin_loss(
                        *ref_operands, margin=0.9, p=2.0, eps=1e-6,
                        swap=swap, reduction=reduction)
                    got = F.triplet_margin_loss(
                        *[from_reference(t, dev) for t in ref_operands],
                        margin=0.9, p=2.0, eps=1e-6, swap=swap,
                        reduction=reduction)
                    assert_reference_close(
                        got, ref,
                        msg=f"functional red={reduction} swap={swap} ({dev})")


class TestValidation(unittest.TestCase):
    def test_dimension_count(self):
        for dev in reference_devices():
            a = from_reference(torch.randn(3, 4), dev)
            b = from_reference(torch.randn(3, 4), dev)
            c = from_reference(torch.randn(3, 4, 5), dev)
            with self.assertRaises(RuntimeError):
                tp.triplet_margin_loss(a, b, c)
            with self.assertRaises(RuntimeError):
                F.triplet_margin_loss(a, c, b)

    def test_reduction_enum(self):
        for dev in reference_devices():
            a, b, c = _triple((3, 4), 31, dev, torch.float32)
            for bad in (-1, 3):
                with self.assertRaises(ValueError):
                    tp.triplet_margin_loss(a, b, c, 1.0, 2.0, 1e-6, False, bad)

    def test_margin_bound(self):
        for dev in reference_devices():
            a, b, c = _triple((3, 4), 32, dev, torch.float32)
            for bad in (0.0, -1.0):
                with self.assertRaises(ValueError):
                    F.triplet_margin_loss(a, b, c, margin=bad)


class TestMetaShape(unittest.TestCase):
    def test_shapes(self):
        a = tp.empty((3, 4), device="meta")
        b = tp.empty((3, 4), device="meta")
        c = tp.empty((3, 4), device="meta")
        assert tuple(tp.triplet_margin_loss(a, b, c).shape) == ()
        assert tuple(tp.triplet_margin_loss(a, b, c, 1.0, 2.0, 1e-6, False, 0).shape) == (3,)
        assert tuple(tp.triplet_margin_loss(a, b, c, 1.0, 2.0, 1e-6, False, 2).shape) == ()
        assert tuple(tp.triplet_margin_loss(
            tp.empty((2, 3, 4), device="meta"), b.expand(2, 3, 4),
            c.expand(2, 3, 4), 1.0, 2.0, 1e-6, True, 0).shape) == (2, 3)
        assert tp.triplet_margin_loss(a, b, c).dtype == tp.float32
        assert tp.triplet_margin_loss(
            tp.empty((3, 4), dtype=tp.float64, device="meta"), b, c).dtype == tp.float64


class TestAutograd(unittest.TestCase):
    def test_operands(self):
        for dev in reference_devices():
            for reduction in ("none", "mean", "sum"):
                for swap in (False, True):
                    self._run((4, 5), reduction, dev, 33, swap)

    def _run(self, shape, reduction, dev, seed, swap):
        torch.manual_seed(seed)
        anchor = torch.randn(*shape)
        positive = torch.randn(*shape)
        negative = torch.randn(*shape)
        ref_in = [t.clone().requires_grad_(True) for t in (anchor, positive, negative)]
        ref_out = torch_F.triplet_margin_loss(
            *ref_in, margin=1.0, p=2.0, eps=1e-6, swap=swap, reduction=reduction)
        grad_t = torch.randn_like(ref_out) if reduction == "none" else torch.tensor(1.0)
        ref_grads = torch.autograd.grad(ref_out, ref_in, grad_outputs=grad_t)

        slots = _triple(shape, seed, dev, torch.float32, requires_grad=True)
        out = F.triplet_margin_loss(
            *slots, margin=1.0, p=2.0, eps=1e-6, swap=swap, reduction=reduction)
        out.backward(from_reference(grad_t, dev))

        tag = f"triplet shape={shape} red={reduction} swap={swap} ({dev})"
        assert_reference_close(out, ref_out, msg=f"fwd {tag}")
        for name, slot, ref_grad in zip(("anchor", "positive", "negative"),
                                        slots, ref_grads):
            assert_reference_close(slot.grad, ref_grad, msg=f"grad {name} {tag}")


class TestModule(unittest.TestCase):
    def test_module(self):
        for dev in reference_devices():
            torch.manual_seed(41)
            anchor = torch.randn(4, 6)
            positive = torch.randn(4, 6)
            negative = torch.randn(4, 6)
            for reduction in ("mean", "sum", "none"):
                for swap in (False, True):
                    ref_in = anchor.clone().requires_grad_(True)
                    ref_mod = torch.nn.TripletMarginLoss(
                        margin=0.6, p=2.0, eps=1e-6, swap=swap,
                        reduction=reduction)
                    ref_out = ref_mod(ref_in, positive, negative)
                    grad_t = (torch.randn_like(ref_out) if reduction == "none"
                              else torch.tensor(1.0))
                    (ref_grad,) = torch.autograd.grad(
                        ref_out, ref_in, grad_outputs=grad_t)

                    x = from_reference(anchor, dev, requires_grad=True)
                    mod = tp.nn.TripletMarginLoss(
                        margin=0.6, p=2.0, eps=1e-6, swap=swap,
                        reduction=reduction)
                    out = mod(x, from_reference(positive, dev),
                              from_reference(negative, dev))
                    out.backward(from_reference(grad_t, dev))
                    tag = f"module red={reduction} swap={swap} ({dev})"
                    assert_reference_close(out, ref_out, msg=f"fwd {tag}")
                    assert_reference_close(x.grad, ref_grad, msg=f"grad {tag}")


if __name__ == "__main__":
    unittest.main()