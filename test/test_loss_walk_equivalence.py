"""The fused squared and absolute loss walks answer the general walks' numbers.

Both losses fold their operands in a single contiguous float pass and fall back
to composing the deviation from primitive ops for every other layout or width.
The checks here pin the fast pass to the arithmetic definition, keep the two
passes in step on the same values, and cover the empty plane where the
reduction decides between a zero and an undefined average.
"""
import os
import sys
import unittest

import torch
import torch.nn.functional as torch_F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp
from tensorplay.testing._internal.reference import (assert_reference_close,
                                                   from_reference,
                                                   reference_devices)

OPS = ("mse_loss", "l1_loss")
REDUCTIONS = {"none": 0, "mean": 1, "sum": 2}
SHAPES = [(4, 5), (7,), (2, 3, 4)]
DTYPES = (torch.float32, torch.float64)


def _pair(shape, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, dtype=dtype, generator=generator),
            torch.randn(*shape, dtype=dtype, generator=generator))


class TestFastWalk(unittest.TestCase):
    """The contiguous float pass agrees with the reference arithmetic."""

    def test_forward(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES:
                    for op in OPS:
                        for name, reduction in REDUCTIONS.items():
                            ref_x, ref_t = _pair(shape, dtype, 3)
                            expected = getattr(torch_F, op)(
                                ref_x, ref_t, reduction=name)
                            got = getattr(tp, op)(
                                from_reference(ref_x, dev),
                                from_reference(ref_t, dev),
                                reduction=reduction)
                            self.assertEqual(tuple(got.shape),
                                             tuple(expected.shape))
                            assert_reference_close(
                                got, expected,
                                msg=f"{op} {shape} {dtype} {name} ({dev})")

    def test_gradient(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES:
                    for op in OPS:
                        for name, reduction in REDUCTIONS.items():
                            ref_x, ref_t = _pair(shape, dtype, 5)
                            ref_leaves = [t.clone().requires_grad_(True)
                                          for t in (ref_x, ref_t)]
                            ref_out = getattr(torch_F, op)(
                                *ref_leaves, reduction=name)
                            seed = (torch.randn(ref_out.shape, dtype=dtype)
                                    if name == "none"
                                    else torch.ones((), dtype=dtype))
                            ref_grads = torch.autograd.grad(
                                ref_out, ref_leaves, grad_outputs=seed)

                            leaves = [from_reference(t, dev,
                                                     requires_grad=True)
                                      for t in (ref_x, ref_t)]
                            out = getattr(tp, op)(*leaves, reduction=reduction)
                            out.backward(from_reference(seed, dev))
                            for leaf, ref_grad in zip(leaves, ref_grads):
                                assert_reference_close(
                                    leaf.grad, ref_grad,
                                    msg=f"{op} grad {shape} {dtype} {name} ({dev})")


class TestGeneralWalk(unittest.TestCase):
    """A strided operand answers what its contiguous copy answers."""

    def test_strided_view(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                base_x, base_t = _pair((6, 5), dtype, 7)
                for op in OPS:
                    for name, reduction in REDUCTIONS.items():
                        expected = getattr(torch_F, op)(
                            base_x[:, ::2], base_t[:, ::2], reduction=name)
                        # The view is not a contiguous pair, so the general
                        # walk answers here.
                        got = getattr(tp, op)(
                            from_reference(base_x, dev)[:, ::2],
                            from_reference(base_t, dev)[:, ::2],
                            reduction=reduction)
                        assert_reference_close(
                            got, expected,
                            msg=f"{op} strided {name} ({dev})")

    def test_strided_gradient(self):
        for dev in reference_devices():
            for op in OPS:
                base_x, base_t = _pair((6, 5), torch.float64, 9)
                # A view is not a leaf, so the answer lands on the base plane
                # with the untouched columns left at zero.
                ref_leaves = [t.clone().requires_grad_(True)
                              for t in (base_x, base_t)]
                ref_out = getattr(torch_F, op)(
                    *(t[:, ::2] for t in ref_leaves), reduction="mean")
                ref_grads = torch.autograd.grad(
                    ref_out, ref_leaves, grad_outputs=torch.ones(()))
                leaves = [from_reference(t, dev, requires_grad=True)
                          for t in (base_x, base_t)]
                out = getattr(tp, op)(*(t[:, ::2] for t in leaves),
                                      reduction=1)
                out.backward(from_reference(torch.ones(()), dev))
                for leaf, ref_grad in zip(leaves, ref_grads):
                    assert_reference_close(
                        leaf.grad, ref_grad,
                        msg=f"{op} strided grad ({dev})")


class TestEmptyPlane(unittest.TestCase):
    """Nothing walked answers an empty plane, zero, or no average at all."""

    def test_empty_reductions(self):
        empty = torch.zeros((0, 3), dtype=torch.float64)
        for dev in reference_devices():
            for op in OPS:
                plane = getattr(tp, op)(from_reference(empty, dev),
                                        from_reference(empty, dev),
                                        reduction=0)
                self.assertEqual(tuple(plane.shape), (0, 3))
                self.assertEqual(plane.numel(), 0)
                total = getattr(tp, op)(from_reference(empty, dev),
                                        from_reference(empty, dev),
                                        reduction=2)
                self.assertEqual(tuple(total.shape), ())
                self.assertEqual(float(total.item()), 0.0)
                average = getattr(tp, op)(from_reference(empty, dev),
                                          from_reference(empty, dev),
                                          reduction=1)
                self.assertEqual(tuple(average.shape), ())
                self.assertTrue(float(average.item()) != float(average.item()))


if __name__ == "__main__":
    unittest.main()