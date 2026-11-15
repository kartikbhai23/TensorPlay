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
KL_MODES = (False, True)
SMOOTH_BETAS = (1.0, 0.5)
POISSON_MODES = ((False, False), (False, True), (True, False), (True, True))
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


class TestDivergenceWalk(unittest.TestCase):
    """The divergence folds both target spaces in one pass as well."""

    def _positive_pair(self, shape, dtype, seed):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(*shape, dtype=dtype, generator=generator)
        target = torch.rand(*shape, dtype=dtype, generator=generator)
        target.reshape(-1)[::3] = 0.0
        return x, target

    def test_forward(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES + [(0, 3)]:
                    for log_target in KL_MODES:
                        ref_x, ref_t = self._positive_pair(shape, dtype, 11)
                        if log_target and ref_t.numel():
                            ref_t = torch.log(ref_t)
                        for name, reduction in REDUCTIONS.items():
                            expected = torch_F.kl_div(
                                ref_x, ref_t, reduction=name,
                                log_target=log_target)
                            got = tp.kl_div(
                                from_reference(ref_x, dev),
                                from_reference(ref_t, dev),
                                reduction=reduction, log_target=log_target)
                            self.assertEqual(tuple(got.shape),
                                             tuple(expected.shape))
                            if not got.numel():
                                continue
                            assert_reference_close(
                                got, expected,
                                msg=f"kl_div {shape} {dtype} {name} "
                                    f"log_target={log_target} ({dev})")

    def test_gradient(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for log_target in KL_MODES:
                    ref_x, ref_t = self._positive_pair((4, 5), dtype, 13)
                    if log_target:
                        ref_t = torch.log(ref_t)
                    for name, reduction in REDUCTIONS.items():
                        ref_leaf = ref_x.clone().requires_grad_(True)
                        ref_out = torch_F.kl_div(
                            ref_leaf, ref_t, reduction=name,
                            log_target=log_target)
                        seed = (torch.randn(ref_out.shape, dtype=dtype)
                                if name == "none"
                                else torch.ones((), dtype=dtype))
                        ref_grad = torch.autograd.grad(
                            ref_out, ref_leaf, grad_outputs=seed)[0]
                        leaf = from_reference(ref_x, dev, requires_grad=True)
                        out = tp.kl_div(leaf, from_reference(ref_t, dev),
                                        reduction=reduction,
                                        log_target=log_target)
                        out.backward(from_reference(seed, dev))
                        assert_reference_close(
                            leaf.grad, ref_grad,
                            msg=f"kl_div grad {dtype} {name} "
                                f"log_target={log_target} ({dev})")


class TestSmoothL1Walk(unittest.TestCase):
    """The fused smooth_l1 pass agrees with the reference arithmetic."""

    def test_forward(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES:
                    for beta in SMOOTH_BETAS:
                        for name, reduction in REDUCTIONS.items():
                            ref_x, ref_t = _pair(shape, dtype, 17)
                            expected = torch_F.smooth_l1_loss(
                                ref_x, ref_t, beta=beta, reduction=name)
                            got = tp.smooth_l1_loss(
                                from_reference(ref_x, dev),
                                from_reference(ref_t, dev),
                                reduction=reduction, beta=beta)
                            self.assertEqual(tuple(got.shape),
                                             tuple(expected.shape))
                            assert_reference_close(
                                got, expected,
                                msg=f"smooth_l1_loss {shape} {dtype} beta={beta} "
                                    f"{name} ({dev})")

    def test_gradient(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES:
                    for beta in SMOOTH_BETAS:
                        for name, reduction in REDUCTIONS.items():
                            ref_x, ref_t = _pair(shape, dtype, 19)
                            ref_leaves = [t.clone().requires_grad_(True)
                                          for t in (ref_x, ref_t)]
                            ref_out = torch_F.smooth_l1_loss(
                                *ref_leaves, beta=beta, reduction=name)
                            seed = (torch.randn(ref_out.shape, dtype=dtype)
                                    if name == "none"
                                    else torch.ones((), dtype=dtype))
                            ref_grads = torch.autograd.grad(
                                ref_out, ref_leaves, grad_outputs=seed)

                            leaves = [from_reference(t, dev,
                                                     requires_grad=True)
                                      for t in (ref_x, ref_t)]
                            out = tp.smooth_l1_loss(
                                *leaves, reduction=reduction, beta=beta)
                            out.backward(from_reference(seed, dev))
                            for leaf, ref_grad in zip(leaves, ref_grads):
                                assert_reference_close(
                                    leaf.grad, ref_grad,
                                    msg=f"smooth_l1_loss grad {shape} {dtype} "
                                        f"beta={beta} {name} ({dev})")

    def test_zero_beta_forward(self):
        # beta = 0 degenerates to the absolute deviation; the fused walk must
        # not divide.
        for dev in reference_devices():
            for name, reduction in REDUCTIONS.items():
                ref_x, ref_t = _pair((4, 5), torch.float64, 23)
                expected = torch_F.smooth_l1_loss(
                    ref_x, ref_t, beta=0.0, reduction=name)
                got = tp.smooth_l1_loss(
                    from_reference(ref_x, dev), from_reference(ref_t, dev),
                    reduction=reduction, beta=0.0)
                assert_reference_close(
                    got, expected,
                    msg=f"smooth_l1_loss beta=0 {name} ({dev})")


class TestSoftMarginWalk(unittest.TestCase):
    """The fused soft_margin pass agrees with the reference arithmetic."""

    def test_forward(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES:
                    for name, reduction in REDUCTIONS.items():
                        ref_x, ref_t = _pair(shape, dtype, 29)
                        expected = torch_F.soft_margin_loss(
                            ref_x, ref_t, reduction=name)
                        got = tp.soft_margin_loss(
                            from_reference(ref_x, dev),
                            from_reference(ref_t, dev),
                            reduction=reduction)
                        self.assertEqual(tuple(got.shape),
                                         tuple(expected.shape))
                        assert_reference_close(
                            got, expected,
                            msg=f"soft_margin_loss {shape} {dtype} {name} ({dev})")

    def test_gradient(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES:
                    for name, reduction in REDUCTIONS.items():
                        ref_x, ref_t = _pair(shape, dtype, 31)
                        ref_leaf = ref_x.clone().requires_grad_(True)
                        ref_out = torch_F.soft_margin_loss(
                            ref_leaf, ref_t, reduction=name)
                        seed = (torch.randn(ref_out.shape, dtype=dtype)
                                if name == "none"
                                else torch.ones((), dtype=dtype))
                        ref_grad = torch.autograd.grad(
                            ref_out, ref_leaf, grad_outputs=seed)[0]

                        leaf = from_reference(ref_x, dev, requires_grad=True)
                        out = tp.soft_margin_loss(
                            leaf, from_reference(ref_t, dev),
                            reduction=reduction)
                        out.backward(from_reference(seed, dev))
                        assert_reference_close(
                            leaf.grad, ref_grad,
                            msg=f"soft_margin_loss grad {shape} {dtype} "
                                f"{name} ({dev})")


class TestPoissonWalk(unittest.TestCase):
    """The fused poisson pass agrees with the reference arithmetic."""

    def _pair(self, shape, dtype, seed, log_input):
        generator = torch.Generator().manual_seed(seed)
        if log_input:
            x = torch.randn(*shape, dtype=dtype, generator=generator) * 0.5
        else:
            x = torch.rand(*shape, dtype=dtype, generator=generator) * 1.5 + 0.5
        target = torch.rand(*shape, dtype=dtype, generator=generator) + 0.5
        return x, target

    def test_forward(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES:
                    for log_input, full in POISSON_MODES:
                        for name, reduction in REDUCTIONS.items():
                            ref_x, ref_t = self._pair(
                                shape, dtype, 33, log_input)
                            expected = torch_F.poisson_nll_loss(
                                ref_x, ref_t, log_input=log_input,
                                full=full, reduction=name)
                            got = tp.poisson_nll_loss(
                                from_reference(ref_x, dev),
                                from_reference(ref_t, dev),
                                log_input=log_input, full=full,
                                reduction=reduction)
                            self.assertEqual(tuple(got.shape),
                                             tuple(expected.shape))
                            assert_reference_close(
                                got, expected,
                                msg=f"poisson_nll_loss {shape} {dtype} "
                                    f"log_input={log_input} full={full} "
                                    f"{name} ({dev})")

    def test_gradient(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for log_input, full in POISSON_MODES:
                    for name, reduction in REDUCTIONS.items():
                        ref_x, ref_t = self._pair((4, 5), dtype, 35, log_input)
                        ref_leaves = [t.clone().requires_grad_(True)
                                      for t in (ref_x, ref_t)]
                        ref_out = torch_F.poisson_nll_loss(
                            *ref_leaves, log_input=log_input, full=full,
                            reduction=name)
                        seed = (torch.randn(ref_out.shape, dtype=dtype)
                                if name == "none"
                                else torch.ones((), dtype=dtype))
                        ref_grads = torch.autograd.grad(
                            ref_out, ref_leaves, grad_outputs=seed)

                        leaves = [from_reference(t, dev, requires_grad=True)
                                  for t in (ref_x, ref_t)]
                        out = tp.poisson_nll_loss(
                            *leaves, log_input=log_input, full=full,
                            reduction=reduction)
                        out.backward(from_reference(seed, dev))
                        for leaf, ref_grad in zip(leaves, ref_grads):
                            assert_reference_close(
                                leaf.grad, ref_grad,
                                msg=f"poisson_nll_loss grad {dtype} "
                                    f"log_input={log_input} full={full} "
                                    f"{name} ({dev})")


class TestMarginRankingWalk(unittest.TestCase):
    """The fused margin_ranking pass agrees with the reference arithmetic."""

    def test_forward(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES:
                    for name, reduction in REDUCTIONS.items():
                        ref_x1, ref_x2 = _pair(shape, dtype, 41)
                        ref_t = torch.randn(*shape, dtype=dtype,
                                            generator=torch.Generator().manual_seed(43)).sign()
                        expected = torch_F.margin_ranking_loss(
                            ref_x1, ref_x2, ref_t, margin=0.7, reduction=name)
                        got = tp.margin_ranking_loss(
                            from_reference(ref_x1, dev),
                            from_reference(ref_x2, dev),
                            from_reference(ref_t, dev),
                            margin=0.7, reduction=reduction)
                        self.assertEqual(tuple(got.shape),
                                         tuple(expected.shape))
                        assert_reference_close(
                            got, expected,
                            msg=f"margin_ranking_loss {shape} {dtype} {name} ({dev})")

    def test_gradient(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES:
                    for name, reduction in REDUCTIONS.items():
                        ref_x1, ref_x2 = _pair(shape, dtype, 45)
                        ref_t = torch.randn(*shape, dtype=dtype,
                                            generator=torch.Generator().manual_seed(47)).sign()
                        ref_leaves = [t.clone().requires_grad_(True)
                                      for t in (ref_x1, ref_x2, ref_t)]
                        ref_out = torch_F.margin_ranking_loss(
                            *ref_leaves, margin=0.7, reduction=name)
                        seed = (torch.randn(ref_out.shape, dtype=dtype)
                                if name == "none"
                                else torch.ones((), dtype=dtype))
                        ref_grads = torch.autograd.grad(
                            ref_out, ref_leaves, grad_outputs=seed)

                        leaves = [from_reference(t, dev, requires_grad=True)
                                  for t in (ref_x1, ref_x2, ref_t)]
                        out = tp.margin_ranking_loss(
                            *leaves, margin=0.7, reduction=reduction)
                        out.backward(from_reference(seed, dev))
                        for leaf, ref_grad in zip(leaves, ref_grads):
                            assert_reference_close(
                                leaf.grad, ref_grad,
                                msg=f"margin_ranking_loss grad {shape} {dtype} "
                                    f"{name} ({dev})")


class TestHingeWalk(unittest.TestCase):
    """The fused hinge_embedding pass agrees with the reference arithmetic."""

    def _pair(self, shape, dtype, seed):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(*shape, dtype=dtype, generator=generator)
        t = torch.where(torch.rand(*shape, generator=generator) > 0.5,
                        torch.ones(*shape, dtype=dtype),
                        -torch.ones(*shape, dtype=dtype))
        return x, t

    def test_forward(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES:
                    for name, reduction in REDUCTIONS.items():
                        ref_x, ref_t = self._pair(shape, dtype, 51)
                        expected = torch_F.hinge_embedding_loss(
                            ref_x, ref_t, margin=1.0, reduction=name)
                        got = tp.hinge_embedding_loss(
                            from_reference(ref_x, dev),
                            from_reference(ref_t, dev),
                            margin=1.0, reduction=reduction)
                        self.assertEqual(tuple(got.shape),
                                         tuple(expected.shape))
                        assert_reference_close(
                            got, expected,
                            msg=f"hinge_embedding_loss {shape} {dtype} {name} ({dev})")

    def test_gradient(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in SHAPES:
                    for name, reduction in REDUCTIONS.items():
                        ref_x, ref_t = self._pair(shape, dtype, 53)
                        ref_leaf = ref_x.clone().requires_grad_(True)
                        ref_out = torch_F.hinge_embedding_loss(
                            ref_leaf, ref_t, margin=1.0, reduction=name)
                        seed = (torch.randn(ref_out.shape, dtype=dtype)
                                if name == "none"
                                else torch.ones((), dtype=dtype))
                        ref_grad = torch.autograd.grad(
                            ref_out, ref_leaf, grad_outputs=seed)[0]

                        leaf = from_reference(ref_x, dev, requires_grad=True)
                        out = tp.hinge_embedding_loss(
                            leaf, from_reference(ref_t, dev),
                            margin=1.0, reduction=reduction)
                        out.backward(from_reference(seed, dev))
                        assert_reference_close(
                            leaf.grad, ref_grad,
                            msg=f"hinge_embedding_loss grad {shape} {dtype} "
                                f"{name} ({dev})")


class TestCosineWalk(unittest.TestCase):
    """The fused cosine_embedding pass agrees with the reference arithmetic."""

    COSINE_SHAPES = [(4, 5), (6, 3)]

    def _pair(self, shape, dtype, seed):
        generator = torch.Generator().manual_seed(seed)
        x1 = torch.randn(*shape, dtype=dtype, generator=generator)
        x2 = torch.randn(*shape, dtype=dtype, generator=generator)
        target = torch.tensor([1.0, -1.0] * ((shape[0] + 1) // 2),
                              dtype=dtype)[:shape[0]]
        return x1, x2, target

    def test_forward(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in self.COSINE_SHAPES:
                    for name, reduction in REDUCTIONS.items():
                        ref_x1, ref_x2, ref_t = self._pair(shape, dtype, 61)
                        expected = torch_F.cosine_embedding_loss(
                            ref_x1, ref_x2, ref_t, margin=0.3, reduction=name)
                        got = tp.cosine_embedding_loss(
                            from_reference(ref_x1, dev),
                            from_reference(ref_x2, dev),
                            from_reference(ref_t, dev),
                            margin=0.3, reduction=reduction)
                        self.assertEqual(tuple(got.shape),
                                         tuple(expected.shape))
                        assert_reference_close(
                            got, expected,
                            msg=f"cosine_embedding_loss {shape} {dtype} {name} ({dev})")

    def test_gradient(self):
        for dev in reference_devices():
            for dtype in DTYPES:
                for shape in self.COSINE_SHAPES:
                    for name, reduction in REDUCTIONS.items():
                        ref_x1, ref_x2, ref_t = self._pair(shape, dtype, 63)
                        ref_leaves = [t.clone().requires_grad_(True)
                                      for t in (ref_x1, ref_x2)]
                        ref_out = torch_F.cosine_embedding_loss(
                            *ref_leaves, ref_t, margin=0.3, reduction=name)
                        seed = (torch.randn(ref_out.shape, dtype=dtype)
                                if name == "none"
                                else torch.ones((), dtype=dtype))
                        ref_grads = torch.autograd.grad(
                            ref_out, ref_leaves, grad_outputs=seed)

                        leaves = [from_reference(t, dev, requires_grad=True)
                                  for t in (ref_x1, ref_x2)]
                        out = tp.cosine_embedding_loss(
                            *leaves, from_reference(ref_t, dev),
                            margin=0.3, reduction=reduction)
                        out.backward(from_reference(seed, dev))
                        for leaf, ref_grad in zip(leaves, ref_grads):
                            assert_reference_close(
                                leaf.grad, ref_grad,
                                msg=f"cosine_embedding_loss grad {shape} {dtype} "
                                    f"{name} ({dev})")


class TestEmptyPlane(unittest.TestCase):
    """Nothing walked answers an empty plane, zero, or no average at all."""

    def test_empty_reductions(self):
        empty = torch.zeros((0, 3), dtype=torch.float64)
        fused_ops = OPS + ("smooth_l1_loss", "soft_margin_loss",
                           "poisson_nll_loss", "hinge_embedding_loss")
        empty_target = torch.zeros((0,), dtype=torch.float64)
        for dev in reference_devices():
            for op in fused_ops:
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
            # margin_ranking_loss takes three operands; its empty plane is (0,3).
            mr = tp.margin_ranking_loss(from_reference(empty, dev),
                                        from_reference(empty, dev),
                                        from_reference(empty, dev),
                                        reduction=0)
            self.assertEqual(tuple(mr.shape), (0, 3))
            self.assertEqual(mr.numel(), 0)
            mr_sum = tp.margin_ranking_loss(from_reference(empty, dev),
                                            from_reference(empty, dev),
                                            from_reference(empty, dev),
                                            reduction=2)
            self.assertEqual(float(mr_sum.item()), 0.0)
            mr_mean = tp.margin_ranking_loss(from_reference(empty, dev),
                                             from_reference(empty, dev),
                                             from_reference(empty, dev),
                                             reduction=1)
            self.assertTrue(float(mr_mean.item()) != float(mr_mean.item()))
            # cosine_embedding_loss reduces rows, so its empty plane is (0,).
            ce = from_reference(empty, dev)
            cet = from_reference(empty_target, dev)
            plane = tp.cosine_embedding_loss(ce, ce, cet, reduction=0)
            self.assertEqual(tuple(plane.shape), (0,))
            self.assertEqual(plane.numel(), 0)
            total = tp.cosine_embedding_loss(ce, ce, cet, reduction=2)
            self.assertEqual(float(total.item()), 0.0)
            average = tp.cosine_embedding_loss(ce, ce, cet, reduction=1)
            self.assertTrue(float(average.item()) != float(average.item()))


if __name__ == "__main__":
    unittest.main()