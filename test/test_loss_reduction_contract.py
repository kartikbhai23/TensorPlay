"""The loss names take their reduction on the native op, not only in the module.

Each of these losses used to answer the mean whatever the caller wanted, so the
native entry point could not express ``none`` (the per-element plane) or ``sum``.
The checks below pin both halves of the contract: the value each reduction mode
answers, and the gradient that flows back through the new entries.
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


REDUCTIONS = ("none", "mean", "sum")

# Keyword extras each loss needs beyond its operands.
EXTRAS = {
    "l1_loss": {},
    "kl_div": {},
    "soft_margin_loss": {},
    "poisson_nll_loss": {},
    "margin_ranking_loss": {"margin": 0.7},
    "hinge_embedding_loss": {"margin": 1.0},
    "cosine_embedding_loss": {"margin": 0.3},
}


def _reduction_enum(reduction):
    return {"none": 0, "mean": 1, "sum": 2}[reduction]


def _torch_inputs(op, dtype, seed, **kwargs):
    """Reference operands that stay inside every loss' domain."""
    torch.manual_seed(seed)
    if op == "kl_div":
        logp = torch.log_softmax(torch.randn(4, 5, dtype=dtype), dim=1)
        if kwargs.get("log_target", False):
            return [logp, torch.log_softmax(torch.randn(4, 5, dtype=dtype), dim=1)]
        return [logp, logp.exp()]
    if op == "poisson_nll_loss":
        return [torch.randn(4, 5, dtype=dtype) * 0.5,
                torch.rand(4, 5, dtype=dtype) + 0.5]
    if op == "margin_ranking_loss":
        return [torch.randn(4, 5, dtype=dtype),
                torch.randn(4, 5, dtype=dtype),
                torch.randn(4, 5, dtype=dtype).sign()]
    if op == "hinge_embedding_loss":
        return [torch.randn(4, 5, dtype=dtype),
                torch.where(torch.rand(4, 5) > 0.5,
                            torch.ones(4, 5, dtype=dtype),
                            -torch.ones(4, 5, dtype=dtype))]
    if op == "cosine_embedding_loss":
        return [torch.randn(4, 5, dtype=dtype),
                torch.randn(4, 5, dtype=dtype),
                torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=dtype)]
    return [torch.randn(4, 5, dtype=dtype), torch.randn(4, 5, dtype=dtype)]


class TestForwardReductions(unittest.TestCase):
    def _check(self, op, dev, dtype, **kwargs):
        for reduction in REDUCTIONS:
            ref_slots = _torch_inputs(op, dtype, 5, **kwargs)
            slots = [from_reference(t, dev) for t in ref_slots]
            ref = getattr(torch_F, op)(*ref_slots, reduction=reduction, **kwargs)
            got = getattr(tp, op)(*slots, reduction=_reduction_enum(reduction),
                                  **kwargs)
            assert_reference_close(
                got, ref,
                msg=f"{op} reduction={reduction} ({dev}, {dtype})")

    def test_l1_loss(self):
        for dev in reference_devices():
            for dtype in (torch.float32, torch.float64):
                self._check("l1_loss", dev, dtype)

    def test_kl_div(self):
        for dev in reference_devices():
            for log_target in (False, True):
                self._check("kl_div", dev, torch.float64, log_target=log_target)

    def test_soft_margin_loss(self):
        for dev in reference_devices():
            for dtype in (torch.float32, torch.float64):
                self._check("soft_margin_loss", dev, dtype)

    def test_poisson_nll_loss(self):
        for dev in reference_devices():
            self._check("poisson_nll_loss", dev, torch.float64)

    def test_margin_ranking_loss(self):
        for dev in reference_devices():
            self._check("margin_ranking_loss", dev, torch.float64,
                        **EXTRAS["margin_ranking_loss"])

    def test_hinge_embedding_loss(self):
        for dev in reference_devices():
            self._check("hinge_embedding_loss", dev, torch.float64,
                        **EXTRAS["hinge_embedding_loss"])

    def test_cosine_embedding_loss(self):
        for dev in reference_devices():
            self._check("cosine_embedding_loss", dev, torch.float64,
                        **EXTRAS["cosine_embedding_loss"])


class TestFunctionalReductions(unittest.TestCase):
    """The module path keeps taking its reduction by name."""

    def test_names(self):
        pairs = (
            ("l1_loss", F.l1_loss, torch_F.l1_loss),
            ("soft_margin_loss", F.soft_margin_loss, torch_F.soft_margin_loss),
            ("poisson_nll_loss", F.poisson_nll_loss, torch_F.poisson_nll_loss),
            ("margin_ranking_loss", F.margin_ranking_loss,
             torch_F.margin_ranking_loss),
            ("hinge_embedding_loss", F.hinge_embedding_loss,
             torch_F.hinge_embedding_loss),
            ("cosine_embedding_loss", F.cosine_embedding_loss,
             torch_F.cosine_embedding_loss),
        )
        for dev in reference_devices():
            for op, tp_fn, ref_fn in pairs:
                for reduction in REDUCTIONS:
                    ref_slots = _torch_inputs(op, torch.float64, 7, **EXTRAS[op])
                    slots = [from_reference(t, dev) for t in ref_slots]
                    ref = ref_fn(*ref_slots, reduction=reduction,
                                 **EXTRAS[op])
                    got = tp_fn(*slots, reduction=reduction, **EXTRAS[op])
                    assert_reference_close(
                        got, ref, msg=f"F.{op} reduction={reduction} ({dev})")


class TestGradient(unittest.TestCase):
    """Every operand the reference differentiates keeps its gradient."""

    def test_l1_and_kl(self):
        for dev in reference_devices():
            for op in ("l1_loss", "kl_div"):
                for reduction in REDUCTIONS:
                    ref_slots = _torch_inputs(op, torch.float64, 11)
                    ref_leaves = [t.clone().requires_grad_(True)
                                  for t in ref_slots]
                    ref_out = getattr(torch_F, op)(*ref_leaves,
                                                   reduction=reduction)
                    grad_t = (torch.randn_like(ref_out) if reduction == "none"
                              else torch.tensor(1.0))
                    ref_grads = torch.autograd.grad(
                        ref_out, ref_leaves, grad_outputs=grad_t, allow_unused=True)

                    leaves = [from_reference(t, dev, requires_grad=True)
                              for t in ref_slots]
                    out = getattr(tp, op)(*leaves,
                                          reduction=_reduction_enum(reduction))
                    out.backward(from_reference(grad_t, dev))
                    tag = f"{op} reduction={reduction} ({dev})"
                    assert_reference_close(out, ref_out, msg=f"fwd {tag}")
                    for leaf, ref_grad in zip(leaves, ref_grads):
                        if ref_grad is None:
                            continue
                        assert_reference_close(leaf.grad, ref_grad,
                                               msg=f"grad {tag}")


class TestMetaShapes(unittest.TestCase):
    def test_shapes(self):
        a = tp.empty((4, 5), device="meta")
        b = tp.empty((4, 5), device="meta")
        t = tp.empty((4,), device="meta")
        assert tuple(tp.l1_loss(a, b).shape) == ()
        assert tuple(tp.l1_loss(a, b, reduction=0).shape) == (4, 5)
        assert tuple(tp.l1_loss(a, b, reduction=2).shape) == ()
        assert tuple(tp.kl_div(a, b, reduction=0).shape) == (4, 5)
        assert tuple(tp.soft_margin_loss(a, b).shape) == ()
        assert tuple(tp.poisson_nll_loss(a, b).shape) == ()
        assert tuple(tp.poisson_nll_loss(a, b, reduction=0).shape) == (4, 5)
        assert tuple(tp.margin_ranking_loss(a, b, b, reduction=2).shape) == ()
        assert tuple(tp.hinge_embedding_loss(a, b, reduction=0).shape) == (4, 5)
        assert tuple(tp.cosine_embedding_loss(a, b, t, reduction=0).shape) == (4,)


class TestValidation(unittest.TestCase):
    def test_reduction_enum(self):
        for dev in reference_devices():
            a = from_reference(torch.randn(4, 5), dev)
            for bad in (-1, 3):
                with self.assertRaises(ValueError):
                    tp.l1_loss(a, a, reduction=bad)
                with self.assertRaises(ValueError):
                    tp.margin_ranking_loss(a, a, a, reduction=bad)
                with self.assertRaises(ValueError):
                    tp.hinge_embedding_loss(a, a, reduction=bad)


if __name__ == "__main__":
    unittest.main()