import unittest

import numpy as np

import tensorplay as tp

# The deferred-sign view lives in the operator namespace, like the rest of the
# internal `_`-prefixed surface.
neg_view = tp.ops.tp._neg_view


class NegViewDeferredSignTest(unittest.TestCase):
    """A negated view carries its sign as metadata until values are reached."""

    def devices(self):
        devices = ["cpu"]
        if tp.cuda.is_available():
            devices.append("cuda")
        return devices

    def make(self, device, shape=(3, 4), dtype=None):
        dtype = tp.float32 if dtype is None else dtype
        np_dtype = np.float32 if dtype == tp.float32 else np.float64
        values = np.arange(int(np.prod(shape)), dtype=np_dtype).reshape(shape)
        return tp.tensor(values.tolist(), dtype=dtype, device=device), values

    def test_values_are_negated(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = neg_view(x)
                np.testing.assert_allclose(y.cpu().numpy(), -values)

    def test_sign_is_reported_and_source_untouched(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = neg_view(x)
                self.assertTrue(tp.is_neg(y))
                self.assertFalse(tp.is_neg(x))
                np.testing.assert_allclose(x.cpu().numpy(), values)

    def test_write_through_the_view_leaves_the_source(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = neg_view(x)
                y.add_(1.0)
                np.testing.assert_allclose(y.cpu().numpy(), 1.0 - values)
                np.testing.assert_allclose(x.cpu().numpy(), values)

    def test_write_to_the_source_leaves_the_view(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = neg_view(x)
                x.mul_(3.0)
                np.testing.assert_allclose(x.cpu().numpy(), values * 3.0)
                np.testing.assert_allclose(y.cpu().numpy(), -values)

    def test_copy_and_fill_on_the_view(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = neg_view(x)
                y.copy_(y + 1.0)
                np.testing.assert_allclose(y.cpu().numpy(), 1.0 - values)
                np.testing.assert_allclose(x.cpu().numpy(), values)
                z = neg_view(x)
                z.fill_(7.0)
                np.testing.assert_allclose(z.cpu().numpy(), np.full_like(values, 7.0))
                np.testing.assert_allclose(x.cpu().numpy(), values)

    def test_strided_source_and_view_of_the_view(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                base = x.t()
                y = neg_view(base)
                # The sign is metadata a view inherits, so it is visible before
                # anything reads the bytes.
                self.assertTrue(tp.is_neg(y.t()))
                np.testing.assert_allclose(y.t().cpu().numpy(), (-values.T).T)
                np.testing.assert_allclose(y.cpu().numpy(), -values.T)
                self.assertEqual(y.stride(), base.stride())
                base.add_(5.0)
                np.testing.assert_allclose(y.cpu().numpy(), -values.T)
                np.testing.assert_allclose(x.cpu().numpy(), values + 5.0)

    def test_double_negation_returns_the_values(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = neg_view(neg_view(x))
                self.assertFalse(tp.is_neg(y))
                np.testing.assert_allclose(y.cpu().numpy(), values)

    def test_resolve_neg_reports_values_without_a_sign(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = neg_view(x)
                z = tp.resolve_neg(y)
                # Clearing the sign makes the bytes hold the values they stand
                # for; the tensor itself is what answers.
                self.assertFalse(tp.is_neg(z))
                np.testing.assert_allclose(z.cpu().numpy(), -values)
                np.testing.assert_allclose(x.cpu().numpy(), values)
                # A tensor without a sign passes through untouched.
                self.assertFalse(tp.is_neg(tp.resolve_neg(x)))

    def test_reductions_and_out_variants_see_the_sign(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = neg_view(x)
                self.assertAlmostEqual(float(y.sum().cpu()), float(-values.sum()), places=4)
                out = tp.empty(x.shape, dtype=x.dtype, device=device)
                tp.abs(y, out=out)
                np.testing.assert_allclose(out.cpu().numpy(), np.abs(values))

    def test_repeated_writes_only_pay_one_copy(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = neg_view(x)
                for _ in range(4):
                    y.add_(1.0)
                np.testing.assert_allclose(y.cpu().numpy(), 4.0 - values)
                np.testing.assert_allclose(x.cpu().numpy(), values)

    def test_gradient_still_reaches_the_source(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                x = x.requires_grad_(True)
                y = neg_view(x)
                y.sum().backward()
                np.testing.assert_allclose(x.grad.cpu().numpy(),
                                           -np.ones_like(values))

    def test_integer_and_double(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device, dtype=tp.float64)
                np.testing.assert_allclose(neg_view(x).cpu().numpy(), -values)
                i = tp.tensor([[1, -2], [3, 4]], dtype=tp.int64, device=device)
                np.testing.assert_allclose(neg_view(i).cpu().numpy(),
                                           np.array([[-1, 2], [-3, -4]]))

    def test_materialized_copy_is_independent(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = tp.ops.tp._neg_view_copy(x)
                self.assertFalse(tp.is_neg(y))
                np.testing.assert_allclose(y.cpu().numpy(), -values)
                y.add_(1.0)
                np.testing.assert_allclose(x.cpu().numpy(), values)


if __name__ == "__main__":
    unittest.main()