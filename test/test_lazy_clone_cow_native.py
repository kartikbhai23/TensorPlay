import unittest

import numpy as np

import tensorplay as tp

# The deferred clone lives in the operator namespace, like the rest of the
# internal `_`-prefixed surface.
lazy_clone = tp.ops.tp._lazy_clone


class LazyCloneCopyOnWriteTest(unittest.TestCase):
    """A deferred clone must behave like a copy the moment either side writes."""

    def devices(self):
        devices = ["cpu"]
        if tp.cuda.is_available():
            devices.append("cuda")
        return devices

    def make(self, device, shape=(3, 4)):
        values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
        return tp.tensor(values.tolist(), dtype=tp.float32, device=device), values

    def test_values_and_metadata_survive(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = lazy_clone(x)
                np.testing.assert_allclose(y.cpu().numpy(), values)
                self.assertEqual(tuple(y.shape), tuple(x.shape))
                self.assertEqual(y.dtype, x.dtype)
                self.assertEqual(y.stride(), x.stride())

    def test_clone_write_leaves_source(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = lazy_clone(x)
                y.add_(101.0)
                np.testing.assert_allclose(y.cpu().numpy(), values + 101.0)
                np.testing.assert_allclose(x.cpu().numpy(), values)

    def test_source_write_leaves_clone(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = lazy_clone(x)
                x.mul_(3.0)
                np.testing.assert_allclose(x.cpu().numpy(), values * 3.0)
                np.testing.assert_allclose(y.cpu().numpy(), values)

    def test_write_through_copy_and_fill(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = lazy_clone(x)
                y.copy_(y + 1.0)
                np.testing.assert_allclose(y.cpu().numpy(), values + 1.0)
                np.testing.assert_allclose(x.cpu().numpy(), values)
                z = lazy_clone(x)
                z.fill_(7.0)
                np.testing.assert_allclose(z.cpu().numpy(), np.full_like(values, 7.0))
                np.testing.assert_allclose(x.cpu().numpy(), values)

    def test_strided_source(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                base = x.t()
                y = lazy_clone(base)
                np.testing.assert_allclose(y.cpu().numpy(), values.T)
                self.assertEqual(y.stride(), base.stride())
                base.add_(5.0)
                np.testing.assert_allclose(y.cpu().numpy(), values.T)
                y.add_(1.0)
                np.testing.assert_allclose(y.cpu().numpy(), values.T + 1.0)
                np.testing.assert_allclose(x.cpu().numpy(), values + 5.0)

    def test_clone_of_clone(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = lazy_clone(x)
                z = lazy_clone(y)
                z.add_(1.0)
                np.testing.assert_allclose(z.cpu().numpy(), values + 1.0)
                np.testing.assert_allclose(y.cpu().numpy(), values)
                np.testing.assert_allclose(x.cpu().numpy(), values)

    def test_read_only_use_keeps_source_intact(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = lazy_clone(x)
                total = y.sum()
                np.testing.assert_allclose(float(total.cpu().numpy()) if hasattr(total, 'cpu') else float(total), values.sum())
                np.testing.assert_allclose(x.cpu().numpy(), values)

    def test_repeated_writes_only_pay_one_copy(self):
        for device in self.devices():
            with self.subTest(device=device):
                x, values = self.make(device)
                y = lazy_clone(x)
                for step in range(4):
                    y.add_(1.0)
                np.testing.assert_allclose(y.cpu().numpy(), values + 4.0)
                np.testing.assert_allclose(x.cpu().numpy(), values)


if __name__ == "__main__":
    unittest.main()