"""Regression tests for the conv2d backward-weights OpenMP team-size bug.

A oneDNN jit primitive bakes the OpenMP team size into the kernel it builds.
The CPU backward-weights path used to shrink that team only around
``execute()``, after the primitive had already been created for the full
team.  Two symptoms followed:

* only the first block of output channels reached ``diff_weights`` -- with
  more than sixteen output channels the remaining blocks stayed
  uninitialized (NaN / garbage in ``weight.grad``);
* once the spatial extent passed 17 the kernel split work across the team
  and its barrier waited for threads that never showed up, so
  ``backward()`` never returned at all (issue #113).

The shapes below pin both symptoms under the default thread configuration.
"""

import unittest

import numpy as np

import tensorplay as tp


def _grad_weight_reference(x, grad_out, kernel=3, padding=1):
    """dW[oc, ic, kh, kw] = sum over n and positions of grad_out * x patch."""
    n, c_in, h, w = x.shape
    c_out = grad_out.shape[1]
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding))).astype(np.float64)
    go = grad_out.astype(np.float64)
    out = np.zeros((c_out, c_in, kernel, kernel), dtype=np.float64)
    for oc in range(c_out):
        for ic in range(c_in):
            for kh in range(kernel):
                for kw in range(kernel):
                    out[oc, ic, kh, kw] = (
                        go[:, oc] * padded[:, ic, kh:kh + h, kw:kw + w]
                    ).sum()
    return out


class TestConv2dGradWeightTeam(unittest.TestCase):
    def _check_grad_weight(self, n, c_in, c_out, h, seed=0):
        rng = np.random.default_rng(seed)
        x_np = rng.random((n, c_in, h, h), dtype=np.float32)
        go_np = rng.random((n, c_out, h, h), dtype=np.float32)
        w_np = rng.random((c_out, c_in, 3, 3), dtype=np.float32)

        conv = tp.nn.Conv2d(c_in, c_out, 3, padding=1)
        conv.weight.data = tp.from_numpy(w_np)
        if conv.bias is not None:
            conv.bias.data = tp.from_numpy(np.zeros(c_out, dtype=np.float32))

        out = conv(tp.from_numpy(x_np))
        (out * tp.from_numpy(go_np)).sum().backward()

        grad = conv.weight.grad.numpy()
        self.assertTrue(np.isfinite(grad).all(), "grad_weight contains non-finite values")

        ref = _grad_weight_reference(x_np, go_np)
        np.testing.assert_allclose(grad, ref, rtol=1e-4, atol=1e-3)

    def test_single_batch_spatial_20(self):
        """Used to hang forever: only the first team member ever ran."""
        self._check_grad_weight(n=1, c_in=3, c_out=32, h=20)

    def test_single_batch_spatial_32(self):
        self._check_grad_weight(n=1, c_in=3, c_out=32, h=32)

    def test_second_channel_block_is_written(self):
        """More than sixteen output channels: the second block stayed uninitialized."""
        self._check_grad_weight(n=2, c_in=3, c_out=48, h=12)

    def test_full_team_path(self):
        """Batch large enough to keep the whole team: the path that was healthy."""
        self._check_grad_weight(n=32, c_in=3, c_out=16, h=18)


if __name__ == "__main__":
    unittest.main()
