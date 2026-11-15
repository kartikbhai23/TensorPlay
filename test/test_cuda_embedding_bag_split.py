"""CUDA bag reductions across bag lengths and embedding widths.

The forward route divides each bag's positions over several warps and covers a
tile of the embedding features per block, so the interesting cases are bags of
one position, bags longer than any split, empty bags, and widths that do not
divide the warp.  Everything is cross-checked against the CPU backend, which
is covered against the reference framework elsewhere.
"""

import numpy as np
import pytest
import torch

import tensorplay as tp
import tensorplay.nn.functional as F
import tensorplay._C as _C


pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")

SUM, MEAN, MAX = 0, 1, 2


def _tensor(array, cuda):
    t = tp.tensor(np.ascontiguousarray(array))
    return t.to(tp.device("cuda")) if cuda else t


def _np(t):
    return np.asarray(t.to(tp.device("cpu")).tolist(), dtype=np.float64)


def _ints(t):
    return t.to(tp.device("cpu")).tolist()


def _compare(host, dev, msg):
    np.testing.assert_allclose(_np(dev[0]), _np(host[0]), rtol=1e-5, atol=1e-6, err_msg=msg + " output")
    assert _ints(dev[1]) == _ints(host[1]), msg + " offset2bag"
    assert _ints(dev[2]) == _ints(host[2]), msg + " bag_size"
    assert _np(dev[3]).shape == _np(host[3]).shape, msg + " max_indices shape"
    assert _ints(dev[3]) == _ints(host[3]), msg + " max_indices"


def _run(weight, indices, offsets, mode, padding_idx=-1, include_last_offset=False,
         per_sample_weights=None):
    kwargs = (False, mode, False, per_sample_weights, include_last_offset, padding_idx)
    host = _C._embedding_bag(_tensor(weight, False), _tensor(indices, False),
                             _tensor(offsets, False), *kwargs)
    dev = _C._embedding_bag(_tensor(weight, True), _tensor(indices, True),
                            _tensor(offsets, True), *kwargs)
    return host, dev


@pytest.mark.parametrize("mode", [SUM, MEAN, MAX])
@pytest.mark.parametrize("width", [1, 7, 32, 33, 64, 129])
def test_widths_that_do_not_fill_a_tile(mode, width):
    rng = np.random.RandomState(31)
    weight = rng.randn(23, width).astype(np.float32)
    indices = rng.randint(0, 23, 37).astype(np.int64)
    offsets = np.array([0, 5, 5, 12, 37], dtype=np.int64)
    host, dev = _run(weight, indices, offsets, mode)
    _compare(host, dev, f"mode={mode} width={width}")


@pytest.mark.parametrize("mode", [SUM, MEAN])
def test_bags_longer_than_a_split(mode):
    rng = np.random.RandomState(32)
    weight = rng.randn(97, 40).astype(np.float32)
    indices = rng.randint(0, 97, 1000).astype(np.int64)
    offsets = np.array([0, 1, 999, 1000], dtype=np.int64)
    host, dev = _run(weight, indices, offsets, mode)
    _compare(host, dev, f"long bags mode={mode}")


@pytest.mark.parametrize("mode", [SUM, MEAN, MAX])
def test_single_position_and_empty_bags(mode):
    weight = np.arange(5 * 9, dtype=np.float32).reshape(5, 9) * 0.25
    indices = np.array([2, 4, 4, 0], dtype=np.int64)
    offsets = np.array([0, 1, 1, 3, 3, 4], dtype=np.int64)
    host, dev = _run(weight, indices, offsets, mode)
    _compare(host, dev, f"sparse bags mode={mode}")


@pytest.mark.parametrize("mode", [SUM, MEAN])
def test_padding_and_include_last_offset(mode):
    rng = np.random.RandomState(33)
    weight = rng.randn(11, 33).astype(np.float32)
    indices = np.array([3, 3, 7, 3, 9, 3, 3], dtype=np.int64)
    offsets = np.array([0, 3, 5, 7], dtype=np.int64)
    host, dev = _run(weight, indices, offsets, mode, padding_idx=3, include_last_offset=True)
    _compare(host, dev, f"padding mode={mode}")


def test_per_sample_weights_sum():
    rng = np.random.RandomState(34)
    weight = rng.randn(17, 40).astype(np.float32)
    indices = rng.randint(0, 17, 64).astype(np.int64)
    offsets = np.array([0, 4, 63, 64], dtype=np.int64)
    psw = rng.randn(64).astype(np.float32)
    host, dev = _run(weight, indices, offsets, SUM, per_sample_weights=_tensor(psw, True))
    _compare(host, dev, "per_sample_weights")


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_functional_two_dimensional_input(mode):
    rng = np.random.RandomState(35)
    weight = rng.randn(29, 24).astype(np.float32)
    indices = rng.randint(0, 29, (6, 5)).astype(np.int64)
    expected = F.embedding_bag(_tensor(indices, True), _tensor(weight, True), mode=mode)
    wanted = torch.nn.functional.embedding_bag(
        torch.tensor(indices), torch.tensor(weight), mode=mode)
    np.testing.assert_allclose(_np(expected), wanted.detach().numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_forward_only_matches_forward(mode):
    rng = np.random.RandomState(36)
    weight = rng.randn(13, 48).astype(np.float32)
    indices = rng.randint(0, 13, 200).astype(np.int64)
    offsets = np.array([0, 7, 200], dtype=np.int64)
    args = (False, 0 if mode == "sum" else 1, False, None, False, -1)
    full = _C._embedding_bag(_tensor(weight, True), _tensor(indices, True),
                             _tensor(offsets, True), *args)
    only = _C._embedding_bag_forward_only(_tensor(weight, True), _tensor(indices, True),
                                          _tensor(offsets, True), *args)
    for part in (0, 1, 2, 3):
        np.testing.assert_array_equal(_np(only[part]), _np(full[part]))