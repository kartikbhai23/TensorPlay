"""median/nanmedian along a dim, comparing with the reference.

A long single-slice reduction takes the ordered-values path: the multi-block
radix sort orders the values and the middle one is read out of the result.
Shorter cuts, and cuts with several slices, keep the per-slice selector.  Both
are checked here, over NaN input, repeated values, integral dtypes, keepdim,
and slices that are strided within the row.

The reported position is the one this library documents, the smallest position
holding the selected value, where the ordered-values path answers.  The
reference reaches its position through an order statistic on the CPU and
through a race between the lanes that find a match on the GPU, so positions are
compared only on the CPU.
"""
import numpy as np
import torch

import tensorplay as tp

LONG = 1 << 19
SHORT = 1 << 12


def _tensor(np_array, device):
    t = tp.from_dlpack(torch.from_numpy(np_array.copy()))
    return t.to(device) if device != "cpu" else t


def _holds(array, value):
    """Positions of `array` that hold `value`, with NaN matching NaN."""
    if np.issubdtype(np.asarray(value).dtype, np.floating) and np.isnan(value):
        return np.isnan(array)
    return array == value


def _check(name, data, dim, device, nanmedian=False, keepdim=False):
    ref = torch.from_numpy(data.copy())
    got = _tensor(data, device)
    if device == "cuda":
        ref = ref.cuda()
    with torch.no_grad():
        want_v, want_i = (torch.nanmedian(ref, dim, keepdim=keepdim)
                          if nanmedian else torch.median(ref, dim, keepdim=keepdim))
        have_v, have_i = (tp.nanmedian(got, dim, keepdim=keepdim)
                          if nanmedian else tp.median(got, dim, keepdim=keepdim))
    want_v = want_v.detach().cpu().numpy()
    have_v = have_v.detach().cpu().numpy()
    want_i = want_i.detach().cpu().numpy()
    have_i = have_i.detach().cpu().numpy()
    assert want_v.shape == have_v.shape, (name, want_v.shape, have_v.shape)
    assert np.allclose(want_v, have_v, equal_nan=True, rtol=0, atol=0), (
        name, want_v.ravel()[:8], have_v.ravel()[:8])

    moved = np.moveaxis(data, dim, -1).reshape(-1, data.shape[dim])
    # A slice of nothing but NaNs has no value, so no position can be the one
    # that holds it and the reference leaves that position open as well.
    if nanmedian:
        comparable = ~np.isnan(moved).all(axis=1)
    else:
        comparable = np.ones(moved.shape[0], dtype=bool)
    if device == "cpu":
        assert np.array_equal(want_i.ravel()[comparable],
                              have_i.ravel()[comparable]), (
            name, want_i.ravel()[:8], have_i.ravel()[:8])

    # The reported position must hold the reported value, and on the ordered
    # path it must be the first position that does.
    for si in range(moved.shape[0]):
        value = have_v.reshape(-1)[si]
        index = have_i.reshape(-1)[si]
        assert 0 <= index < moved.shape[1], (name, si, index)
        hits = _holds(moved[si], value)
        assert hits[index], (name, si, index, moved[si, index], value)
        if device == "cuda":
            earlier = hits[:index]
            assert not earlier.any(), (
                f"{name}: slice {si} reports {index} but the first position "
                f"holding {value} is {np.flatnonzero(earlier)[0]}")


def test_median_long_single_row():
    rs = np.random.RandomState(0)
    for device in ("cpu", "cuda"):
        _check("float32", rs.standard_normal(LONG).astype(np.float32), -1, device)
        _check("float64", rs.standard_normal(LONG), -1, device)
        _check("int64", rs.randint(-50, 50, LONG).astype(np.int64), -1, device)
        _check("int32", rs.randint(-50, 50, LONG).astype(np.int32), -1, device)
        _check("dups", np.zeros(LONG, dtype=np.float32), -1, device)


def test_median_long_with_nan():
    rs = np.random.RandomState(1)
    data = rs.standard_normal(LONG).astype(np.float32)
    data[::97] = np.nan
    for device in ("cpu", "cuda"):
        _check("nan-median", data, -1, device)
        _check("nan-nanmedian", data, -1, device, nanmedian=True)


def test_median_long_all_nan():
    data = np.full(LONG, np.nan, dtype=np.float32)
    for device in ("cpu", "cuda"):
        _check("all-nan-median", data, -1, device)
        # A slice of nothing but NaNs has no value, so no position can be the
        # one that holds it; only the value is settled there.
        _check("all-nan-nanmedian", data, -1, device, nanmedian=True)


def test_median_single_row_keepdim():
    rs = np.random.RandomState(3)
    _check("keepdim", rs.standard_normal((1, LONG)).astype(np.float32), 1,
           "cuda", keepdim=True)


def test_median_strided_slices():
    rs = np.random.RandomState(2)
    # inner = 4: each slice is a strided column of a row-major array.  Several
    # slices keep the per-slice selector even when each of them is long.
    data = rs.standard_normal((LONG, 4)).astype(np.float32)
    for device in ("cpu", "cuda"):
        _check("dim0", data, 0, device)
        _check("dim0-keepdim", data, 0, device, keepdim=True)
    data2 = rs.standard_normal((4, LONG)).astype(np.float32)
    for device in ("cpu", "cuda"):
        _check("dim1", data2, 1, device)


def test_median_short_slice_path():
    # Under the length gate the single-block selector answers; both agree.
    rs = np.random.RandomState(4)
    data = rs.standard_normal((8, SHORT)).astype(np.float32)
    for device in ("cpu", "cuda"):
        _check("short", data, 1, device)
    _check("short-1d", rs.standard_normal(SHORT).astype(np.float32), -1, "cuda")


def test_median_short_with_nan():
    rs = np.random.RandomState(5)
    data = rs.standard_normal((4, SHORT)).astype(np.float32)
    data[0, 3] = np.nan
    data[2, 100] = np.nan
    data[3, :] = np.nan
    for device in ("cpu", "cuda"):
        _check("short-nan", data, 1, device)
        _check("short-nan-nanmedian", data, 1, device, nanmedian=True)