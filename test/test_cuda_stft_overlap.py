import numpy as np
import pytest
import torch
import tensorplay as tp

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('n_fft,hop', [(7, 1), (8, 3), (16, 16), (32, 5)])
@pytest.mark.parametrize('center', [False, True])
@pytest.mark.parametrize('normalized', [False, True])
def test_stft_overlapping_frames(dtype, n_fft, hop, center, normalized):
    data = np.random.default_rng(15).normal(size=79).astype(getattr(np, dtype))
    actual_input = tp.tensor(data, device='cuda')
    expected_input = torch.tensor(data, device='cuda')
    actual = tp.stft(actual_input, n_fft, hop, center=center, normalized=normalized, return_complex=True)
    expected = torch.stft(expected_input, n_fft, hop, center=center, normalized=normalized, return_complex=True)
    np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual_input.cpu().numpy(), data)


def test_stft_plan_layout_switches_and_real_output():
    data = np.random.default_rng(21).normal(size=(1, 96)).astype(np.float32)
    actual_input, expected_input = tp.tensor(data, device='cuda'), torch.tensor(data, device='cuda')
    for n_fft, hop in [(16, 4), (16, 8), (12, 5), (16, 4)]:
        actual = tp.stft(actual_input, n_fft, hop, return_complex=False)
        expected = torch.view_as_real(torch.stft(expected_input, n_fft, hop, return_complex=True))
        np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(), rtol=2e-5, atol=2e-6)


def test_stft_batched_real_output():
    data = np.random.default_rng(22).normal(size=(2, 64)).astype(np.float32)
    actual = tp.stft(tp.tensor(data, device='cuda'), 16, 4, return_complex=False)
    expected = torch.view_as_real(torch.stft(torch.tensor(data, device='cuda'), 16, 4, return_complex=True))
    np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(), rtol=2e-5, atol=2e-6)
