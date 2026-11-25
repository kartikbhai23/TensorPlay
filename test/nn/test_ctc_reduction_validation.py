import pytest
import tensorplay as tp
import tensorplay.nn.functional as F


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA required"))])
def test_ctc_reduction_validation(device):
    log_probs = tp.randn(6, 2, 4, device=device).log_softmax(2)
    targets = tp.tensor([[1, 2], [2, 1]], device=device)
    args = (log_probs, targets, [6, 6], [2, 2])
    with pytest.raises(ValueError, match="not a valid value for reduction"):
        F.ctc_loss(*args, reduction="invalid")
    with pytest.raises(ValueError, match="not a valid value for reduction"):
        F.ctc_loss(*args, reduction="elementwise_mean")
