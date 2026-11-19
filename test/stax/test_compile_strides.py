import pytest

import tensorplay as tp


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if tp.cuda.is_available() else []))
def test_expanded_input_capture_preserves_values_and_mutations(device):
    base = tp.arange(6, dtype=tp.float32, device=device).reshape(2, 3)
    expanded = base.unsqueeze(0).expand(4, 2, 3)
    before = base.clone()
    compiled = tp.compile(lambda x: x * 2, fullgraph=True, strict_native=True)
    assert tp.equal(compiled(expanded), expanded * 2)
    assert tp.equal(base, before)

    def update_first(x):
        x[0].add_(1)
        return x * 2

    with pytest.raises(RuntimeError, match="overlapping storage"):
        tp.compile(update_first, fullgraph=True, strict_native=True)(expanded)
    assert tp.equal(base, before)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_square_layout_changes_recompile(dynamic, requires_grad):
    value = tp.randn(4, 4, requires_grad=requires_grad)
    expanded = tp.randn(4, 1, requires_grad=requires_grad).expand(4, 4)
    sliced = tp.randn(4, 8, requires_grad=requires_grad)[:, ::2]
    compiled = tp.compile(lambda x: x.sin() * 2, dynamic=dynamic, fullgraph=True)

    for operand in (value, value.t(), expanded, sliced):
        result = compiled(operand)
        assert tp.allclose(result, operand.sin() * 2)
        if requires_grad:
            gradient, = tp.autograd.grad(result.sum(), operand)
            assert tp.allclose(gradient, operand.cos() * 2)
    assert len(compiled._tensorplay_cache) == 4
    for operand in (value.t(), value, sliced, expanded):
        assert tp.allclose(compiled(operand), operand.sin() * 2)
    assert len(compiled._tensorplay_cache) == 4


def test_dynamic_layouts_reuse_specializations_across_sizes():
    compiled = tp.compile(lambda x: x.sin() + 1, dynamic=True, fullgraph=True)
    for size in (4, 6, 8):
        value = tp.randn(size, size)
        for operand in (value, value.t()):
            # ``sin(x) + 1`` approaches zero where ``sin(x)`` approaches -1.
            # There the default relative bound is tighter than the one-ulp
            # difference between the fused and eager sine, so compare with an
            # absolute tolerance that matches the rest of the suite.
            assert tp.allclose(
                compiled(operand), operand.sin() + 1, atol=1e-5, rtol=1e-5
            )
    assert len(compiled._tensorplay_cache) == 2


@pytest.mark.parametrize("dynamic", [False, True])
def test_nested_keyword_tensor_layouts_recompile(dynamic):
    compiled = tp.compile(lambda *, batch: batch["x"].cos(), dynamic=dynamic)
    value = tp.randn(4, 4)
    for operand in (value, value.t(), value):
        assert tp.allclose(compiled(batch={"x": operand}), operand.cos())
    assert len(compiled._tensorplay_cache) == 2
