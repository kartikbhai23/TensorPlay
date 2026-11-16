"""The public embedding_bag spellings dispatch to the native CPU kernels.

Two overloads reach the same forward: the default one pins the padding
sentinel, the padding one takes the index explicitly (absent meaning the same
sentinel).  Compilation captures the public spelling and must answer with the
same numbers the eager call produces.
"""

import unittest

import tensorplay as tp


def _reference_embedding_bag(weight, indices, offsets, padding_idx=-1, mode=0):
    bags = []
    for b in range(offsets.numel()):
        start = int(offsets[b])
        end = int(offsets[b + 1]) if b + 1 < offsets.numel() else indices.numel()
        bag = [int(indices[i]) for i in range(start, end)]
        bag = [r for r in bag if r != padding_idx]
        if not bag:
            bags.append(tp.zeros((weight.shape[1],), dtype=weight.dtype))
            continue
        picked = tp.stack([weight[r] for r in bag], 0)
        if mode == 1:
            bags.append(picked.sum(0) / picked.shape[0])
        else:
            bags.append(picked.sum(0))
    return tp.stack(bags, 0)


class TestEmbeddingBagCPU(unittest.TestCase):
    def setUp(self):
        self.weight = tp.randn(6, 3)
        self.indices = tp.tensor([0, 1, 2, 2, 5, 4])
        self.offsets = tp.tensor([0, 2, 5])

    def test_default_overload_matches_reference(self):
        out = tp.embedding_bag(self.weight, self.indices, self.offsets,
                               False, 0, False, None, False)
        expected = _reference_embedding_bag(self.weight, self.indices, self.offsets)
        self.assertTrue(tp.allclose(out[0], expected))

    def test_padding_overload_skips_the_sentinel_row(self):
        out = tp.embedding_bag(self.weight, self.indices, self.offsets,
                               False, 0, False, None, False, 5)
        expected = _reference_embedding_bag(self.weight, self.indices,
                                            self.offsets, padding_idx=5)
        self.assertTrue(tp.allclose(out[0], expected))

    def test_mean_mode_matches_reference(self):
        out = tp.embedding_bag(self.weight, self.indices, self.offsets,
                               False, 1, False, None, False)
        expected = _reference_embedding_bag(self.weight, self.indices,
                                            self.offsets, mode=1)
        self.assertTrue(tp.allclose(out[0], expected))

    def test_compiled_public_spelling_matches_eager(self):
        def fn(w, i, o):
            return tp.embedding_bag(w, i, o, False, 0, False, None, False, 5)

        eager = fn(self.weight, self.indices, self.offsets)
        compiled = tp.compile(fn, backend="stax")
        result = compiled(self.weight, self.indices, self.offsets)
        self.assertTrue(tp.allclose(result[0], eager[0]))


if __name__ == "__main__":
    unittest.main()
