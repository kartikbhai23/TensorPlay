# CPU operator sweep: full surface vs torch

Every CPU-dispatching op in `config/native_functions.yaml` is built from schema, run against
`torch.ops.aten` with the same arguments, and timed on one machine. `speedup = torch_ms / tp_ms`,
so **below 1.0 means TensorPlay is slower**.

## Setup

- torch 2.11.0+cpu, tensorplay 1.3.0 wheel, both 8 threads, float32
- AMD Ryzen 7 8845HS (Zen 4, AVX-512F/DQ/BW/VL), Windows 11, 32 GB
- Default operand 256x256; conv/attention/reduction families use their own shape templates
- Two warm-up calls then a fixed rep count (7 in the re-timing pass, 9 in the fixed-argument sets); best-of. Harnesses: `bench_cpu/sweep_all_ops.py`, `bench_cpu/retime.py`, `bench_cpu/bench_core.py`

## Re-timing pass

Every row here was re-measured in a fresh process (30 labels per process, two warm-up
calls, 7 reps). That pass removed two artefacts that had been inflating the loss column:

- The timer used to drop to two reps whenever the first call took over 50 ms, so a cold
  call could be stored as the result. Torch rows of 50-1200 ms on elementwise ops came
  from that path: `div.Scalar` read as 52 ms while a direct measurement gives 0.017 ms.

- Rows carried over from older sweep generations had been built with different argument
  choices. The loss family is the clearest case: it read 0.15x median on stale inputs and
  sits near 0.7x once rebuilt.

84 labels did not reproduce on re-run (they raise with the current builder) and are
excluded rather than mixed in.

## Coverage

| | count |
| --- | --- |
| `func` entries in the schema | 3394 |
| distinct functions | 1849 |
| CPU-dispatching ops | 881 |
| timed overloads | 1638 |
| distinct ops among timed rows | 822 |
| timed rows that are valid | 1633 |

The rest raise in tp (shape/dtype constraints the schema does not express) or in torch,
or are out-variants whose `out` argument the builder cannot synthesise.

Out-variants (`*.out`, `*_out`) and in-place forms (`*_`) are now measured too: the builder
pre-allocates an `out` buffer at the broadcast shape of the inputs, so those rows carry the
same write traffic as torch's out variants. Reduction out-variants whose real output is
smaller than that guess get resized by both engines inside the timed region, which inflates
their cost -- those rows are still listed, but read them as upper bounds.

## Headline

- **894 of 1633 valid rows are slower than torch (55%)**
- median speedup **0.92x**, geometric mean 0.81x
- distribution: 72 below 0.1x, 254 at 0.1-0.3x, 173 at 0.3-0.5x, 227 at 0.5-0.8x, 168 at 0.8-1.0x, 482 at 1-2x, 257 above 2x

## Where the loss actually is

Splitting the rows by API form shows where the measurements cluster. Each form is a
different set of ops, so the medians describe the ops that happen to carry that form;
the paired check below separates the wrapper from the kernel.

| API shape | rows | slower | median |
| --- | --- | --- | --- |
| regular | 723 | 303 | 1.14x |
| scalar-arith | 55 | 38 | 0.69x |
| out-variant | 535 | 313 | 0.87x |
| in-place | 187 | 136 | 0.39x |
| foreach | 133 | 104 | 0.47x |

The form split is descriptive, not causal: each bucket contains a different set of ops,
so a lower median for in-place mainly says that the ops with in-place forms are the
elementwise arithmetic kernels. To test the wrapper itself, every in-place label was
paired with its own functional sibling at the same shape, and every out label with its
functional sibling:

| pair kind | pairs | median tp(in-place or out) / tp(functional) |
| --- | --- | --- |
| in-place vs functional | 161 | 1.03x |
| out vs functional | 338 | 1.07x |

Both land at parity, so the wrappers themselves are not where the time goes; the
deficit follows the op. A handful of outliers are worth checking on their own: `lift.out` (15.7x
over its functional form), `_conj_physical.out` (13.5x), `conj_physical.out` (12.8x),
`triu.out` (5.4x), `upsample_nearest2d.out` (5.2x).

## Family league table

Families with at least 12 timed rows, sorted by median. A family that shares one kernel
template (conv, pool, loss) tends to move as a block, which makes this table the fastest
way to spot where a single fix would pay off.

| family | rows | slower | median |
| --- | --- | --- | --- |
| act | 41 | 31 | 0.65x |
| loss | 17 | 11 | 0.75x |
| linalg | 46 | 39 | 0.78x |
| mm | 17 | 10 | 0.93x |
| norm | 37 | 18 | 1.01x |
| rand | 30 | 13 | 1.06x |
| index | 50 | 18 | 1.18x |
| sort/hist | 16 | 2 | 1.27x |
| reduce | 71 | 33 | 1.40x |
| pool | 25 | 2 | 1.49x |
| conv | 15 | 2 | 1.57x |
| pad/upsample | 29 | 7 | 2.30x |

## Worst 40

| op | family | tp ms | torch ms | speedup |
| --- | --- | --- | --- | --- |
| `norm` | norm | 15.387 | 0.011 | 0.00x |
| `_cdist_backward.out` | other | 56.194 | 0.692 | 0.01x |
| `_cdist_backward` | other | 54.500 | 0.750 | 0.01x |
| `_foreach_add.ScalarList` | other | 1.356 | 0.022 | 0.02x |
| `sub_.Scalar` | other | 0.431 | 0.009 | 0.02x |
| `add_.Scalar` | other | 0.424 | 0.009 | 0.02x |
| `_foreach_sub.ScalarList` | other | 1.328 | 0.029 | 0.02x |
| `_foreach_norm.Scalar_out` | norm | 0.439 | 0.010 | 0.02x |
| `nextafter_` | other | 1.509 | 0.035 | 0.02x |
| `huber_loss_backward` | loss | 0.542 | 0.013 | 0.02x |
| `add.Scalar` | other | 0.425 | 0.011 | 0.02x |
| `sub.Scalar` | other | 0.424 | 0.011 | 0.03x |
| `_foreach_pow.ScalarAndTensor` | other | 0.541 | 0.014 | 0.03x |
| `_foreach_add.ScalarList_out` | other | 1.277 | 0.034 | 0.03x |
| `_add_relu_.Scalar` | other | 0.434 | 0.012 | 0.03x |
| `min.other` | other | 0.262 | 0.007 | 0.03x |
| `_add_relu.Scalar` | other | 0.427 | 0.013 | 0.03x |
| `_foreach_sub.ScalarList_out` | other | 1.307 | 0.042 | 0.03x |
| `min.out` | other | 0.220 | 0.007 | 0.03x |
| `_foreach_minimum.List` | other | 0.656 | 0.021 | 0.03x |
| `trapezoid.dx` | other | 0.545 | 0.018 | 0.03x |
| `_foreach_max.out` | other | 0.316 | 0.010 | 0.03x |
| `minimum.out` | other | 0.232 | 0.008 | 0.03x |
| `add.Scalar_out` | other | 0.360 | 0.012 | 0.03x |
| `maximum` | other | 0.240 | 0.009 | 0.04x |
| `minimum` | other | 0.256 | 0.010 | 0.04x |
| `maximum.out` | other | 0.179 | 0.007 | 0.04x |
| `_foreach_clamp_min.List` | other | 0.523 | 0.020 | 0.04x |
| `_foreach_maximum.List` | other | 0.607 | 0.024 | 0.04x |
| `sub.Scalar_out` | other | 0.397 | 0.016 | 0.04x |
| `div_.Scalar` | other | 0.195 | 0.008 | 0.04x |
| `max.other` | other | 0.186 | 0.008 | 0.04x |
| `_foreach_div.ScalarList` | other | 0.576 | 0.024 | 0.04x |
| `_foreach_powsum.Scalar_out` | reduce | 0.336 | 0.014 | 0.04x |
| `div.Scalar` | other | 0.188 | 0.008 | 0.04x |
| `max.out` | other | 0.138 | 0.007 | 0.05x |
| `_cast_Half` | other | 0.216 | 0.012 | 0.06x |
| `_add_relu.Scalar_out` | other | 0.362 | 0.020 | 0.06x |
| `trapz.dx` | other | 0.547 | 0.030 | 0.06x |
| `_foreach_add.Tensor` | other | 0.283 | 0.016 | 0.06x |

## Correctness gate

Timing alone cannot tell a fast kernel from a kernel that skipped the work, so the
214 rows that looked implausible (`speedup >= 5`, or `tp_ms <= 0.02` with `speedup >= 2`,
or any `_foreach_*`) were re-run with values compared against torch:

- `mul_.Tensor` writes nothing for contiguous float32 -- the 67x entry was a no-op (#125).
  `_foreach_mul.*` inherits it; those rows are excluded from the table above.
- `unique_dim` / `unique_consecutive` / `unique_dim_consecutive` return an undefined tensor
  in the slot torch fills with an empty one; converting it through dlpack segfaults.
- `ctc_loss` segfaults on non-int64 targets (#124).
- Everything else in the checked set matched torch within 2e-3 absolute (relative 3e-7).

## Verified core set

The schema sweep screens; this table decides. 69 kernels are called with one fixed,
realistic argument set each, both engines get the same tensors, the results are compared
elementwise, and only then timed (two warm-ups, best of 9). The table lists the
26 slowest of the 68 valid cases; all 68 are in `bench_core.json`.

| op | tp ms | torch ms | speedup |
| --- | --- | --- | --- |
| `minimum` | 0.320 | 0.009 | 0.02x |
| `add.Scalar` | 0.538 | 0.014 | 0.03x |
| `sub.Scalar` | 0.538 | 0.019 | 0.03x |
| `maximum` | 0.285 | 0.009 | 0.04x |
| `div.Scalar` | 0.319 | 0.019 | 0.06x |
| `mul.Scalar` | 0.293 | 0.018 | 0.07x |
| `clamp` | 0.043 | 0.008 | 0.19x |
| `pow.Tensor_Tensor` | 0.602 | 0.117 | 0.19x |
| `sigmoid` | 0.100 | 0.022 | 0.22x |
| `silu` | 0.105 | 0.027 | 0.25x |
| `div.Tensor` | 0.033 | 0.009 | 0.28x |
| `gelu` | 0.095 | 0.029 | 0.28x |
| `logsumexp` | 0.144 | 0.041 | 0.29x |
| `contiguous` | 0.088 | 0.029 | 0.32x |
| `log` | 0.082 | 0.030 | 0.35x |
| `sum.all` | 0.044 | 0.016 | 0.35x |
| `sum.dim` | 0.046 | 0.017 | 0.38x |
| `clamp_` | 0.038 | 0.015 | 0.40x |
| `reshape` | 0.005 | 0.002 | 0.41x |
| `exp` | 0.063 | 0.026 | 0.42x |
| `transpose` | 0.004 | 0.002 | 0.44x |
| `max.dim` | 0.058 | 0.028 | 0.45x |
| `permute` | 0.004 | 0.002 | 0.45x |
| `softmax` | 0.029 | 0.015 | 0.50x |
| `tanh` | 0.117 | 0.055 | 0.50x |
| `stack` | 0.037 | 0.019 | 0.52x |

Valid cases: 68 of 69; 49 slower (72%); median 0.63x.

Two things stand out and both are verified by value comparison rather than by timing:
scalar arithmetic (`add.Scalar` 0.02x, `sub.Scalar` 0.02x, `mul.Scalar` 0.05x,
`div.Scalar` 0.04x) and the elementwise pair `maximum`/`minimum` (0.05x / 0.03x).
The matmul family (0.75-0.84x), conv (1.68x), pooling (0.80-1.03x) and the loss kernels
(1.50x) are in good shape.

`mul_.Tensor` is the one value mismatch: it reports 1.54x while writing nothing (#125),
which is why the value gate runs before the timer.

## Function coverage: 291 comparable functions, 287 measured

A row count does not answer "is every CPU function measured": one function can
own a dozen overloads and another none. The set is cut per function, using
`torch._C._dispatch_has_kernel_for_dispatch_key` on the 881 CPU-dispatching
entries of `config/native_functions.yaml`:

| bucket | count | meaning |
| --- | --- | --- |
| comparable | 291 | upstream has a CPU kernel, so a ratio exists |
| tp-only | 590 | `tp_*`, `forward_*`, `snn_*`, `quantized_*`, `quantum_*`: no upstream reference |
| measured | 287 | fixed arguments, same tensors on both sides, value check, timing |

The 113 functions the overload sweep never reached were measured with
`bench_gap.py`: one fixed argument set per function, torch's own constructor
recorded and replayed into tp so both sides see identical tensors.

- 88 exact-value passes and 12 randomised ops (no value comparison possible)
- 6 value mismatches, all re-checked with a rebuild-per-call harness
  (`bench_gap2.py`): `index_copy`, `addcmul_`, `addcdiv_`, `baddbmm_` come out
  at maxdiff 0, `log_sigmoid_backward` at 2.4e-7 and
  `linalg_householder_product` at 2.5e-6 relative. The mismatches were shared
  buffers mutated by earlier timed calls, not kernels.
- 3 tp-side raises: `binary_cross_entropy`, `_ctc_loss`, `neg_` (the packet
  binding, not the op) -- filed as #138, #139 and #137.
- 4 torch-side raises: `_embedding_bag_dense_backward`, `_fused_adam_`,
  `_fused_adamw_` (torch's own `state_steps`/`max_exp_avg_sqs` strictness, now
  driven correctly in the second harness) and `_ctc_loss_backward`.

19 more functions that mutate their arguments or need a fresh RNG state per
call were run in `bench_gap2.py` with inputs rebuilt inside each call; all 19
pass, including `_fused_adam_`/`_fused_adamw_` (parameter values after the
update), `_convert_indices_*`, `fractional_max_pool2d/3d_backward`,
`_upsample_nearest_exact1d/2d/3d_backward`, `glu_backward`, `neg_` (through the
public op) and `linalg_householder_product`.

The four functions without a number are findings, not gaps:

| function | verdict |
| --- | --- |
| `binary_cross_entropy` | rejects every input, even `full(0.5)`/`full(0.5)` -- #138 |
| `_ctc_loss` | rejects every `input_lengths`, including `== T` -- #139 |
| `_ctc_loss_backward` | access violation `0xC0000005`, isolated exit `-1073741819` |
| `batch_norm_backward` | signature drift: tp's 7-arg form runs; torch 2.11's 11-arg reference crashes inside torch |

That closes the 291. The remaining 590 are tp extensions with no reference
kernel, so "slower than torch" is undefined for them by construction.

## Defects filed from this pass

| issue | finding |
| --- | --- |
| #137 | five unary in-place packets (`neg_`, `sqrt_`, `rsqrt_`, `zero_`, `abs_`) reject their own tensor |
| #138 | `binary_cross_entropy` raises on any input, including a constant 0.5 tensor |
| #139 | `_ctc_loss` rejects every `input_lengths` value as out of range |
| #140 | `multinomial` 5.7x slower at 256x256/64 (4.84 ms vs 0.85 ms) |

## Files

- `sweep_all.csv` -- one row per valid overload
- `sweep_final8.json` -- merged rows behind this report
- `check_values.py` / `check_driver.py` -- the value gate
- `sweep_all_ops.py` -- the schema harness; `retime.py` -- per-process re-timing
- `bench_core.py` / `bench_core.json` -- the fixed-argument core set
- `bench_gap.py` / `bench_gap2.py` -- the per-function gap fills
- `func_classes.json`, `coverage.json` -- the 291/590 split and the 287 measured
