# Newcomer Hub

Looking for a first contribution? This page is the curated entry point: what to
pick, how to claim it, and what a good starter change looks like in TensorPlay.

Each starter task pairs a label with a curated index and a named maintainer,
so you always know what to read and who to ask.

## Pick a task

Two labels mark work that is ready for someone new:

| Label | Means | Typical size |
| --- | --- | --- |
| `good first issue` | scoped, names the files and a verification command | 1-2 files |
| `help wanted` | needs an extra pair of hands; may be larger | 1-5 files |

Browse them live:

- [good first issues](https://github.com/lexing-2026/TensorPlay/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- [help wanted](https://github.com/lexing-2026/TensorPlay/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)

Other labels you will see: `needs triage` (not checked by a maintainer yet),
`performance` (reports must carry measured numbers), `platform: windows|linux|macos`,
`kind/flaky`, and `release notes: *` (which subsystem a change touches).

## Claim a task

1. Comment "I'd like to take this" on the issue; a maintainer assigns it to you.
2. Ask for pointers if the issue is thin — a starter task should name the files,
   the build command and the acceptance check.
3. If the thread goes quiet for about two weeks, the task may be unassigned so
   someone else can pick it up.

## Curated tasks

| Issue | Area | Level | What you will learn |
| --- | --- | --- | --- |
| [#117](https://github.com/lexing-2026/TensorPlay/issues/117) | build, CPU kernels | `good first issue` | how the ISA tiers (`DEFAULT` / `AVX2` / `AVX512`) are compiled and selected, and why a Windows wheel can silently ship without any of them |
| [#124](https://github.com/lexing-2026/TensorPlay/issues/124) | loss kernels, CPU | `good first issue` | how a C++ kernel validates rank and dtype before it indexes memory, and why a missing int64 check turns a wrong tensor into an access violation |
| [#125](https://github.com/lexing-2026/TensorPlay/issues/125) | arithmetic, CPU | `good first issue` | how an in-place fast path can silently skip the write, and how a timing that looks too good exposes it |
| [#122](https://github.com/lexing-2026/TensorPlay/issues/122) | loss kernels, CPU | `good first issue` | how a row-wise loss is assembled from norms, and why a hand-written loop can lose 20x to the same math expressed with ordinary ops |
| [#120](https://github.com/lexing-2026/TensorPlay/issues/120) | vector math, CPU | `help wanted` | how the vector math layer gates its SIMD bodies behind compiler macros, and what a Windows build loses when it falls through |
| [#121](https://github.com/lexing-2026/TensorPlay/issues/121) | arithmetic / pointwise, CPU | `help wanted` | how 200 vector fast paths sit behind an arch macro, and what a Windows build loses when MSVC never matches it |
| [#131](https://github.com/lexing-2026/TensorPlay/issues/131) | coverage, CPU kernels | `help wanted` | how a kernel registration turns an unimplemented op into a real path, and how a dispatch probe separates a genuine gap from a backend that never ran on CPU |
| [#135](https://github.com/lexing-2026/TensorPlay/issues/135) | loss kernels, CPU | `good first issue` | how an op can return the right number in the wrong shape: mapping `nll_loss` onto the two-result forward kernel leaks a tuple to callers |

The list is mirrored in the pinned **Newcomer tasks** issue and grows as
maintainers triage new reports.

## What a good starter issue contains

- **Where** — the files or modules to touch (`p10/CMakeLists.txt`, `docs/...`).
- **How to see it** — a command that shows the problem, e.g.
  `python benchmark/bench_hotops_vs_torch.py --threads 8 --reps 7`.
- **Done when** — an observable check: a capability string, a timing number,
  or a passing test.

If one of the three is missing, say so in the thread. That is a maintainer gap,
not a gap in your skills.

## Build and test

```bash
pip install -e .                     # editable build
USE_CUDA=OFF pip install -e .        # CPU-only build
pytest test/                         # full suite
pytest test/test_tensor_methods.py   # a single module
```

Windows uses MSVC, Linux/macOS use GCC/Clang. Toolchain details, commit
conventions and the release-note labels live in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Mentoring

Every `good first issue` gets a maintainer watching the thread. Expect a first
reply within two working days; if the thread has gone quiet, ping it again.
