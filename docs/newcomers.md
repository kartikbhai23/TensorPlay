# Newcomer Hub

Looking for a first contribution? This page is the curated entry point: what to
pick, how to claim it, and what a good starter change looks like in TensorPlay.

The conventions here follow what established projects do for first-time
contributors — PyTorch's
[good first issues](https://github.com/pytorch/pytorch/contribute),
Kubernetes' [new contributor guide](https://github.com/kubernetes/community/blob/master/contributors/guide/README.md)
and Rust's [mentor program](https://rust-lang.org/contribute/) all pair a
label with a curated index and a named maintainer per task.

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

| Issue | Area | What you will learn |
| --- | --- | --- |
| [#117](https://github.com/lexing-2026/TensorPlay/issues/117) | build, CPU kernels | how the ISA tiers (`DEFAULT` / `AVX2` / `AVX512`) are compiled and selected, and why a Windows wheel can silently ship without any of them |

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
