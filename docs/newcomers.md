# Newcomer Hub

Looking for a first contribution? This page is the curated entry point: what to
pick, how to claim it, and what a good starter change looks like in TensorPlay.

Every starter task is scoped to one or two files, names the files to touch,
ships a command that shows the problem, and states an observable acceptance
check. A maintainer watches each starter thread.

## Start here

1. **Get the code running.** Follow [Set up your
   environment](#set-up-your-environment) and run one test module. Most
   starter tasks do not need CUDA or a GPU.
2. **Pick a task.** Browse the open [good first
   issues](https://github.com/lexing-2026/TensorPlay/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22+no%3Aassignee)
   or the [curated list](#curated-tasks) below.
3. **Say hello.** Comment on the issue with the command you ran and what you
   plan to change. A maintainer assigns the task to you before you invest in a
   larger change.

Questions about using the project belong in [GitHub Discussions →
Q&A](https://github.com/lexing-2026/TensorPlay/discussions/categories/q-a) or
[Discord](https://discord.gg/u6T5e2kGJm); bugs and concrete work items belong in
issues. Security problems must be reported privately, never as a public issue
(see [SECURITY.md](../SECURITY.md)).

## Ways to contribute

You do not have to write kernels to help. All of these count, and a documentation
or test contribution is often the fastest way to learn the codebase.

| Kind | Examples | Start with |
| --- | --- | --- |
| Code | a failing operator, a dispatch gap, a small compiler fix | `good first issue` |
| Tests | a regression test for a fixed bug, a missing dtype or shape case | the bug or PR that needs coverage |
| Docs | setup instructions, docstrings, tutorials, Chinese translations | `documentation` |
| Benchmarks | a case for a slow operator, a clearer measurement method | `performance` |
| Triage | reproduce a report, confirm a version, ask for missing output | issues labeled `needs triage` |

## Labels you will see

| Label | Means | Typical size |
| --- | --- | --- |
| `good first issue` | curated starter task with the fields below | 1-2 files |
| `help wanted` | needs an extra pair of hands; may be larger | 1-5 files |
| `documentation` | docs, tutorials, examples and translations | 1-2 files |
| `needs triage` | not checked by a maintainer yet | wait before starting |
| `kind/flaky` | intermittent test or build failure | include rerun evidence |
| `platform: linux\|macos\|windows` | the report or change is OS-specific | as scoped |
| `release notes: *` | which subsystem a change lands in | applied by maintainers |

Only `good first issue` is curated for a first contribution. `help wanted`
means a maintainer wants help, not that the task is small. The full inventory
lives in [.github/labels.yml](../.github/labels.yml).

Browse them live:

- [good first issues](https://github.com/lexing-2026/TensorPlay/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- [help wanted](https://github.com/lexing-2026/TensorPlay/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
- [documentation](https://github.com/lexing-2026/TensorPlay/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation)

## Claim a task

1. Read the issue and run its reproduction command. If it does not reproduce,
   say so in the thread with your version, platform and output.
2. Comment with your plan, not just "can I take this?". A one-line plan plus
   the command you ran tells a maintainer that you have reproduced the problem.
3. Wait to be assigned before starting anything that touches more than a couple
   of lines. Typo and documentation fixes can go straight to a pull request.
4. Work on one claimed task at a time. If you cannot continue, comment and
   unassign yourself so someone else can pick it up. There is no penalty for
   handing work back.
5. If the thread goes quiet for about two weeks, the task may be unassigned so
   someone else can take it.

Asking a question never requires claiming the task. Maintainers would rather
answer a question early than review work built on a wrong assumption.

## Set up your environment

Prerequisites: Python 3.9+, CMake 3.18+ and a C++20 compiler (MSVC on Windows,
GCC/Clang on Linux and macOS). A CUDA toolkit is optional; the CPU build is
enough for most starter tasks.

```bash
# Fork on GitHub, then clone your fork and track the main repository.
git clone https://github.com/<your-username>/TensorPlay.git
cd TensorPlay
git remote add upstream https://github.com/lexing-2026/TensorPlay.git

# Editable CPU build (drop USE_CUDA=OFF if you have a CUDA toolkit).
USE_CUDA=OFF pip install -e .

# Conventional Commits and lint hooks.
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg

# Smoke test.
pytest test/test_tensor_methods.py
```

On Windows run these from an MSVC developer shell; in PowerShell set the build
variable with `$env:USE_CUDA = "OFF"` before installing. The first build takes
several minutes because PEP 517 fetches the build requirements; `MAX_JOBS=N`
caps build parallelism if memory is tight.

Run the whole suite with `pytest test/`. When a device run fails and the real
culprit is unclear, `python tools/run_tests_isolated.py` runs each file in its
own interpreter and reports a per-file verdict.

The full workflow, golden-file tests and sharding are documented in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Curated tasks

| Issue | Area | Level | What you will learn |
| --- | --- | --- | --- |
| [#117](https://github.com/lexing-2026/TensorPlay/issues/117) | build, CPU kernels | `good first issue` | how the ISA tiers (`DEFAULT` / `AVX2` / `AVX512`) are compiled and selected, and why a Windows wheel can silently ship without any of them |
| [#122](https://github.com/lexing-2026/TensorPlay/issues/122) | loss kernels, CPU | `good first issue` | how a row-wise loss is assembled from norms, and why a hand-written loop can lose 20x to the same math expressed with ordinary ops |
| [#120](https://github.com/lexing-2026/TensorPlay/issues/120) | vector math, CPU | `help wanted` | how the vector math layer gates its SIMD bodies behind compiler macros, and what a Windows build loses when it falls through |
| [#121](https://github.com/lexing-2026/TensorPlay/issues/121) | arithmetic / pointwise, CPU | `help wanted` | how 200 vector fast paths sit behind an arch macro, and what a Windows build loses when MSVC never matches it |

The pinned **Newcomer tasks** issue lists the same tasks and grows as
maintainers triage new reports. This page is the source of truth; the pinned
issue is the short index.

## What a good starter issue contains

- **Where** — the files or modules to touch (`p10/CMakeLists.txt`, `docs/...`).
- **How to see it** — a command that shows the problem, e.g.
  `python benchmark/bench_hotops_vs_torch.py --threads 8 --reps 7`.
- **Done when** — an observable check: a capability string, a timing number,
  or a passing test.
- **Mentor** — the maintainer who will answer in the thread.

If one of the four is missing, say so in the thread. That is a maintainer gap,
not a gap in your skills.

## Anatomy of a starter pull request

A good first pull request is small enough to review in one sitting.

- Branch from `main` and keep one logical change per pull request. Split
  unrelated refactors, formatting sweeps and renames into separate pull
  requests.
- Add a test that fails without the change, or explain in the pull request why
  the behaviour cannot be tested.
- Update the docs when user-facing behaviour changes.
- Use a Conventional Commits title; it becomes the squash-commit subject, for
  example `fix(kernels): select AVX2 tiers on MSVC builds`.
- Link the issue with `Fixes #123` in the summary.
- Run `pre-commit run` on your staged files and the test module you touched.
- Open a draft pull request early if you want direction on the approach.

The repository pull request template repeats the checklist. CI builds the wheel
matrix on every pull request, so a red lane unrelated to your change should be
reported in the thread rather than worked around.

## Review and merge

- We aim for a first response within two working days. Maintainers are
  volunteers in different time zones; a polite ping after three working days
  is welcome.
- Changes requested during review are normal; they are about the code, not you.
- Pull requests are squash-merged, so the pull request title must follow the
  commit convention.
- Merged pull requests carrying a `release notes: *` label are collected into
  that subsystem's release notes.
- An unassigned issue may be offered to someone else after about two weeks of
  silence. The stale bot marks abandoned pull requests after 60 days and closes
  them 14 days later.

## If you get stuck

- Ask in the issue thread first; the reproduction command, versions and review
  history live there.
- Use [Discord](https://discord.gg/u6T5e2kGJm) for a quick question or
  [Discussions → Q&A](https://github.com/lexing-2026/TensorPlay/discussions/categories/q-a)
  for a question that should stay searchable.
- Read [CONTRIBUTING.md](../CONTRIBUTING.md) for the full development workflow
  and [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) for how we treat each other.

## Mentoring

Every `good first issue` gets a maintainer watching the thread. Expect a first
reply within two working days; if the thread has gone quiet, ping it again.
Questions are never too basic, and "I do not understand this part of the
codebase" is useful feedback that we can turn into better docs.

## Maintainers: curating a starter task

Label an issue `good first issue` only when it has all four fields above, fits
in one or two files, and does not require a public API decision. Add a named
mentor, then add the row to the table in this page and the pinned index. Update
this page first; the pinned issue is only a short index. Anything larger is
`help wanted`, even when it is a good entry point for someone with some
experience.