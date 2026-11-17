"""The whole-region artifact cache: a finished region is not lowered again.

A region's artifact is decided before lowering starts -- by the program, the
values it reads from its module, the inputs it was specialized to, the settings
in force and the toolchain.  So a region compiled before is recognised there
and its artifact is loaded.  These tests pin what the key covers, that a stored
artifact really replaces a lowering, and that a region whose key cannot be
proven is simply lowered as before.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import tensorplay as tp
from tensorplay.compiler.backends.stax import config
from tensorplay.compiler.backends.stax import graph_artifact_cache
from tensorplay.compiler.backends.stax import kernel_cache
from tensorplay.compiler.backends.stax import graph_lowering
from tensorplay.graph import Tracer


@pytest.fixture(autouse=True)
def isolated_artifacts(tmp_path, monkeypatch):
    """Keep this test's artifacts in a directory of its own."""

    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(kernel_cache, "_default_caches", {})
    # The suite leaves the artifact cache off so that a test which watches a
    # region being lowered always sees the lowering; this file is about the
    # cache itself.
    monkeypatch.setattr(config, "cache_graph_artifacts", True)
    return tmp_path


@pytest.fixture
def lowering_calls(monkeypatch):
    """Every lowering this test performs, in order."""

    calls = []
    original = graph_lowering.GraphLowering.run

    def counting(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(graph_lowering.GraphLowering, "run", counting)
    return calls


def _key(graph_module, example_inputs, **overrides):
    settings = {
        "is_inference": True,
        "is_backward": False,
        "aot_mode": False,
        "use_cuda_codegen": False,
    }
    settings.update(overrides)
    return graph_artifact_cache.region_key(
        graph_module, example_inputs, **settings
    )


def _traced(module, sample):
    return Tracer(execute=True).trace(module, sample_inputs=sample)


def _artifact_files(cache_root):
    return sorted(Path(cache_root).glob("cache/*/graph/*.pkl"))


def test_a_finished_region_is_reused_without_lowering(lowering_calls):
    """Two compile calls over the same region lower it once.

    The second call goes through the whole frontend again -- different
    callable, different specialization cache -- and still reaches the stored
    artifact: what the artifact is decided by has not changed.
    """

    def first(x):
        return x * 2 + 1

    def second(x):
        return x * 2 + 1

    x = tp.tensor([1.0, 2.0])
    assert tp.compile(first, backend="stax")(x).tolist() == [3.0, 5.0]
    assert len(lowering_calls) == 1
    assert tp.compile(second, backend="stax")(x).tolist() == [3.0, 5.0]
    assert len(lowering_calls) == 1


def test_a_stored_region_is_reused_in_a_new_process(isolated_artifacts):
    """The artifact survives the process that built it.

    The second process runs with lowering turned into a failure, so reaching
    the right answer means it read the stored artifact instead of lowering.
    """

    script = textwrap.dedent(
        """
        import os

        import tensorplay as tp
        from tensorplay.compiler.backends.stax import graph_lowering

        if os.environ.get("FAIL_LOWER"):
            def unreachable(self, *args, **kwargs):
                raise AssertionError("a stored artifact was not used")

            graph_lowering.GraphLowering.run = unreachable

        class Net(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = tp.nn.Parameter(tp.tensor([2.0, 3.0]))

            def forward(self, x):
                return x * self.w + 1

        compiled = tp.compile(Net(), backend="stax")
        print(compiled(tp.tensor([1.0, 2.0])).tolist())
        """
    )
    repo = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "TP_CACHE_DIR": str(isolated_artifacts)}
    environment.pop("FAIL_LOWER", None)

    def run(environment):
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )

    assert run(environment).stdout.strip() == "[3.0, 7.0]"
    assert _artifact_files(isolated_artifacts)
    assert run({**environment, "FAIL_LOWER": "1"}).stdout.strip() == "[3.0, 7.0]"


def test_a_rewritten_region_is_reused_in_a_new_process(isolated_artifacts):
    script = textwrap.dedent(
        """
        import os
        import tensorplay as tp
        from tensorplay.compiler.backends.stax import graph_lowering

        if os.environ.get("FAIL_LOWER"):
            def unreachable(self, *args, **kwargs):
                raise AssertionError("a stored artifact was not used")
            graph_lowering.GraphLowering.run = unreachable

        def region(a, b):
            joined = tp.cat([a, b], 1)
            return tp.cat([joined, joined[:, :2]], 1)

        a = tp.tensor([[1.0, 2.0, 3.0]])
        b = tp.tensor([[4.0, 5.0]])
        print(tp.compile(region, strict_native=True)(a, b).tolist())
        """
    )
    environment = {**os.environ, "TP_CACHE_DIR": str(isolated_artifacts)}
    environment.pop("FAIL_LOWER", None)
    repo = Path(__file__).resolve().parents[2]

    def run(env):
        return subprocess.run(
            [sys.executable, "-c", script], cwd=repo, env=env,
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    expected = "[[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]]"
    assert run(environment) == expected
    assert _artifact_files(isolated_artifacts)
    assert run({**environment, "FAIL_LOWER": "1"}) == expected


def test_a_cached_row_kernel_loads_in_a_new_process(isolated_artifacts):
    if not tp.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tensorplay.compiler.backends.stax.op_lowerings import _row_kernels_loadable

    if not _row_kernels_loadable():
        pytest.skip("row kernels are unavailable")
    script = textwrap.dedent(
        """
        import os
        import tensorplay as tp
        from tensorplay.compiler.backends.stax import graph_lowering

        if os.environ.get("FAIL_LOWER"):
            def unreachable(self, *args, **kwargs):
                raise AssertionError("a stored artifact was not used")
            graph_lowering.GraphLowering.run = unreachable

        def region(x):
            return tp.softmax(x, -1)

        x = tp.ones((2, 32), device="cuda")
        output = tp.compile(region, strict_native=True)(x)
        print(output.cpu().tolist())
        """
    )
    environment = {**os.environ, "TP_CACHE_DIR": str(isolated_artifacts)}
    environment.pop("FAIL_LOWER", None)
    repo = Path(__file__).resolve().parents[2]

    def run(env):
        return subprocess.run(
            [sys.executable, "-c", script], cwd=repo, env=env,
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    expected = run(environment)
    assert _artifact_files(isolated_artifacts)
    assert any(
        "cutedsl.kernels._extern.softmax_forward" in path.read_text()
        for path in Path(isolated_artifacts).rglob("*.py")
    )
    assert run({**environment, "FAIL_LOWER": "1"}) == expected


def test_a_value_the_region_reads_is_part_of_the_key():
    """An artifact built for one buffer is not reused for another.

    The printed program reads the buffer by name, so the content is what the
    artifact was built against; a different content is a different region.
    """

    class Net(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", tp.tensor([1.0, 2.0]))

        def forward(self, x):
            return x + self.bias

    module = Net()
    x = tp.tensor([0.0, 0.0])
    graph_module = _traced(module, {"x": x})
    before = _key(graph_module, [x])

    module.bias.copy_(tp.tensor([9.0, 9.0]))
    after = _key(graph_module, [x])

    assert before is not None and after is not None
    assert before != after


def test_the_settings_in_force_are_part_of_the_key():
    """A setting that changes the printed code has to change the key."""

    def fn(x):
        return x * 2 + 1

    x = tp.tensor([1.0, 2.0])
    graph_module = _traced(fn, {"x": x})
    plain = _key(graph_module, [x])
    with config.patch({"max_autotune": True}):
        tuned = _key(graph_module, [x])

    assert plain is not None and tuned is not None
    assert plain != tuned


def test_a_region_reading_an_undescribable_value_is_not_cached(monkeypatch):
    """A value whose stability cannot be proven disables caching, not keys it.

    Recording it as unchanged would hand a later call an artifact built for a
    value that has since changed; treating the region as uncachable costs a
    lowering instead.
    """

    class Net(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", tp.tensor([1.0, 2.0]))

        def forward(self, x):
            return x + self.bias

    module = Net()
    x = tp.tensor([0.0, 0.0])
    graph_module = _traced(module, {"x": x})
    assert _key(graph_module, [x]) is not None

    monkeypatch.setattr(
        graph_module, "_get_attr", lambda target: object(), raising=False
    )
    assert _key(graph_module, [x]) is None


def test_disabling_caches_bypasses_the_artifact(lowering_calls, monkeypatch):
    monkeypatch.setattr(config, "force_disable_caches", True)

    def first(x):
        return x * 2 + 1

    def second(x):
        return x * 2 + 1

    x = tp.tensor([1.0, 2.0])
    tp.compile(first, backend="stax")(x)
    tp.compile(second, backend="stax")(x)
    assert len(lowering_calls) == 2


def test_an_unreadable_artifact_is_a_miss(lowering_calls, isolated_artifacts):
    """A damaged entry costs a lowering, never an error."""

    def first(x):
        return x * 2 + 1

    def second(x):
        return x * 2 + 1

    x = tp.tensor([1.0, 2.0])
    tp.compile(first, backend="stax")(x)
    files = _artifact_files(isolated_artifacts)
    assert files
    files[0].write_bytes(b"not a stored artifact")

    assert tp.compile(second, backend="stax")(x).tolist() == [3.0, 5.0]
    assert len(lowering_calls) == 2
