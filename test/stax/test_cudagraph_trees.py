"""Recorded graph trees: nodes per input layout over one memory pool."""

import pytest

import tensorplay as tp
from tensorplay._stax.cudagraph_trees import (
    CUDAGraphTreeManager,
    _output_structure,
    _stage_input,
)
from tensorplay._stax.cudagraphs import CudaGraphError


# --------------------------------------------------------------------------
# stand-ins
# --------------------------------------------------------------------------


class FakeDevice:
    def is_cuda(self):
        return True

    def __repr__(self):
        return "cuda:0"


class FakeTensor:
    """Duck-typed CUDA tensor delegating storage to a real CPU tensor."""

    def __init__(self, data):
        self.data = data
        self.device = FakeDevice()

    @property
    def shape(self):
        return tuple(self.data.shape)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def requires_grad(self):
        return self.data.requires_grad

    def clone(self):
        return FakeTensor(self.data.clone())

    def copy_(self, other):
        self.data.copy_(other.data)
        return self

    def new_empty_strided(self, size, stride):
        return FakeTensor(tp.empty_strided(size, stride, dtype=self.data.dtype))

    def stride(self):
        return tuple(self.data.stride())

    def tolist(self):
        return self.data.tolist()


class FakeGraph:
    def __init__(self):
        self.in_capture = False
        self.captured = False
        self.replays = 0
        self.resets = 0
        self.pool = None
        self.staged = []
        self.recompute = None

    def capture_begin(self, pool=0, capture_error_mode="global", stream=None):
        assert not self.in_capture
        self.in_capture = True
        self.pool = pool

    def capture_end(self):
        assert self.in_capture
        self.in_capture = False
        self.captured = True

    def stage_and_launch(self, static_inputs, inputs):
        assert self.captured
        self.staged.append(list(inputs))
        for destination, source in zip(static_inputs, inputs):
            destination.copy_(source)
        self._launch()

    def replay(self):
        assert self.captured
        self._launch()

    def _launch(self):
        if self.recompute is not None:
            self.recompute()
        self.replays += 1

    def reset(self):
        self.resets += 1


class FakeNative:
    def __init__(self, pool_handle=None):
        self.graphs = []
        self._pool_handle = pool_handle

    def CUDAGraph(self):
        graph = FakeGraph()
        self.graphs.append(graph)
        return graph

    graph_pool_handle = property(
        lambda self: (
            (lambda: self._pool_handle) if self._pool_handle is not None else None
        )
    )


class NoStageGraph(FakeGraph):
    stage_and_launch = None


class NoStageNative(FakeNative):
    def CUDAGraph(self):
        graph = NoStageGraph()
        self.graphs.append(graph)
        return graph


def _doubling(graphs):
    """Stand-in artifact: out = x * 2, refreshed from the staged input."""

    def artifact(x):
        out = FakeTensor(x.data * 2)
        for graph in graphs:
            if graph.in_capture:

                def recompute(source=x, result=out):
                    result.data = source.data * 2

                graph.recompute = recompute
                break
        return out

    return artifact


# --------------------------------------------------------------------------
# input staging
# --------------------------------------------------------------------------


def test_stage_keeps_a_contiguous_layout():
    source = FakeTensor(tp.randn(4))
    staged = _stage_input(source)
    assert staged is not source
    assert staged.stride() == source.stride()


def test_stage_keeps_a_sliced_view_layout():
    source = FakeTensor(tp.randn(8)[::2])
    staged = _stage_input(source)
    # a contiguous copy would be read at the wrong positions by the kernels
    assert staged.stride() == source.stride()
    assert staged.tolist() == source.tolist()


def test_output_structure_shapes():
    assert _output_structure(FakeTensor(tp.randn(2))) == ("single",)
    assert _output_structure((FakeTensor(tp.randn(2)),)) == ("tuple", 1)
    assert _output_structure([FakeTensor(tp.randn(2))]) == ("list", 1)
    assert _output_structure(3.5) == ("unsupported",)


# --------------------------------------------------------------------------
# capture / replay
# --------------------------------------------------------------------------


def test_moved_inputs_replay_without_seeing_the_callers_storage():
    native = FakeNative()
    manager = CUDAGraphTreeManager(native=native)
    manager.capture("k", _doubling(native.graphs), FakeTensor(tp.tensor([0.0])))

    first = FakeTensor(tp.tensor([3.0]))
    assert manager.replay("k", first)[0].tolist() == [6.0]
    second = FakeTensor(tp.tensor([-4.0]))
    assert manager.replay("k", second)[0].tolist() == [-8.0]
    graph = native.graphs[0]
    # values travelled through the staging copies, once per replay
    assert [call[0].tolist() for call in graph.staged] == [[3.0], [-4.0]]
    assert graph.replays == 2


def test_returned_outputs_are_independent():
    native = FakeNative()
    manager = CUDAGraphTreeManager(native=native)
    manager.capture("k", _doubling(native.graphs), FakeTensor(tp.tensor([0.0])))
    first = manager.replay("k", FakeTensor(tp.tensor([1.0])))[0]
    again = manager.replay("k", FakeTensor(tp.tensor([5.0])))[0]
    assert first is not again
    assert first.tolist() == [2.0]
    assert again.tolist() == [10.0]


def test_staging_uses_the_bulk_native_entry():
    native = FakeNative()
    manager = CUDAGraphTreeManager(native=native)
    manager.capture("k", _doubling(native.graphs), FakeTensor(tp.tensor([0.0])))
    manager.replay("k", FakeTensor(tp.tensor([2.0])))
    graph = native.graphs[0]
    assert [call[0].tolist() for call in graph.staged] == [[2.0]]
    # the bulk entry launched the graph itself
    assert graph.replays == 1


def test_managers_without_the_bulk_entry_copy_then_replay():
    native = NoStageNative()
    manager = CUDAGraphTreeManager(native=native)
    manager.capture("k", _doubling(native.graphs), FakeTensor(tp.tensor([0.0])))
    assert manager.replay("k", FakeTensor(tp.tensor([2.0])))[0].tolist() == [4.0]
    assert native.graphs[0].replays == 1


def test_recording_a_taken_key_enters_the_existing_node():
    native = FakeNative()
    manager = CUDAGraphTreeManager(native=native)
    sample = FakeTensor(tp.tensor([0.0]))
    node = manager.capture("k", _doubling(native.graphs), sample)
    assert manager.capture("k", _doubling(native.graphs), sample) is node
    assert len(native.graphs) == 1

    with pytest.raises(CudaGraphError, match="was recorded for"):
        manager.capture("k", _doubling(native.graphs), FakeTensor(tp.tensor([0.0, 0.0])))


def test_replay_without_a_node_is_refused():
    manager = CUDAGraphTreeManager(native=FakeNative())
    with pytest.raises(CudaGraphError, match="no recorded graph"):
        manager.replay("missing")


# --------------------------------------------------------------------------
# the tree
# --------------------------------------------------------------------------


def _capture_shapes(native, manager, *shapes):
    keys = []
    for index, shape in enumerate(shapes):
        key = f"k{index}"
        manager.capture(key, _doubling(native.graphs), FakeTensor(tp.randn(*shape)))
        keys.append(key)
    return keys


def test_new_layouts_record_below_the_path_that_produced_them():
    native = FakeNative()
    manager = CUDAGraphTreeManager(native=native)
    _capture_shapes(native, manager, (2,), (3,), (2,))
    assert sorted(len(path) for path in manager.tree()) == [1, 2, 3]


def test_reentering_a_node_reparents_the_next_recording():
    native = FakeNative()
    manager = CUDAGraphTreeManager(native=native)
    first, _, _ = _capture_shapes(native, manager, (2,), (3,), (2,))
    # re-entering the root node makes it the path of the next recording
    manager.replay(first, FakeTensor(tp.randn(2)))
    manager.capture("k3", _doubling(native.graphs), FakeTensor(tp.randn(4)))
    path = (manager._nodes[first].signature, manager._nodes["k3"].signature)
    assert path in manager.tree()


def test_alternating_layouts_reuse_both_nodes():
    native = FakeNative()
    manager = CUDAGraphTreeManager(native=native)
    manager.capture("a", _doubling(native.graphs), FakeTensor(tp.randn(2)))
    manager.capture("b", _doubling(native.graphs), FakeTensor(tp.randn(3)))
    for _ in range(3):
        assert manager.replay("a", FakeTensor(tp.randn(2)))[0].shape == (2,)
        assert manager.replay("b", FakeTensor(tp.randn(3)))[0].shape == (3,)
    assert len(native.graphs) == 2


def test_every_node_captures_into_one_pool():
    native = FakeNative(pool_handle=41)
    manager = CUDAGraphTreeManager(native=native)
    _capture_shapes(native, manager, (2,), (3,), (4,))
    assert [graph.pool for graph in native.graphs] == [41, 41, 41]


def test_eviction_drops_the_least_recently_used_node():
    native = FakeNative()
    manager = CUDAGraphTreeManager(native=native, max_entries=2)
    manager.capture("a", _doubling(native.graphs), FakeTensor(tp.randn(2)))
    manager.capture("b", _doubling(native.graphs), FakeTensor(tp.randn(3)))
    manager.replay("a", FakeTensor(tp.randn(2)))  # a is now the newest
    manager.capture("c", _doubling(native.graphs), FakeTensor(tp.randn(4)))
    assert not manager.has("b")
    assert manager.has("a") and manager.has("c")
    assert native.graphs[1].resets == 1


def test_clear_releases_every_recording():
    native = FakeNative()
    manager = CUDAGraphTreeManager(native=native)
    _capture_shapes(native, manager, (2,), (3,))
    manager.clear()
    assert manager.tree() == []
    assert [graph.resets for graph in native.graphs] == [1, 1]


# --------------------------------------------------------------------------
# real CUDA capture (gated)
# --------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA runtime is not available"
)


@requires_cuda
def test_real_capture_replays_fresh_inputs_of_new_layouts():
    native = _distinct_graphs()
    manager = CUDAGraphTreeManager(native=native)
    sample = tp.randn(8, device="cuda")
    node = manager.capture("a", lambda x: x * 2, sample)
    assert node.graph.has_graph_exec

    fresh = tp.randn(8, device="cuda")
    got = manager.replay("a", fresh)[0]
    assert tp.allclose(got, fresh * 2)

    # a different layout records a second node under the same pool
    wide = tp.randn(4, 16, device="cuda")
    manager.capture("b", lambda x: x * 2, wide)
    assert tp.allclose(manager.replay("b", wide)[0], wide * 2)
    assert [g.pool_id() for g in native.graphs] == [native.graphs[0].pool_id()] * 2


@requires_cuda
def test_real_replay_does_not_touch_the_callers_tensor():
    manager = CUDAGraphTreeManager(native=_distinct_graphs())
    manager.capture("k", lambda x: x * 2, tp.randn(8, device="cuda"))
    held = tp.randn(8, device="cuda")
    snapshot = held.clone()
    first = manager.replay("k", held)[0]
    held.copy_(tp.randn(8, device="cuda"))
    second = manager.replay("k", held)[0]
    assert first is not second
    assert tp.allclose(first, snapshot * 2)


def _distinct_graphs():
    """A native surface whose CUDAGraph entries are independent."""

    from tensorplay import _C

    class Native:
        def __init__(self):
            self.graphs = []

        def CUDAGraph(self):
            graph = _C.CUDAGraph()
            self.graphs.append(graph)
            return graph

        @staticmethod
        def graph_pool_handle():
            return _C.graph_pool_handle()

    return Native()