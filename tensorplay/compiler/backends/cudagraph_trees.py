"""Recorded CUDA graphs as a tree of input paths over one memory pool.

A recording captures the addresses of the tensors it was recorded against, so
a call whose tensors live elsewhere cannot replay it.  The manager here gives
every recording private input buffers, copies a call's tensors onto them and
replays: a moved input costs a copy instead of a fall back to the uncaptured
artifact.  Outputs are copied out for the same reason, so a caller may keep
the result of one call while the next one runs.

Each distinct input layout is a node below the path that produced it.  A
layout that has been recorded before replays its node; a layout that has not
records a child, so alternating between shapes keeps both recordings.  Every
recording captures into one allocator pool, so memory released by an evicted
node can serve a later recording.

What is not here: the allocator-side checkpointing that would let a live node
release its memory back to the shared pool while its outputs stay readable.
Until that exists every live node holds its own static outputs, and memory is
reclaimed when a node is evicted.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .cudagraphs import (
    CudaGraphError,
    CudaGraphInputDrift,
    _input_data_ptr as _input_ptr,
    _looks_tensor,
    _output_structure,
    _pin_launch_stream_to_capture,
    _shape_signature,
    _stage_input,
)


class GraphNode:
    """One recorded path: a graph and the inputs it was recorded against."""

    __slots__ = (
        "key",
        "signature",
        "parent",
        "children",
        "graph",
        "static_inputs",
        "static_outputs",
        "tensor_positions",
        "warmup_structure",
        "replays",
        "last_used",
        "bound_positions",
    )

    def __init__(
        self,
        key: str,
        signature: tuple,
        parent: Optional["GraphNode"],
        graph: Any,
        static_inputs: list[Any],
        static_outputs: list[Any],
        tensor_positions: tuple[int, ...],
        warmup_structure: tuple,
        bound_positions: frozenset = frozenset(),
    ) -> None:
        self.key = key
        self.signature = signature
        self.parent = parent
        self.children: dict[tuple, GraphNode] = {}
        self.graph = graph
        self.static_inputs = static_inputs
        self.static_outputs = static_outputs
        self.tensor_positions = tensor_positions
        self.warmup_structure = warmup_structure
        self.replays = 0
        self.last_used = 0
        # Arguments whose layout has no private buffer: the recording reads
        # them where the caller keeps them, so a replay checks their address
        # instead of copying.
        self.bound_positions = bound_positions

    def path(self) -> tuple[tuple, ...]:
        """The signatures from the root down to this node."""

        path: list[tuple] = []
        node: Optional[GraphNode] = self
        while node is not None and node.parent is not None:
            path.append(node.signature)
            node = node.parent
        path.reverse()
        return tuple(path)


class CUDAGraphTreeManager:
    """Record a graph per input path, sharing one memory pool.

    ``capture`` records ``fn`` against private copies of the sample arguments;
    ``replay`` copies a call's tensors onto those copies and replays.  A key
    names one recording: capturing it again returns the node recorded first.
    """

    def __init__(
        self,
        native: Optional[Any] = None,
        max_entries: int = 8,
        copy_outputs: Optional[Sequence[int]] = None,
    ) -> None:
        self._native_module = native
        self.max_entries = max_entries
        # Which results a caller keeps: ``None`` copies every one (the safe
        # default for a caller that hands them all out), a sequence names the
        # positions that must survive the next replay, and an empty sequence
        # leaves the caller to take its own copies of what it keeps.
        self._copy_outputs: Optional[frozenset[int]] = (
            None if copy_outputs is None else frozenset(int(i) for i in copy_outputs)
        )
        self._nodes: dict[str, GraphNode] = {}
        self._root = GraphNode("", (), None, None, [], [], (), ())
        self._last: Optional[GraphNode] = None
        self._pool_handle: Optional[int] = None
        self._tick = 0
        self.capturing: Optional[str] = None

    # -- native plumbing ------------------------------------------------------

    @property
    def native(self) -> Any:
        if self._native_module is None:
            from .cudagraphs import _default_native

            self._native_module = _default_native()
        return self._native_module

    def _new_graph(self) -> Any:
        return self.native.CUDAGraph()

    def _pool(self) -> int:
        """The one pool every recording of this manager captures into."""

        if self._pool_handle is None:
            handle = getattr(self.native, "graph_pool_handle", None)
            self._pool_handle = int(handle()) if handle is not None else 0
        return self._pool_handle

    def _next_tick(self) -> int:
        self._tick += 1
        return self._tick

    # -- API ------------------------------------------------------------------

    def has(self, key: str) -> bool:
        return key in self._nodes

    def capture(self, key: str, fn: Any, *sample_args: Any) -> GraphNode:
        if self.capturing is not None:
            raise CudaGraphError(
                f"nested capture attempted ({self.capturing!r} already active)"
            )
        signature = _shape_signature(sample_args)
        existing = self._nodes.get(key)
        if existing is not None:
            if existing.signature != signature:
                raise CudaGraphError(
                    f"node {key!r} was recorded for {existing.signature}, "
                    f"refusing to re-record it for {signature}"
                )
            # Re-entering a recorded node is a use: it becomes the path any
            # node recorded next hangs below.
            existing.last_used = self._next_tick()
            self._last = existing
            return existing
        while len(self._nodes) >= self.max_entries and self._nodes:
            self._evict_one()

        # Recording against the call's own tensors would bake their addresses
        # in; private buffers stay with the node and are written before each
        # replay, so the graph sees a stable address and fresh values.  An
        # input whose layout cannot be reproduced (an expanded view) has no
        # private buffer: the node reads it where the caller keeps it, and a
        # call that moves it falls back to the artifact.
        static_inputs: list[Any] = []
        bound: set[int] = set()
        for index, arg in enumerate(sample_args):
            buffer = _stage_input(arg) if _looks_tensor(arg) else None
            if buffer is None:
                if _looks_tensor(arg):
                    bound.add(index)
                static_inputs.append(arg)
            else:
                static_inputs.append(buffer)
        tensor_positions = tuple(
            index for index, arg in enumerate(sample_args) if _looks_tensor(arg)
        )
        graph = self._new_graph()
        structure: tuple = ("unsupported",)
        try:
            # Warmup runs lazy initialisations outside the capture (cuBLAS
            # workspaces and the like).
            warmup_output = fn(*static_inputs)
            structure = _output_structure(warmup_output)
            self.capturing = key
            graph.capture_begin(self._pool())
            restore_stream = _pin_launch_stream_to_capture()
            try:
                try:
                    outputs = fn(*static_inputs)
                except BaseException:
                    # A raising body must still close the capture window;
                    # capture_end also clears the sticky driver error.
                    try:
                        graph.capture_end()
                    except Exception:
                        pass
                    raise
            finally:
                if restore_stream is not None:
                    restore_stream()
            graph.capture_end()
        finally:
            self.capturing = None

        static_outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        parent = self._last if self._last is not None else self._root
        node = GraphNode(
            key,
            signature,
            parent,
            graph,
            static_inputs,
            static_outputs,
            tensor_positions,
            structure,
            frozenset(bound),
        )
        node.last_used = self._next_tick()
        parent.children.setdefault(signature, node)
        self._nodes[key] = node
        self._last = node
        return node

    def replay(
        self, key: str, *args: Any, verify: bool = True, layout_checked: bool = False
    ) -> list[Any]:
        node = self._nodes.get(key)
        if node is None:
            raise CudaGraphError(f"no recorded graph under key {key!r}")
        if len(args) != len(node.static_inputs):
            raise CudaGraphInputDrift(
                f"node {key!r} expects {len(node.static_inputs)} inputs, "
                f"got {len(args)}"
            )
        if verify and not layout_checked:
            signature = _shape_signature(args)
            if signature != node.signature:
                raise CudaGraphInputDrift(
                    f"node {key!r} recorded for {node.signature}, replay args "
                    f"are {signature}"
                )
        static = [node.static_inputs[i] for i in node.tensor_positions]
        tensors = [args[i] for i in node.tensor_positions]
        bound = node.bound_positions
        if bound:
            # A bound argument is read where the recording found it, so it
            # must still be there; every other argument is copied below.
            for position in sorted(bound):
                recorded = _input_ptr(node.static_inputs[position])
                if _input_ptr(args[position]) != recorded:
                    raise CudaGraphInputDrift(
                        f"node {key!r} input at position {position} lives at "
                        f"{_input_ptr(args[position])!r}, but the recording "
                        f"reads it at {recorded!r}",
                        (position,),
                    )
        stage = getattr(node.graph, "stage_and_launch", None)
        if stage is not None and not bound:
            # One native call stages every input and launches the graph.
            stage(static, tensors)
        else:
            for position, destination, source in zip(
                node.tensor_positions, static, tensors
            ):
                if position in bound:
                    continue
                destination.copy_(source)
            node.graph.replay()
        node.replays += 1
        node.last_used = self._next_tick()
        self._last = node
        # The recording's own outputs are overwritten by the next replay, so a
        # caller that keeps a value gets a buffer of its own.  A caller that
        # knows which of the results are read (see ``copy_outputs``) can have
        # the rest handed over as they lie: they are dead once the call ends.
        if self._copy_outputs is None:
            copy = range(len(node.static_outputs))
        else:
            copy = self._copy_outputs
        return [
            value.clone() if _looks_tensor(value) and index in copy else value
            for index, value in enumerate(node.static_outputs)
        ]

    def clear(self, key: Optional[str] = None) -> None:
        if key is None:
            nodes = list(self._nodes.values())
            self._nodes.clear()
            self._last = None
            self._root.children.clear()
        else:
            node = self._nodes.pop(key, None)
            nodes = [] if node is None else [node]
            if self._last is not None and self._last.key == key:
                self._last = None
        for node in nodes:
            if node.parent is not None:
                node.parent.children.pop(node.signature, None)
            for child in node.children.values():
                child.parent = None
            node.children.clear()
            reset = getattr(node.graph, "reset", None)
            if reset is not None:
                reset()

    def tree(self) -> list[tuple[tuple, ...]]:
        """The recorded paths, each from the root to one node."""

        return [node.path() for node in self._nodes.values()]

    def _evict_one(self) -> None:
        victim = min(self._nodes.values(), key=lambda node: node.last_used)
        self.clear(victim.key)
