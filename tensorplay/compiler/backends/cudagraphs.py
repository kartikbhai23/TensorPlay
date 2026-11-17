"""CUDA graphs orchestration.

(capture once, replay against the captured inputs), driven entirely by the
native :class:`tensorplay._C.CUDAGraph` class:

* ``capture_begin/capture_end`` own the dedicated per-device side stream,
  route allocations into a graph-private allocator pool and register
  graph-safe RNG state; instantiation happens eagerly at ``capture_end``.
* Replay binds the graph directly to the caller's input storage: capture
  records the input addresses, and every replay first checks that the
  incoming tensors still live at those addresses.  Matching inputs are read
  live by the captured kernels - no staging copy at all, which keeps large
  memory-bound workloads from paying a full extra read+write of every input
  per replay.  Inputs that drifted to a different address raise
  :class:`CudaGraphInputDrift`; callers run the underlying artifact
  directly for that call.

Tests may inject a stand-in via ``CudaGraphManager(native=...)``; the
stand-in must expose a ``CUDAGraph`` class with ``capture_begin``,
``capture_end``, ``replay`` and ``reset``.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .._core.registry import check_backend_kwargs


class CudaGraphError(RuntimeError):
    """Raised for capture/replay contract violations."""


class CudaGraphInputDrift(CudaGraphError):
    """Replay arguments no longer fit the captured graph.

    Either the tensors moved to different addresses (the graph reads the
    captured storage directly, so a moved input cannot be seen) or their
    shape/layout signature changed.  Transient by nature: the next call
    with matching arguments replays again.
    """


def _default_native() -> Any:
    try:
        from ... import _C  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure diagnostics
        raise NotImplementedError(
            "CUDA graph support requires tensorplay._C; import failed: "
            f"{exc!r}. Was TensorPlay built with CUDA support?"
        ) from exc
    if not hasattr(_C, "CUDAGraph"):
        raise NotImplementedError(
            "CUDA graphs are not supported by this TensorPlay build "
            "(tensorplay._C exposes no CUDAGraph class). Was it built "
            "with CUDA support?"
        )
    return _C


def _looks_tensor(value: Any) -> bool:
    """Duck-typed tensor probe: a device with an ``is_cuda`` predicate.

    Keeps the manager usable with stand-in tensors in tests and avoids
    importing the runtime type here.
    """
    device = getattr(value, "device", None)
    return device is not None and callable(getattr(device, "is_cuda", None))


def _output_structure(value: Any) -> Tuple:
    """The shape of a recorded result, one level deep.

    A region returns a tensor, a flat sequence of tensors, or a flat sequence
    in which a part the program never computed stands as ``None`` -- what a
    module returns for an auxiliary branch it leaves out of an evaluation
    call.  Such an entry carries no value to replay, so it keeps its place in
    the result and is handed back unchanged.
    """

    if value is None or _looks_tensor(value):
        return ("single",)
    if isinstance(value, (list, tuple)):
        if all(_looks_tensor(item) or item is None for item in value):
            return ("tuple" if isinstance(value, tuple) else "list", len(value))
    return ("unsupported",)


def _shape_signature(args: Sequence[Any]) -> Tuple:
    signature: List[Any] = []
    for a in args:
        if _looks_tensor(a):
            # Strides join the signature because replay reads the input
            # storage with the strides the kernels were captured against:
            # a same-shape view with a different layout (e.g. a transpose)
            # must not reuse the entry.
            signature.append(
                (tuple(a.shape), tuple(a.stride()), str(a.dtype), str(a.device))
            )
        else:
            # Non-tensor inputs are baked into the capture; their value
            # joins the signature so a drift is refused at replay instead
            # of silently replaying a stale constant.
            signature.append(("scalar", type(a).__name__, a))
    return tuple(signature)


def _input_data_ptr(value: Any) -> Optional[int]:
    """Address of a tensor's storage, or None when unverifiable."""
    ptr = getattr(value, "data_ptr", None)
    if ptr is None:
        return None
    try:
        return int(ptr())
    except Exception:  # noqa: BLE001 - exotic stand-ins: treat as unverifiable
        return None


def _pin_triton_driver_stream(handle: int) -> list:
    """Shadow the Triton driver's stream query for the capture window.

    Every Triton launch path (generated launchers, row-fusion, plain
    ``JITFunction`` dispatch) resolves its raw launch stream through the
    active driver's ``get_current_stream``; pinning it to the capture side
    stream is what makes the kernels enter the recorded graph.
    """
    try:
        from triton.runtime import driver as triton_driver

        active = triton_driver.active
        original = active.get_current_stream
    except Exception:  # noqa: BLE001 - no Triton runtime: nothing to pin
        return []

    # The driver may define the query on the class or install it per
    # instance; record which, so the restore puts back exactly what was
    # there instead of deleting a lazily-installed method for good.
    was_instance_attr = "get_current_stream" in vars(active)

    def pinned(device: Any = None) -> int:
        return handle

    active.get_current_stream = pinned

    def restore():
        if was_instance_attr:
            active.get_current_stream = original
        else:
            try:
                del active.get_current_stream
            except AttributeError:
                pass

    return [restore]


def _pin_launch_stream_to_capture():
    """Route generated-kernel launches onto the capture stream.

    During a capture window the driver-level stream query still reports the
    legacy default stream, so kernels launched through it stay outside the
    window and the recorded graph ends up empty.  Returns a restorer, or
    None when there is no CUDA runtime to pin (stand-in natives, CPU-only
    builds).
    """
    try:
        import tensorplay

        handle = tensorplay.cuda.current_stream().cuda_stream
    except Exception:  # noqa: BLE001 - no CUDA runtime: nothing to pin
        return None
    restore = _pin_triton_driver_stream(handle)
    fastlaunch = None
    previous = None
    try:
        from .stax.runtime import fastlaunch

        previous = fastlaunch.set_capture_stream(handle)
    except Exception:  # noqa: BLE001 - launcher layer optional
        pass

    def undo():
        if fastlaunch is not None:
            fastlaunch.set_capture_stream(previous)
        for step in restore:
            step()

    return undo


class _GraphEntry:
    __slots__ = ("key", "signature", "graph", "static_inputs",
                 "static_outputs", "tensor_positions", "warmup_structure",
                 "replays", "input_ptrs")

    def __init__(self, key: str, signature: Tuple, graph: Any,
                 static_inputs: List[Any], static_outputs: List[Any],
                 tensor_positions: Tuple[int, ...],
                 warmup_structure: Tuple) -> None:
        self.key = key
        self.signature = signature
        self.graph = graph
        # The captured input tensors themselves.  The executable reads this
        # storage live on every replay, so the manager pins them for the
        # entry's lifetime; a caller mutating them between replays sees the
        # mutation on the next replay, exactly as an eager call would.
        self.static_inputs = static_inputs
        self.static_outputs = static_outputs
        # Positions of tensor inputs; their addresses are recorded at
        # capture and re-verified on every replay.
        self.tensor_positions = tensor_positions
        # Shape of the warmup run's return value: ("single",), ("tuple", n),
        # ("list", n) over one level of tensors, or ("unsupported",).  The
        # buffers behind static_outputs are the replay outputs themselves.
        self.warmup_structure = warmup_structure
        self.replays = 0
        self.input_ptrs: Tuple[Optional[int], ...] = ()


class CudaGraphManager:
    """Capture functions once, replay them against the captured inputs."""

    def __init__(self, native: Optional[Any] = None, max_entries: int = 8) -> None:
        self._native_module = native
        self._owns_lookup = native is None
        self._entries: Dict[str, _GraphEntry] = {}
        self.max_entries = max_entries
        self.capturing: Optional[str] = None

    # -- native plumbing ------------------------------------------------------

    @property
    def native(self) -> Any:
        if self._native_module is None:
            self._native_module = _default_native()
        return self._native_module

    def _new_graph(self) -> Any:
        return self.native.CUDAGraph()

    # -- API ------------------------------------------------------------------

    def capture(self, key: str, fn: Callable[..., Any], *sample_args: Any) -> _GraphEntry:
        if self.capturing is not None:
            raise CudaGraphError(
                f"nested capture attempted ({self.capturing!r} already active)"
            )
        existing = self._entries.get(key)
        signature = _shape_signature(sample_args)
        if existing is not None:
            if existing.signature != signature:
                raise CudaGraphError(
                    f"entry {key!r} was captured for {existing.signature}, "
                    f"refusing re-capture for {signature}; use a new key"
                )
            return existing
        if len(self._entries) >= self.max_entries:
            raise CudaGraphError(
                f"graph cache full ({self.max_entries}); clear stale entries"
            )

        graph = self._new_graph()
        try:
            # Warmup executes lazy initialisations outside capture (cuBLAS
            # workspaces etc.); the native capture stream matches the stream
            # capture will run on.
            warmup_output = fn(*sample_args)
            warmup_structure: Tuple = _output_structure(warmup_output)
            # The graph is captured against the caller's own tensors: the
            # recorded kernels read their storage live, so replay needs no
            # staging copy.  The manager keeps references for the entry's
            # lifetime, which also stops the allocator from recycling those
            # addresses for unrelated allocations while the entry exists.
            static_inputs = list(sample_args)
            tensor_positions = tuple(
                index for index, arg in enumerate(sample_args)
                if _looks_tensor(arg)
            )
            self.capturing = key
            graph.capture_begin()
            restore_stream = _pin_launch_stream_to_capture()
            try:
                try:
                    outputs = fn(*static_inputs)
                except BaseException:
                    # A function that raises inside the capture window (e.g. a
                    # cuBLAS workspace allocation failure) must still close the
                    # window: otherwise the stream stays in an active capture
                    # state and every later CUDA operation on the device fails
                    # with "operation failed due to a previous error during
                    # capture".  capture_end tears down pool/rng state and
                    # clears the sticky driver error; its own error is
                    # secondary to the original one.
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
        out_list = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        entry = _GraphEntry(key, signature, graph, static_inputs, out_list,
                            tensor_positions, warmup_structure)
        entry.input_ptrs = tuple(
            _input_data_ptr(sample_args[position])
            for position in tensor_positions
        )
        self._entries[key] = entry
        return entry

    def replay(self, key: str, *args: Any, verify: bool = True) -> List[Any]:
        entry = self._entries.get(key)
        if entry is None:
            raise CudaGraphError(f"no captured graph under key {key!r}")
        if len(args) != len(entry.static_inputs):
            raise CudaGraphInputDrift(
                f"entry {key!r} expects {len(entry.static_inputs)} inputs, got {len(args)}"
            )
        if verify:
            signature = _shape_signature(args)
            if signature != entry.signature:
                raise CudaGraphInputDrift(
                    f"entry {key!r} captured for {entry.signature}, replay args are {signature}"
                )
            # The executable reads the captured input storage directly, so
            # every replay verifies the tensors still live there.  A drifted
            # tensor cannot be seen by the graph - copying it onto the
            # captured storage is not an option, that storage is the caller's
            # own tensor.  A call whose arguments are the very objects the
            # last call passed, unchanged, is already known to live there.
            for position, recorded in zip(entry.tensor_positions, entry.input_ptrs):
                actual = _input_data_ptr(args[position])
                if actual != recorded:
                    raise CudaGraphInputDrift(
                        f"entry {key!r} input at position {position} lives at "
                        f"{actual!r}, but the graph was captured against "
                        f"{recorded!r}; run the artifact directly for drifted "
                        "inputs"
                    )
        entry.graph.replay()
        entry.replays += 1
        return list(entry.static_outputs)

    def clear(self, key: Optional[str] = None) -> None:
        if key is None:
            entries = list(self._entries.values())
            self._entries.clear()
        else:
            entry = self._entries.pop(key, None)
            entries = [] if entry is None else [entry]
        for entry in entries:
            reset = getattr(entry.graph, "reset", None)
            if reset is not None:
                reset()


# ---------------------------------------------------------------------------
# ``cudagraphs`` compiler backend
# ---------------------------------------------------------------------------
#
# ``tensorplay.compile(fn, backend="cudagraphs")`` runs the captured graph on
# its Python executor once per input layout, records that run into a CUDA
# graph and afterwards only replays it: one launch reading the captured input
# storage, plus copies of the outputs.  Regions that cannot
# be replayed safely are left uncaptured and run on the executor; the reason
# is logged.

log = logging.getLogger(__name__)

# Host-synchronizing or data-dependent-shape operations: a capture would
# either fail or freeze one run's result/shape into every replay.
_UNSAFE_OPS = frozenset(
    {
        "item",
        "tolist",
        "numpy",
        "nonzero",
        "argwhere",
        "unique",
        "unique_consecutive",
        "masked_select",
        "_local_scalar_dense",
        "eigh",
        "_assert_scalar",
        "_assert_async",
    }
)

_OUT_OF_PLACE_EXCEPTIONS = frozenset({"requires_grad_", "retain_grad_"})

#: Consecutive input drifts after which a compiled artifact stops replaying
#: against the caller's storage and starts staging its inputs instead.
_DRIFT_LIMIT = 2


def format_default_skip_message(reason: str) -> str:
    return f"skipping cudagraphs due to {reason}"


def _is_tensor(value: Any) -> bool:
    import tensorplay

    return isinstance(value, tensorplay.Tensor)


def _grad_enabled() -> bool:
    """Whether the ambient switch records history for this call."""

    import tensorplay

    return bool(tensorplay.is_grad_enabled())


def _storage_key(value: Any) -> Any:
    try:
        return value.untyped_storage().data_ptr()
    except Exception:  # noqa: BLE001 - storage-less tensors never alias
        return None


def _target_name(node: Any) -> str:
    if node.op == "call_method":
        return str(node.target)
    return str(getattr(node.target, "__name__", node.target))


def _schema_written_arguments(node: Any) -> list[Any] | None:
    """Values an operator-overload node writes, from its schema; None when
    the target carries no schema."""

    schema = getattr(node.target, "_schema", None)
    if schema is None or isinstance(schema, str):
        return None
    written = []
    for index, argument in enumerate(schema.arguments):
        if argument.alias_info is None or not argument.alias_info.is_write:
            continue
        if argument.name in node.kwargs:
            written.append(node.kwargs[argument.name])
        elif index < len(node.args) and not argument.kwarg_only:
            written.append(node.args[index])
    return written


def _mutated_argument(node: Any) -> Any:
    """The graph value an in-place node writes to, if any."""

    if node.op not in ("call_function", "call_method"):
        return None
    written = _schema_written_arguments(node)
    if written is not None:
        return tuple(written) if written else None
    out = node.kwargs.get("out")
    if out is not None:
        return out
    name = _target_name(node)
    if (
        name.endswith("_")
        and not name.startswith("__")
        and name not in _OUT_OF_PLACE_EXCEPTIONS
        and node.args
    ):
        return node.args[0]
    return None


def find_input_mutations(gm: Any) -> set[int]:
    """Placeholder indices whose storage an in-place node writes."""

    from ...graph import Node

    storages: dict[Any, set[int]] = {}
    mutated: set[int] = set()
    index = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            value = node.meta.get("val")
            if _is_tensor(value):
                key = _storage_key(value)
                if key is not None:
                    storages.setdefault(key, set()).add(index)
            index += 1
            continue
        target = _mutated_argument(node)
        written = target if isinstance(target, (list, tuple)) else (target,)
        for item in written:
            if isinstance(item, Node) and _is_tensor(item.meta.get("val")):
                mutated |= storages.get(_storage_key(item.meta["val"]), set())
    return mutated


def declared_input_mutations(gm: Any) -> set[int]:
    """Placeholder indices whose value the wrapper writes after the run.

    A region whose input is written is compiled as a mutation-free graph and
    the update is copied back outside it, so the graph no longer shows the
    write.  The graph module records which inputs those are, which keeps the
    region from being replayed where a write to an input would break it.
    """

    return set(gm.meta.get("input_update_placeholders", ()))


def get_device_node_mapping(gm: Any) -> dict[str, Any]:
    from ..._higher_order_ops._hop_base import value_device

    mapping: dict[str, Any] = {}
    for node in gm.graph.nodes:
        value = node.meta.get("val")
        if _is_tensor(value):
            mapping.setdefault(str(value_device(value)), node)
    return mapping


def check_multiple_devices_or_any_cpu_nodes(mapping: dict[str, Any]) -> str | None:
    mapping = {key: node for key, node in mapping.items() if key != "meta"}
    cpu_node = mapping.get("cpu")
    if cpu_node is not None:
        return format_default_skip_message(f"cpu device ({cpu_node.name})")
    if len(mapping) == 1 and next(iter(mapping)).startswith("cuda"):
        return None
    if not mapping:
        return format_default_skip_message("no CUDA tensors")
    return format_default_skip_message(f"multiple devices: {', '.join(mapping)}")


# Operators whose output size is computed from tensor data only for mask
# indices; with integer indices the size follows from the index shapes.
_MASK_SIZED_OPS = frozenset({"index", "index_put", "index_put_", "_index_put_impl_"})


def _has_mask_index(node: Any) -> bool:
    from ...graph import Node

    for item in _iter_index_values(node.args[1:] if len(node.args) > 1 else ()):
        if isinstance(item, Node):
            value = item.meta.get("val")
            if _is_tensor(value) and str(value.dtype).rsplit(".", 1)[-1] in ("bool", "uint8"):
                return True
    return False


def _data_dependent_node(node: Any) -> bool:
    """An operator node whose output size or value needs the device data."""

    tags = getattr(node.target, "tags", None)
    if tags is None:
        return False
    if "cudagraph_unsafe" in tags or "data_dependent_output" in tags:
        return True
    if "dynamic_output_shape" in tags:
        name = getattr(node.target, "_opname", "")
        return _has_mask_index(node) if name in _MASK_SIZED_OPS else True
    return False


def get_first_incompatible_cudagraph_node(gm: Any) -> Any:
    from ...graph import Node

    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        if _data_dependent_node(node):
            return node
        if _target_name(node).rsplit(".", 1)[-1] in _UNSAFE_OPS:
            return node
        # Boolean-mask indexing computes the result size from the data.
        if _target_name(node) in ("getitem", "__getitem__", "index_put", "index_put_"):
            for item in _iter_index_values(node.args[1:] if len(node.args) > 1 else ()):
                if isinstance(item, Node):
                    value = item.meta.get("val")
                    if _is_tensor(value) and str(value.dtype).rsplit(".", 1)[-1] in (
                        "bool",
                        "uint8",
                    ):
                        return node
    return None


def _iter_index_values(value: Any):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_index_values(item)
    else:
        yield value


def check_for_skip(gm: Any) -> str | None:
    mutated = find_input_mutations(gm) | declared_input_mutations(gm)
    if mutated:
        return format_default_skip_message(f"mutated inputs ({len(mutated)} instances)")
    if skip := check_multiple_devices_or_any_cpu_nodes(get_device_node_mapping(gm)):
        return skip
    if (node := get_first_incompatible_cudagraph_node(gm)) is not None:
        return format_default_skip_message(f"incompatible op ({node.name})")
    return None


def _graph_requires_grad(gm: Any) -> bool:
    for node in gm.graph.nodes:
        value = node.meta.get("val")
        if _is_tensor(value) and value.requires_grad:
            return True
    return False


class _OutputSlot:
    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index


def _flatten_outputs(value: Any, flat: List[Any]) -> Any:
    if _is_tensor(value):
        flat.append(value)
        return _OutputSlot(len(flat) - 1)
    if isinstance(value, tuple):
        return tuple(_flatten_outputs(item, flat) for item in value)
    if isinstance(value, list):
        return [_flatten_outputs(item, flat) for item in value]
    if isinstance(value, dict):
        return {key: _flatten_outputs(item, flat) for key, item in value.items()}
    return value


def _fill_outputs(template: Any, flat: Sequence[Any]) -> Any:
    if isinstance(template, _OutputSlot):
        return flat[template.index]
    if isinstance(template, tuple):
        return tuple(_fill_outputs(item, flat) for item in template)
    if isinstance(template, list):
        return [_fill_outputs(item, flat) for item in template]
    if isinstance(template, dict):
        return {key: _fill_outputs(item, flat) for key, item in template.items()}
    return template


class _CudagraphRunner:
    """Per-specialization replay state for one captured region."""

    def __init__(self, gm: Any, manager: CudaGraphManager) -> None:
        self.gm = gm
        self.executor = gm.forward
        self.manager = manager
        self.placeholders = gm.graph.placeholders
        self.requires_grad = _graph_requires_grad(gm)
        self.templates: Dict[str, Any] = {}
        self._limit_logged = False

    def _bind(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> List[Any]:
        if not kwargs and len(args) == len(self.placeholders):
            return list(args)
        bound = self.gm.signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        values = []
        for node in self.placeholders:
            key = node.target if isinstance(node.target, str) else node.name
            values.append(bound.arguments[key] if key in bound.arguments else bound.arguments[node.name])
        return values

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import tensorplay

        # Replayed outputs carry no autograd history; a region that would
        # record one runs on the executor instead.
        if self.requires_grad and tensorplay.is_grad_enabled():
            return self.executor(*args, **kwargs)
        values = self._bind(args, kwargs)
        positions = [i for i, value in enumerate(values) if _is_tensor(value)]
        tensors = [values[i] for i in positions]
        key = repr(_shape_signature(tensors))
        template = self.templates.get(key)
        if template is None:
            if len(self.templates) >= self.manager.max_entries:
                if not self._limit_logged:
                    self._limit_logged = True
                    log.warning(
                        format_default_skip_message(
                            f"more than {self.manager.max_entries} input layouts"
                        )
                    )
                return self.executor(*args, **kwargs)
            captured_template: List[Any] = []

            def region(*static: Any) -> List[Any]:
                full = list(values)
                for position, tensor in zip(positions, static):
                    full[position] = tensor
                flat: List[Any] = []
                captured_template[:] = [_flatten_outputs(self.executor(*full), flat)]
                return flat

            self.manager.capture(key, region, *tensors)
            template = self.templates[key] = captured_template[0]
        try:
            outputs = self.manager.replay(key, *tensors)
        except CudaGraphInputDrift:
            # This call's tensors live elsewhere; run the executor uncaptured
            # and keep the entry for the calls that match the captured
            # addresses.
            return self.executor(*args, **kwargs)
        return _fill_outputs(template, [output.clone() for output in outputs])


class CudagraphsBackend:
    """``backend="cudagraphs"``: capture the region once, then replay it."""

    compiler_name = "cudagraphs"
    # Both managers (direct-binding and trees) record graphs that ``reset``
    # must tear down, each through its own ``clear``.
    _managers: "weakref.WeakSet[Any]" = weakref.WeakSet()

    @staticmethod
    def reset() -> None:
        for manager in list(CudagraphsBackend._managers):
            manager.clear()

    @staticmethod
    def __call__(gm: Any, example_inputs: Sequence[Any], **kwargs: Any) -> Any:
        strict_native = bool(kwargs.pop("strict_native", False))
        dynamic = bool(kwargs.pop("dynamic", False))
        # ``name`` is only a label; anything else the caller asked for has to
        # be answered or refused, since this backend has no modes or options
        # of its own.
        check_backend_kwargs("cudagraphs", kwargs)
        return cudagraphs(
            gm, example_inputs, strict_native=strict_native, dynamic=dynamic
        )


def _cudagraphify(gm: Any, strict_native: bool, dynamic: bool = False) -> Any:
    """A replaying runner for ``gm``, or None when it cannot be captured."""

    skip = (
        format_default_skip_message("dynamic shape specialization")
        if dynamic
        else check_for_skip(gm)
    )
    if skip is not None:
        if strict_native:
            raise CudaGraphError(f"strict_native=True: {skip}")
        log.warning(skip)
        return None
    manager = CudaGraphManager()
    CudagraphsBackend._managers.add(manager)
    runner = _CudagraphRunner(gm, manager)
    runner._tensorplay_codegen = "cudagraphs"  # type: ignore[attr-defined]
    return runner


def cudagraphs(
    gm: Any,
    example_inputs: Sequence[Any],
    *,
    strict_native: bool = False,
    dynamic: bool = False,
) -> Any:
    """Capture the forward and the backward graph of ``gm`` separately.

    Ahead-of-time autograd splits the region into a forward and a backward
    graph; each is captured into its own CUDA graph and replayed, so
    training replays both passes.  A forward that cannot be captured also
    leaves its backward uncaptured.
    """

    import functools

    from .._core.common import aot_autograd

    do_cudagraphs = [True]

    def forward_cudagraphs(aot_model: Any, aot_inputs: list[Any], is_inference: bool = False) -> Any:
        runner = _cudagraphify(aot_model, strict_native, dynamic)
        if runner is None:
            do_cudagraphs[0] = False
            return aot_model.forward
        return runner

    def backward_cudagraphs(aot_model: Any, aot_inputs: list[Any]) -> Any:
        if not do_cudagraphs[0]:
            return aot_model.forward
        runner = _cudagraphify(aot_model, strict_native, dynamic)
        return aot_model.forward if runner is None else runner

    aot_cudagraphs = aot_autograd(
        fw_compiler=forward_cudagraphs,
        bw_compiler=backward_cudagraphs,
        inference_compiler=functools.partial(forward_cudagraphs, is_inference=True),
    )
    return aot_cudagraphs(gm, list(example_inputs))


# ---------------------------------------------------------------------------
# mode="reduce-overhead" replay shell around a stax-compiled artifact
# ---------------------------------------------------------------------------
#
# Where the ``cudagraphs`` backend above captures the Python executor, the
# ``stax`` backend with ``mode="reduce-overhead"`` (or ``"max-autotune"``)
# wraps its own compiled artifact: the region's kernels are warmed up,
# recorded into a CUDA graph once, and afterwards only replayed.  The wrapper
# records against the caller's own tensors by default, because an artifact's
# arguments are its parameters plus the input buffers: parameters never move,
# so a steady-state call needs no copy at all.  A caller whose inputs move
# every call switches the wrapper to staged replay after repeated drift; with
# ``triton.cudagraph_trees`` on, recordings live in a tree of input paths over
# one memory pool from the first call, copying every call's tensors onto the
# recording's own inputs.  Regions that
# cannot be replayed safely are left unwrapped with a reason attached to the
# graph module, mirroring the skip checks of the backend runner.


def _visible_output_indices(gm: Any) -> tuple[int, ...] | None:
    """Positions of a region's results that its caller reads.

    A region hands back every value its consumer needs, which includes the
    partition's saved values; only the marked prefix belongs to the caller.
    ``None`` means the graph does not say, and every result must be treated
    as kept.
    """

    graph = getattr(gm, "graph", None)
    find_nodes = getattr(graph, "find_nodes", None)
    if find_nodes is None:
        return None
    output_nodes = find_nodes(op="output")
    if not output_nodes:
        return None
    marked = output_nodes[0].meta.get("user_visible_output_idxs")
    if marked is None:
        return None
    return tuple(int(index) for index in marked)


def wrap_skip_reason(
    compiled: Any,
    gm: Any,
    example_inputs: Sequence[Any],
    *,
    dynamic: bool | None = None,
) -> str | None:
    """Why a compiled artifact may not run behind a captured CUDA graph."""

    if dynamic:
        return format_default_skip_message("dynamic shape specialization")
    saw_cuda_tensor = False
    for value in example_inputs:
        if _is_tensor(value):
            if not value.device.is_cuda():
                return format_default_skip_message("input on a non-CUDA device")
            # A tensor that demands grad does not rule the capture out: a
            # replay writes no autograd history, so the wrapper only replays
            # while autograd is off and hands a recording call to the artifact
            # itself (see CudagraphCompiledCallable).  Refusing here instead
            # would keep parameters--which demand grad by construction--from
            # ever running behind a graph, even in inference.
            saw_cuda_tensor = True
        elif not isinstance(value, (int, float, bool, type(None))):
            return format_default_skip_message(
                f"unsupported input type ({type(value).__name__})"
            )
    if not saw_cuda_tensor:
        return format_default_skip_message("no CUDA tensor input")
    if getattr(compiled, "_mutations", None):
        return format_default_skip_message("artifact replays input mutations")
    mutated = find_input_mutations(gm) | declared_input_mutations(gm)
    if mutated:
        names = ", ".join(gm.graph.placeholders[i].name for i in sorted(mutated))
        return format_default_skip_message(f"mutated inputs ({names})")
    if (node := get_first_incompatible_cudagraph_node(gm)) is not None:
        return format_default_skip_message(f"incompatible op ({node.name})")
    devices = get_device_node_mapping(gm)
    if devices:
        device_skip = check_multiple_devices_or_any_cpu_nodes(devices)
        if device_skip is not None:
            return device_skip
    return None


def _cudagraph_trees_enabled() -> bool:
    """Whether recordings are kept as a tree of input paths.

    The switch lives with the region compiler's settings; a build without
    that package still gets the direct behaviour, which needs nothing from it.
    """

    try:
        from .stax import config as stax_config
    except Exception:  # noqa: BLE001 - the switch belongs to that package
        return False
    return bool(getattr(stax_config.triton, "cudagraph_trees", False))


class CudagraphCompiledCallable:
    """Capture a compiled artifact once and replay it against its inputs.

    With trees (the default) every input layout is a node below the path that
    produced it, all recordings share one allocator pool, and a call's tensors
    are copied onto the recording's private inputs: a moved tensor replays
    normally, and the results returned to a caller stay valid across later
    calls.  With trees off the first call records against the caller's own
    tensors, steady-state calls replay with no input staging at all, and calls
    whose tensors drifted run the artifact directly for that round; outputs
    alias the graph-private pool and each replay overwrites the tensors
    returned by the previous one.
    """

    def __init__(
        self,
        compiled: Any,
        native: Any = None,
        *,
        trees: Optional[bool] = None,
        visible_outputs: Optional[Sequence[int]] = None,
    ) -> None:
        self._compiled = compiled
        self._native = native
        if trees is None:
            # The staging switch is off by default: a compiled artifact's
            # arguments are its parameters plus the input buffers, parameters
            # never move, and a steady-state call then needs no copy at all.
            # Repeated drift switches the wrapper to staged replay (see
            # _switch_to_staging), which is what the tree mode does from the
            # start.
            trees = _cudagraph_trees_enabled()
        self._trees = bool(trees)
        # Which results the program's caller reads.  Every other result a
        # region hands back is an intermediate for a consumer that is not the
        # caller (the partition's saved values); copying them out on every
        # replay costs an allocation per value, so only the read ones are
        # copied and the rest are handed over where the graph wrote them.
        self._visible: Optional[frozenset[int]] = (
            None if visible_outputs is None
            else frozenset(int(index) for index in visible_outputs)
        )
        if self._trees:
            from .cudagraph_trees import CUDAGraphTreeManager

            # The wrapper owns copying, so the manager hands every result back
            # as recorded and the wrapper copies just the read ones.
            self._manager: Any = CUDAGraphTreeManager(
                native=native, copy_outputs=()
            )
        else:
            self._manager = CudaGraphManager(native=native)
        CudagraphsBackend._managers.add(self._manager)
        self._key = f"artifact-{id(self):x}"
        self._state = "cold"
        self._lock = threading.Lock()
        self._single_output = True
        self._arity = 0
        # A replay writes no autograd history.  When the recording was taken
        # with inputs that demand grad -- parameters do by construction -- a
        # call that is recording must run the artifact instead, or the caller
        # would silently receive tensors with no history.
        self._needs_eager_when_grad = False
        # Resolution of the last call's arguments: their identities and
        # versions, and the layout key resolved from them.  See _layout_matches.
        self._layout_memo: Optional[Tuple[Tuple, Tuple]] = None
        self._layout_key_cache: Optional[str] = None
        self._drifts = 0
        self.__name__ = getattr(compiled, "__name__", "cudagraph_compiled")

    def _records_autograd(self, args: Sequence[Any]) -> bool:
        """Whether this call is recording history a replay cannot supply."""

        if not any(_is_tensor(value) and value.requires_grad for value in args):
            return False
        return _grad_enabled()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs or self._state == "fallback" or self._records_autograd(args):
            return self._compiled(*args, **kwargs)
        if self._state == "live":
            return self._replay(args)
        with self._lock:
            if self._state == "cold":
                return self._capture(args)
            if self._state == "live":
                return self._replay(args)
            return self._compiled(*args)

    def _layout_key(self, args: Tuple[Any, ...]) -> str:
        """The node key for one argument layout.

        The signature names the layout, so alternating between two layouts
        reuses both recordings instead of recording a new node per call.
        """

        return f"{self._key}:{_shape_signature(args)!r}"

    def _layout_matches(self, args: Tuple[Any, ...]) -> bool:
        """Whether these are the objects the last call passed, unchanged.

        A region's arguments include every parameter it reads, so both the
        layout signature and the address check cost a scan of hundreds of
        values.  Identical objects with unchanged versions describe the same
        layout at the same storage, which is what those checks establish.
        """

        ids = tuple(map(id, args))
        versions = tuple(getattr(value, "_version", None) for value in args)
        memo = self._layout_memo
        if (
            memo is not None
            and memo[0] == ids
            and memo[1] == versions
            and all(_looks_tensor(value) for value in args)
        ):
            return True
        self._layout_memo = (ids, versions)
        self._layout_key_cache = None
        return False

    def _resolve_layout(self, args: Tuple[Any, ...]) -> Tuple[str, bool]:
        """The node key for this call, and whether the layout is certified."""

        if self._layout_matches(args) and self._layout_key_cache is not None:
            return self._layout_key_cache, True
        key = self._layout_key(args)
        self._layout_key_cache = key
        return key, False

    def _capture(self, args: Tuple[Any, ...], key: Optional[str] = None) -> Any:
        if key is None:
            key = self._layout_key(args) if self._trees else self._key
        first = self._state != "live"
        # Remembered for the C trampoline, which replays without passing
        # through ``__call__`` again: it needs the same autograd guard.
        self._needs_eager_when_grad = any(
            _is_tensor(value) and value.requires_grad for value in args
        )
        try:
            node = self._manager.capture(key, self._compiled, *args)
        except (CudaGraphError, NotImplementedError, RuntimeError, TypeError) as exc:
            if first:
                self._state = "fallback"
            log.warning(
                "cudagraph capture failed; running the artifact directly: %s", exc
            )
            return self._compiled(*args)
        if node.warmup_structure[0] == "unsupported":
            if first:
                self._state = "fallback"
                log.warning(
                    format_default_skip_message("unsupported output structure")
                )
            return self._compiled(*args)
        self._single_output = node.warmup_structure == ("single",)
        self._arity = len(args)
        self._state = "live"
        if first:
            self._install_fast_call()
        return self._replay(args, key=key, retry=False)

    def _replay(
        self, args: Tuple[Any, ...], *, key: Optional[str] = None, retry: bool = True
    ) -> Any:
        certified = False
        if key is None:
            if self._trees:
                key, certified = self._resolve_layout(args)
            else:
                key = self._key
                certified = self._layout_matches(args)
        if self._trees and not self._manager.has(key):
            return self._capture(args, key=key)
        try:
            outputs = self._manager.replay(key, *args, verify=not certified)
        except CudaGraphInputDrift:
            if self._trees:
                # The layout is new (e.g. a different arity): record it.
                if retry:
                    return self._capture(args, key=key)
                return self._compiled(*args)
            # A direct recording replays from the caller's own storage, so a
            # call whose tensors moved cannot use it.  Once that repeats, the
            # caller clearly moves its inputs every call: switch to staging.
            self._drifts += 1
            if self._drifts >= _DRIFT_LIMIT:
                return self._switch_to_staging(args)
            return self._compiled(*args)
        except (CudaGraphError, RuntimeError, TypeError) as exc:
            if self._trees and retry:
                # An evicted or failed node: drop what is left of it and
                # record the layout again.
                self._manager.clear(key)
                return self._capture(args, key=key)
            self._state = "fallback"
            log.warning(
                "cudagraph replay failed; running the artifact directly: %s", exc
            )
            return self._compiled(*args)
        self._drifts = 0
        outputs = [
            value.clone()
            if _looks_tensor(value)
            and (self._visible is None or index in self._visible)
            else value
            for index, value in enumerate(outputs)
        ]
        if self._single_output:
            return outputs[0]
        return tuple(outputs)

    def _switch_to_staging(self, args: Tuple[Any, ...]) -> Any:
        """Re-record against private buffers after repeated input drift.

        The staged manager copies each call's tensors onto buffers the
        recording reads, so a moved input costs a copy instead of the whole
        uncaptured artifact.  That only pays off when inputs move every call,
        which the drift count has established by now.
        """

        from .cudagraph_trees import CUDAGraphTreeManager

        CudagraphsBackend._managers.discard(self._manager)
        self._manager.clear()
        self._manager = CUDAGraphTreeManager(copy_outputs=(), native=self._native)
        CudagraphsBackend._managers.add(self._manager)
        self._trees = True
        self._state = "cold"
        self._drifts = 0
        self._layout_memo = None
        self._layout_key_cache = None
        log.info(
            "cudagraph inputs moved on every call; switching to staged replay"
        )
        return self._capture(args)

    def _replay_flat(self, values: List[Any], *, guard_autograd: bool) -> List[Any]:
        """Replay the recording against one full argument vector."""

        if guard_autograd and self._needs_eager_when_grad and self._records_autograd(values):
            # This call is recording; a replay would drop that history.
            # The artifact returns the caller's own shape of result, which
            # the trampoline expects flattened into a list.
            result = self._compiled(*values)
            if isinstance(result, (list, tuple)):
                return list(result)
            return [result]
        if self._trees:
            key = self._layout_key(tuple(values))
            if not self._manager.has(key):
                return self._replay(tuple(values), key=key)
            # The key was just built from these very arguments, so the
            # signature check would only repeat it.
            return self._manager.replay(key, *values, verify=False)
        # The entry that reached here certified every argument by identity and
        # version, which is what the signature and address checks establish.
        return self._manager.replay(self._key, *values, verify=False)

    def _build_fast_call(
        self, head: Optional[Sequence[Any]], nargs: int, *, guard_autograd: bool
    ) -> Any:
        """A native steady-state entry over the recording, or None."""

        try:
            import tensorplay

            installer = tensorplay._C._stax.install_call_trampoline
        except Exception:  # noqa: BLE001 - trampoline is an optional accelerator
            return None

        def replay_flat(values: List[Any]) -> List[Any]:
            return self._replay_flat(values, guard_autograd=guard_autograd)

        try:
            return installer(
                self,
                replay_flat,
                [],
                tensorplay.Tensor,
                nargs,
                1 if self._single_output else 2,
                False,
                0,
                None if head is None else list(head),
            )
        except Exception:  # noqa: BLE001 - trampoline is an optional accelerator
            return None

    def _replay_entry(self, *, head: Sequence[Any], nargs: int) -> Any:
        """A steady-state entry for a caller that holds only its own arguments.

        A recorded region reads the module state it was captured against and
        then the caller's arguments.  The state list is the same objects on
        every steady call, so the entry carries it and asks the C dispatcher
        to rebuild the argument vector; the caller's own arguments are the
        only part the argument memo has to check.
        """

        if self._state != "live":
            return None
        if self._trees:
            # A staged recording is keyed by the argument layout, which is a
            # signature over every value it reads; naming it again per call is
            # what the Python replay already does, so the entry would only add
            # a hand-off, not a shortcut.
            return None
        if self._needs_eager_when_grad and _grad_enabled():
            # The recording writes no autograd history, and this call would
            # have to carry one.  A later change of the ambient switch is
            # refused by the certificate the dispatcher installs alongside.
            return None
        return self._build_fast_call(head, nargs, guard_autograd=False)

    def _install_fast_call(self) -> None:
        """Bind the steady-state replay into the C call trampoline.

        Same contract as the lowerings use: identical objects with unchanged
        versions take the C fast path, everything else vectorcalls back into
        this wrapper.  Best effort - a build without the installer keeps the
        plain Python replay.
        """

        self._fast_call = self._build_fast_call(
            None, self._arity, guard_autograd=True
        )


def cudagraph_wrap(
    compiled: Any,
    gm: Any,
    example_inputs: Sequence[Any],
    *,
    dynamic: bool | None = None,
    native: Any = None,
    trees: Optional[bool] = None,
) -> tuple[Any, str | None]:
    """Wrap a compiled CUDA artifact for reduce-overhead replay.

    Returns ``(wrapped, None)`` when the artifact is eligible, otherwise
    ``(compiled, reason)`` with the skip reason.
    """
    reason = wrap_skip_reason(compiled, gm, example_inputs, dynamic=dynamic)
    if reason is not None:
        log.warning(reason)
        return compiled, reason
    return (
        CudagraphCompiledCallable(
            compiled, native=native, trees=trees,
            visible_outputs=_visible_output_indices(gm),
        ),
        None,
    )
