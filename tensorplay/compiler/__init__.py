"""Public compiler frontend for TensorPlay."""

from __future__ import annotations

import builtins
import contextlib
import functools
import hashlib
import inspect
import itertools
import threading
import types
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from tensorplay.graph._utils import (
    GraphCaptureError,
    _capture_disabled,
    _compiling,
    _native_capture_state,
    disallow_target as _disallow_target,
)

from . import config
from .annotations import Final, annotate, isinstance

if TYPE_CHECKING:
    from tensorplay.compiler._core.registry import InvalidBackend

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True)
class NestedCompileRegionOptions:
    """Backend selections for a nested compiler region."""

    fw_compiler: Callable[..., Any] | None = None
    bw_compiler: Callable[..., Any] | None = None


@dataclass
class _CallableRecord:
    original: Callable[..., Any]
    mode: str
    replacement: Callable[..., Any] | None = None
    reason: str | None = None
    recursive: bool = True
    options: Any = None
    max_reuse_entries: int = 8
    reuse_hash_fn: Callable[..., Any] | None = None
    adapter: Callable[..., Any] | None = None
    aliases: list[Callable[..., Any]] = field(default_factory=list)
    constant_ready: bool = False
    constant_value: Any = None
    disabled_runtime: Callable[..., Any] | None = None


@dataclass(frozen=True)
class _NestedRegionOptions:
    options: Any
    max_reuse_entries: int
    reuse_hash_fn: Callable[..., Any] | None
    fw_compiler: Callable[..., Any] | None = None
    bw_compiler: Callable[..., Any] | None = None


_records: dict[int, _CallableRecord] = {}
_records_lock = threading.RLock()
_global_patch_lock = threading.RLock()
_nested_region_ids = itertools.count()
_nested_region_ids_lock = threading.Lock()
_exporting: ContextVar[bool] = ContextVar(
    "tensorplay_compiler_exporting", default=False
)


__all__ = [
    "Final",
    "annotate",
    "assume_constant_result",
    "allow_in_graph",
    "build_aot_package",
    "compile",
    "config",
    "disallow_in_graph",
    "disable",
    "disable_capture",
    "export",
    "get_default_backend",
    "InvalidBackend",
    "is_compiling",
    "is_exporting",
    "isinstance",
    "list_backends",
    "list_mode_options",
    "load_aot_package",
    "lookup_backend",
    "mark_static",
    "NestedCompileRegionOptions",
    "nested_compile_region",
    "nonstrict_trace",
    "overload_method",
    "register_backend",
    "register_debug_backend",
    "register_experimental_backend",
    "reset",
    "set_default_backend",
    "substitute_in_graph",
    "unregister_backend",
    "unused",
]


def is_compiling() -> bool:
    """Return whether the current Python frame is being captured."""

    return _compiling.get() and not _capture_disabled.get()


def is_exporting() -> bool:
    """Return whether an export capture session is active."""

    return _exporting.get()


@contextlib.contextmanager
def _exporting_context() -> Iterator[None]:
    token = _exporting.set(True)
    native_entered = False
    try:
        native_entered = _native_capture_state(True, exporting=True)
        if not native_entered:
            raise GraphCaptureError(
                "TensorPlay native export state is unavailable"
            )
        yield
    finally:
        if native_entered:
            _native_capture_state(False, exporting=True)
        _exporting.reset(token)


@contextlib.contextmanager
def disable_capture() -> Iterator[None]:
    """Suspend the public capture state for a dynamic Python region."""

    token = _capture_disabled.set(True)
    native_entered = False
    try:
        native_entered = _native_capture_state(True, disabled=True)
        yield
    finally:
        if native_entered:
            _native_capture_state(False, disabled=True)
        _capture_disabled.reset(token)


def _mark_callable(fn: Callable[..., Any], attribute: str) -> Callable[..., Any]:
    if not callable(fn):
        raise TypeError(f"expected a callable, got {type(fn)!r}")
    try:
        setattr(fn, attribute, True)
    except (AttributeError, TypeError):
        pass
    return fn


def export(fn: Callable[..., Any] | None = None) -> Any:
    """Mark a module method as an entry point for definition tooling.

    The mark is recorded on the callable for tooling to read.  Capture does
    not need it: a region traces whatever the program calls, so a method
    marked here is captured exactly when it is reached, whether or not it
    carries the mark.
    """

    if fn is None:
        return lambda actual: _mark_callable(
            actual, "__tensorplay_compiler_export__"
        )
    return _mark_callable(fn, "__tensorplay_compiler_export__")


def unused(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a helper as one the scripting path leaves alone.

    The mark is recorded on the callable and nothing else changes: this build
    has no scripting path, and capture treats the callable like any other one,
    tracing it when it is reached.  Declaring a callable out of the graph is
    :func:`disallow_in_graph`.
    """

    return _mark_callable(fn, "__tensorplay_compiler_unused__")


def overload_method(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method declaration as an overload for definition tooling.

    Python keeps the last definition of a name, and so does capture: the mark
    says which declarations are alternatives, not which one a call reaches.
    """

    return _mark_callable(fn, "__tensorplay_compiler_overload__")


def _record_for(value: Any) -> _CallableRecord | None:
    record = _records.get(id(value))
    if record is None:
        return None
    if (
        record.original is value
        or record.adapter is value
        or any(alias is value for alias in record.aliases)
    ):
        return record
    return None


def _register_record(
    fn: Callable[..., Any],
    mode: str,
    *,
    replacement: Callable[..., Any] | None = None,
    reason: str | None = None,
    recursive: bool = True,
    options: Any = None,
    max_reuse_entries: int = 8,
    reuse_hash_fn: Callable[..., Any] | None = None,
) -> _CallableRecord:
    if not callable(fn):
        raise TypeError(f"expected a callable, got {type(fn)!r}")
    existing = _record_for(fn)
    if existing is not None:
        if existing.mode != mode:
            raise ValueError(
                f"callable {fn!r} cannot be marked as both "
                f"{existing.mode!r} and {mode!r}"
            )
        if mode == "substitute":
            raise ValueError(f"a substitution is already registered for {fn!r}")
        return existing

    record = _CallableRecord(
        original=fn,
        mode=mode,
        replacement=replacement,
        reason=reason,
        recursive=recursive,
        options=options,
        max_reuse_entries=max_reuse_entries,
        reuse_hash_fn=reuse_hash_fn,
    )
    with _records_lock:
        _records[id(fn)] = record
    try:
        setattr(fn, "__tensorplay_compiler_mode__", mode)
    except (AttributeError, TypeError):
        pass
    return record


def _iter_proxies(value: Any) -> Iterator[Any]:
    """Yield the proxies inside an argument structure.

    This runs on every eager call of a compiled region, over argument lists
    that hold hundreds of tensors, so it walks an explicit stack rather than
    recursing: a plain tensor costs one exact-type test and no frame.
    """

    from tensorplay.graph.proxy import Proxy

    stack = [value]
    while stack:
        item = stack.pop()
        kind = type(item)
        if kind is Proxy:
            yield item
        elif kind is tuple or kind is list:
            stack.extend(item)
        elif kind is dict:
            for key, entry in item.items():
                stack.append(entry)
                stack.append(key)
        elif kind is slice:
            stack.extend((item.start, item.stop, item.step))


def _tracer_for(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    proxies = list(_iter_proxies((args, kwargs)))
    if not proxies:
        return None
    tracer = proxies[0].tracer
    if any(proxy.tracer is not tracer for proxy in proxies[1:]):
        raise GraphCaptureError("cannot combine values from different traces")
    return tracer


_UNRESOLVED = object()


def _nested_region_name() -> str:
    with _nested_region_ids_lock:
        return f"_tensorplay_nested_region_{next(_nested_region_ids)}"


def _nested_target(value: Any) -> Any:
    if callable(getattr(value, "forward", None)) and callable(
        getattr(value, "named_children", None)
    ):
        return value.forward
    return value


def _nested_sample_inputs(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(_nested_target(target))
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError) as exc:
        raise GraphCaptureError(
            "nested compile region arguments cannot be bound to its signature"
        ) from exc
    return dict(bound.arguments)


def _nested_output_template(value: Any) -> Any:
    import tensorplay

    if isinstance(value, tensorplay.Tensor):
        return ("tensor",)
    if isinstance(value, tuple):
        return ("tuple", tuple(_nested_output_template(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_nested_output_template(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (key, _nested_output_template(item))
                for key, item in value.items()
            ),
        )
    raise GraphCaptureError(
        "nested compile regions must return tensors or tensor containers"
    )


def _nested_flatten_output(value: Any, template: Any) -> list[Any]:
    kind = template[0]
    if kind == "tensor":
        import tensorplay

        if not isinstance(value, tensorplay.Tensor):
            raise RuntimeError("nested compile region returned a non-tensor output")
        return [value]
    if kind == "tuple":
        if not isinstance(value, tuple) or len(value) != len(template[1]):
            raise RuntimeError("nested compile region returned an invalid tuple output")
        result: list[Any] = []
        for item, item_template in zip(value, template[1]):
            result.extend(_nested_flatten_output(item, item_template))
        return result
    if kind == "list":
        if not isinstance(value, list) or len(value) != len(template[1]):
            raise RuntimeError("nested compile region returned an invalid list output")
        result = []
        for item, item_template in zip(value, template[1]):
            result.extend(_nested_flatten_output(item, item_template))
        return result
    if kind == "dict":
        if not isinstance(value, dict):
            raise RuntimeError("nested compile region returned an invalid mapping output")
        result = []
        for key, item_template in template[1]:
            if key not in value:
                raise RuntimeError(
                    f"nested compile region omitted output key {key!r}"
                )
            result.extend(_nested_flatten_output(value[key], item_template))
        return result
    raise RuntimeError(f"unknown nested output template kind {kind!r}")


def _nested_value_key(value: Any) -> Any:
    import tensorplay

    if isinstance(value, tensorplay.Tensor):
        try:
            shape = tuple(int(item) for item in value.shape)
            data_digest = hashlib.sha256(repr(value).encode()).hexdigest()
        except Exception as exc:
            raise GraphCaptureError(
                "nested compile region input cannot be fingerprinted"
            ) from exc
        return (
            "tensor",
            type(value),
            shape,
            repr(value.dtype),
            repr(value.device),
            bool(getattr(value, "requires_grad", False)),
            data_digest,
        )
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value), value)
    if isinstance(value, tuple):
        return (tuple, tuple(_nested_value_key(item) for item in value))
    if isinstance(value, list):
        return (list, tuple(_nested_value_key(item) for item in value))
    if isinstance(value, dict):
        return (
            dict,
            tuple(
                sorted(
                    (
                        (_nested_value_key(key), _nested_value_key(item))
                        for key, item in value.items()
                    ),
                    key=repr,
                )
            ),
        )
    try:
        representation = repr(value)
    except Exception as exc:
        raise GraphCaptureError(
            "nested compile region input cannot be fingerprinted"
        ) from exc
    return (type(value), hashlib.sha256(representation.encode()).hexdigest())


def _nested_reuse_key(
    record: _CallableRecord,
    sample_args: tuple[Any, ...],
    sample_kwargs: dict[str, Any],
) -> Any:
    if record.reuse_hash_fn is not None:
        try:
            value = record.reuse_hash_fn(*sample_args, **sample_kwargs)
        except Exception as exc:
            raise GraphCaptureError(
                "nested compile region reuse_hash_fn failed"
            ) from exc
        if type(value) is not int:
            raise GraphCaptureError(
                "nested compile region reuse_hash_fn must return an integer"
            )
        return ("hash", value)
    return ("automatic", _nested_value_key((sample_args, sample_kwargs)))


def _invoke_nested_region(subgraph: Any, *args: Any, **kwargs: Any) -> Any:
    return subgraph(*args, **kwargs)


def _disabled_runtime(record: _CallableRecord) -> Callable[..., Any]:
    runtime = record.disabled_runtime
    if runtime is not None:
        return runtime

    if record.recursive:

        @functools.wraps(record.original)
        def runtime(*args: Any, **kwargs: Any) -> Any:
            with disable_capture():
                return record.original(*args, **kwargs)

    else:

        @functools.wraps(record.original)
        def runtime(*args: Any, **kwargs: Any) -> Any:
            return record.original(*args, **kwargs)

    record.disabled_runtime = runtime
    return runtime


def _capture_opaque_region(
    record: _CallableRecord,
    tracer: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    sample_args = _resolve_sample(args)
    sample_kwargs = _resolve_sample(kwargs)
    if sample_args is _UNRESOLVED or sample_kwargs is _UNRESOLVED:
        raise GraphCaptureError(
            "disabled callable needs sample values for every tensor input"
        )
    runtime = (
        _disabled_runtime(record)
        if record.mode == "disabled"
        else record.original
    )
    try:
        sample_output = runtime(*sample_args, **sample_kwargs)
        template = _nested_output_template(sample_output)
        flat_outputs = _nested_flatten_output(sample_output, template)
    except GraphCaptureError:
        raise
    except Exception as exc:
        raise GraphCaptureError(
            "opaque compiler callable could not produce a tensor output during capture"
        ) from exc

    node = tracer.graph.call_function(runtime, args, kwargs)
    proxy = tracer.proxy(node)
    custom = dict(node.meta.get("custom") or {})
    custom["opaque_region"] = True
    custom["opaque_callable"] = runtime
    custom["opaque_mode"] = record.mode
    if record.mode == "disabled":
        custom["disabled_region"] = True
        custom["disabled_recursive"] = record.recursive
    custom["nested_output_count"] = len(flat_outputs)
    custom["nested_output_template"] = template
    custom["nested_region_compiled"] = False
    node.meta["custom"] = custom
    tracer._node_samples[node.name] = sample_output
    node.meta["val"] = sample_output
    return proxy


def _capture_nested_region(
    record: _CallableRecord,
    tracer: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    sample_args = _resolve_sample(args)
    sample_kwargs = _resolve_sample(kwargs)
    if sample_args is _UNRESOLVED or sample_kwargs is _UNRESOLVED:
        raise GraphCaptureError(
            "nested compile region needs sample values for every input"
        )

    cache = getattr(tracer, "_tensorplay_nested_regions", None)
    if cache is None:
        cache = {}
        tracer._tensorplay_nested_regions = cache
    sample_inputs = _nested_sample_inputs(record.original, sample_args, sample_kwargs)
    sample_key = _nested_reuse_key(record, sample_args, sample_kwargs)
    cache_key = (id(record), sample_key)
    cached = cache.get(cache_key)
    if cached is None:
        entries = sum(1 for key in cache if key[0] == id(record))
        if entries >= record.max_reuse_entries:
            raise GraphCaptureError(
                "nested compile region reuse limit reached"
            )
        from tensorplay.graph import Tracer

        child_tracer = Tracer(execute=True)
        child = child_tracer.trace(
            record.original,
            sample_inputs=sample_inputs,
        )
        sample_output = child_tracer.resolve_sample(
            child.graph.output_node.args[0]
        )
        from tensorplay.graph.tracer import _UNRESOLVED as _TRACER_UNRESOLVED

        if (
            sample_output is None
            or sample_output is _TRACER_UNRESOLVED
        ):
            raise GraphCaptureError(
                "nested compile region output has no executable sample"
            )
        template = _nested_output_template(sample_output)
        cached = (child, sample_output, template)
        cache[cache_key] = cached
    child, sample_output, template = cached

    graph_attr = _nested_region_name()
    tracer._graph_attrs[graph_attr] = child
    graph_attr_proxy = tracer.proxy(tracer.graph.get_attr(graph_attr))
    proxy = tracer.create_proxy(
        "call_function",
        _invoke_nested_region,
        (graph_attr_proxy, *args),
        kwargs,
    )
    flat_outputs = _nested_flatten_output(sample_output, template)
    custom = dict(proxy.node.meta.get("custom") or {})
    custom["nested_region_config"] = _nested_region_options(record)
    custom["nested_region_attr"] = graph_attr
    custom["nested_output_count"] = len(flat_outputs)
    custom["nested_output_template"] = template
    custom["nested_region_compiled"] = False
    proxy.node.meta["custom"] = custom
    tracer._node_samples[proxy.node.name] = sample_output
    proxy.node.meta["val"] = sample_output
    return proxy


def _resolve_sample(value: Any) -> Any:
    from tensorplay.graph.proxy import Proxy

    if builtins.isinstance(value, Proxy):
        sample = value._sample()
        return _UNRESOLVED if sample is None else sample
    if builtins.isinstance(value, tuple):
        result = tuple(_resolve_sample(item) for item in value)
        return _UNRESOLVED if any(item is _UNRESOLVED for item in result) else result
    if builtins.isinstance(value, list):
        result = [_resolve_sample(item) for item in value]
        return _UNRESOLVED if any(item is _UNRESOLVED for item in result) else result
    if builtins.isinstance(value, dict):
        resolved_items = [
            (_resolve_sample(key), _resolve_sample(item))
            for key, item in value.items()
        ]
        if any(
            key is _UNRESOLVED or item is _UNRESOLVED
            for key, item in resolved_items
        ):
            return _UNRESOLVED
        return dict(resolved_items)
    if builtins.isinstance(value, slice):
        start = _resolve_sample(value.start)
        stop = _resolve_sample(value.stop)
        step = _resolve_sample(value.step)
        if any(item is _UNRESOLVED for item in (start, stop, step)):
            return _UNRESOLVED
        return slice(start, stop, step)
    return value


def _nested_region_options(record: _CallableRecord) -> _NestedRegionOptions:
    options = record.options
    return _NestedRegionOptions(
        options=options,
        max_reuse_entries=record.max_reuse_entries,
        reuse_hash_fn=record.reuse_hash_fn,
        fw_compiler=getattr(options, "fw_compiler", None),
        bw_compiler=getattr(options, "bw_compiler", None),
    )


def _capture_adapter(record: _CallableRecord) -> Callable[..., Any]:
    if record.adapter is not None:
        return record.adapter

    original = record.original

    @functools.wraps(original)
    def adapter(*args: Any, **kwargs: Any) -> Any:
        tracer = _tracer_for(args, kwargs)
        if tracer is None:
            return original(*args, **kwargs)

        if record.mode == "disabled":
            return _capture_opaque_region(record, tracer, args, kwargs)

        if record.mode == "disallow":
            name = getattr(record.original, "__name__", None) or type(
                record.original
            ).__name__
            raise GraphCaptureError(
                f"{name} is declared disallowed in a captured region; call it "
                "outside the region it must not join"
            )

        if record.mode in {"allow", "nonstrict"}:
            return _capture_opaque_region(record, tracer, args, kwargs)

        if record.mode == "constant":
            sample_args = _resolve_sample(args)
            sample_kwargs = _resolve_sample(kwargs)
            if sample_args is _UNRESOLVED or sample_kwargs is _UNRESOLVED:
                raise GraphCaptureError(
                    "a constant-result callable needs sample values during capture"
                )
            with _records_lock:
                if record.constant_ready:
                    return record.constant_value
            value = original(*sample_args, **sample_kwargs)
            with _records_lock:
                if not record.constant_ready:
                    record.constant_value = value
                    record.constant_ready = True
                return record.constant_value

        if record.mode == "substitute":
            replacement = record.replacement
            if replacement is None:
                raise RuntimeError("substitution has no implementation")
            return replacement(*args, **kwargs)

        if record.mode == "nested":
            return _capture_nested_region(record, tracer, args, kwargs)

        proxy = tracer.create_proxy("call_function", original, args, kwargs)
        return proxy

    record.adapter = adapter
    with _records_lock:
        _records[id(adapter)] = record
    return adapter


def allow_in_graph(fn: Any) -> Any:
    """Capture a callable as one graph operation without entering its body."""

    if builtins.isinstance(fn, (list, tuple)):
        return [allow_in_graph(item) for item in fn]
    if not callable(fn):
        raise AssertionError("allow_in_graph expects a callable")
    _register_record(fn, "allow")
    return fn


def disallow_in_graph(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Declare that a callable must not be absorbed into a captured graph.

    A captured region that reaches ``fn`` is refused rather than recorded.
    The marker suits calls whose effect the graph cannot carry -- an
    in-place update of a caller-owned buffer, say -- because recording them
    would repeat the effect once per compiled call while the surrounding
    capture already performed it on the example.

    The callable is returned unchanged; outside capture the marker has no
    effect.
    """

    if builtins.isinstance(fn, (list, tuple)):
        return [disallow_in_graph(item) for item in fn]
    if not callable(fn):
        raise AssertionError("disallow_in_graph expects a callable")
    _register_record(fn, "disallow")
    _disallow_target(fn)
    return fn


def mark_static(tensor: Any, dim: int | None = None) -> Any:
    """Pin one tensor's extents to the sizes seen at compile time.

    Called before ``compile``, the mark tells the shape policy that the
    tensor's dimensions (or the one named by ``dim``) may not be treated as
    symbolic, even when the compile call asks for dynamic shapes.  Called
    while a region is being captured, the mark pins the extent the value had
    on the example and keeps a condition on it; reads of that extent made
    afterwards answer with the concrete size.

    A tensor with no marks keeps the policy of the compile call: symbolic
    when it asks for dynamic shapes, concrete otherwise.
    """

    from tensorplay.graph.proxy import Proxy

    if isinstance(tensor, Proxy):
        hook = getattr(tensor.tracer, "mark_static_dim", None)
        if hook is not None:
            hook(tensor, dim)
        return tensor

    shape = getattr(tensor, "shape", None)
    if shape is None:
        if callable(getattr(tensor, "forward", None)):
            # A module's attributes are read through the graph's own state
            # guards, so a static module needs no separate record here.
            return tensor
        raise TypeError(
            f"mark_static expects a tensor or a module, got {type(tensor)!r}"
        )

    rank = len(shape)
    axes = range(rank) if dim is None else (dim,)
    marked = set(getattr(tensor, "_tensorplay_static_dims", ()))
    for axis in axes:
        if type(axis) is not int:
            raise TypeError(f"mark_static expects an int dim, got {type(axis)!r}")
        index = axis + rank if axis < 0 else axis
        if index < 0 or index >= rank:
            raise IndexError(
                f"dimension {axis} is out of range for a "
                f"{rank}-dimensional tensor"
            )
        marked.add(index)
    tensor._tensorplay_static_dims = tuple(sorted(marked))
    return tensor


def nonstrict_trace(traceable_fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Capture a callable as an opaque operation with structured arguments."""

    if not callable(traceable_fn):
        raise AssertionError("nonstrict_trace expects a callable")
    record = _register_record(traceable_fn, "nonstrict")
    return _capture_adapter(record)


def assume_constant_result(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Declare that a callable returns one stable value across captures."""

    record = _register_record(fn, "constant")
    return record.original


_FUNCTION_TYPES = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
)


def _is_function(value: Any) -> bool:
    return builtins.isinstance(value, _FUNCTION_TYPES)


def _check_substitution_signature(
    original_fn: Callable[..., Any], replacement: Callable[..., Any]
) -> None:
    try:
        original_signature = inspect.signature(original_fn)
    except (TypeError, ValueError):
        return
    try:
        replacement_signature = inspect.signature(replacement)
    except (TypeError, ValueError) as exc:
        raise TypeError("unable to inspect the replacement signature") from exc

    def signature_identity(
        signature: inspect.Signature,
    ) -> tuple[tuple[str, ...], set[str], dict[str, Any]]:
        parameters = tuple(signature.parameters.values())
        return (
            tuple(
                parameter.name
                for parameter in parameters
                if parameter.kind
                not in {
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            ),
            {
                parameter.name
                for parameter in parameters
                if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            },
            {
                parameter.name: parameter.default
                for parameter in parameters
                if parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            },
        )

    wildcard_signature = inspect.signature(lambda *args, **kwargs: None)
    original_identity = signature_identity(original_signature)
    replacement_identity = signature_identity(replacement_signature)
    wildcard_identity = signature_identity(wildcard_signature)
    if (
        original_identity != replacement_identity
        and original_identity != wildcard_identity
        and replacement_identity != wildcard_identity
    ):
        raise TypeError(
            f"substitution signature {replacement_signature} does not match "
            f"{original_signature}"
        )


def substitute_in_graph(
    original_fn: Callable[_P, _R],
    *,
    can_constant_fold_through: bool = False,
    skip_signature_check: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Register a graph-time implementation for a callable.

    ``can_constant_fold_through`` asks that arguments which are plain Python
    constants be answered by the original function rather than by recording a
    call.  Capture here already evaluates a call whose arguments are all
    constants while tracing, so a constant-only call never becomes a node: the
    shortcut this asks for is what capture does anyway, and the argument is
    accepted as a hint rather than as a mode to switch between.

    ``skip_signature_check`` skips the registration-time comparison between
    the two signatures, for a replacement whose arity has to differ.
    """

    if not _is_function(original_fn):
        raise TypeError(
            "substitute_in_graph expects a function but got "
            f"{type(original_fn)!r}"
        )
    if not builtins.isinstance(can_constant_fold_through, bool):
        raise TypeError("can_constant_fold_through must be a bool")
    if not builtins.isinstance(skip_signature_check, bool):
        raise TypeError("skip_signature_check must be a bool")

    def decorator(
        replacement: Callable[_P, _R],
    ) -> Callable[_P, _R]:
        if not _is_function(replacement):
            raise TypeError(
                "@substitute_in_graph(...) expects a function but got "
                f"{type(replacement)!r}"
            )
        if not skip_signature_check:
            _check_substitution_signature(original_fn, replacement)
        _register_record(original_fn, "substitute", replacement=replacement)
        record = _record_for(original_fn)
        if record is None:
            raise RuntimeError("substitution registration did not create a record")

        @functools.wraps(replacement)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return original_fn(*args, **kwargs)

        record.aliases.append(wrapped)
        with _records_lock:
            _records[id(wrapped)] = record
        return wrapped

    return decorator


def disable(
    fn: Callable[_P, _R] | None = None,
    recursive: bool = True,
    *,
    reason: str | None = None,
) -> Any:
    """Prevent capture from entering a callable."""

    if not builtins.isinstance(recursive, bool):
        raise TypeError("recursive must be a bool")

    def decorate(actual: Callable[_P, _R]) -> Callable[_P, _R]:
        if not callable(actual):
            raise AssertionError("fn must be callable")
        record = _register_record(
            actual,
            "disabled",
            reason=reason,
            recursive=recursive,
        )
        return _capture_adapter(record)

    if fn is None:
        return decorate
    return decorate(fn)


def nested_compile_region(
    fn: Callable[_P, _R] | None = None,
    *,
    options: Any = None,
    max_reuse_entries: int = 8,
    reuse_hash_fn: Callable[..., Any] | None = None,
) -> Any:
    """Mark a callable as a reusable nested graph region."""

    if not builtins.isinstance(max_reuse_entries, int):
        raise TypeError("max_reuse_entries must be an integer")
    if isinstance(max_reuse_entries, bool):
        raise TypeError("max_reuse_entries must be an integer")
    if max_reuse_entries < 1:
        raise ValueError("max_reuse_entries must be positive")
    if reuse_hash_fn is not None and not callable(reuse_hash_fn):
        raise TypeError("reuse_hash_fn must be callable")
    if options is not None and not isinstance(options, NestedCompileRegionOptions):
        raise TypeError(
            "options must be a NestedCompileRegionOptions instance or None"
        )
    if options is not None and any(
        compiler is not None and not callable(compiler)
        for compiler in (options.fw_compiler, options.bw_compiler)
    ):
        raise TypeError("nested region compiler options must be callable or None")

    def decorate(actual: Callable[_P, _R]) -> Callable[_P, _R]:
        record = _register_record(
            actual,
            "nested",
            options=options,
            max_reuse_entries=max_reuse_entries,
            reuse_hash_fn=reuse_hash_fn,
        )
        return _capture_adapter(record)

    if fn is None:
        return decorate
    return decorate(fn)


def _callable_globals(root: Any) -> dict[str, Any] | None:
    target = root
    forward = getattr(root, "forward", None)
    if callable(forward) and callable(getattr(root, "named_children", None)):
        target = forward
    if inspect.ismethod(target):
        target = target.__func__
    namespace = getattr(target, "__globals__", None)
    if namespace is not None:
        return namespace
    call = getattr(target, "__call__", None)
    namespace = getattr(call, "__globals__", None)
    return namespace


def _callable_function(target: Any) -> Any:
    """The function a callable body runs, or ``target`` when there is none."""

    if callable(getattr(target, "forward", None)) and callable(
        getattr(target, "named_children", None)
    ):
        target = target.forward
    if inspect.ismethod(target):
        return target.__func__
    return target


def _callable_values(root: Any) -> Iterator[Any]:
    """Values a callable reads by name: module globals and closure cells."""

    namespace = _callable_globals(root)
    if namespace is not None:
        yield from namespace.values()
    closure = getattr(_callable_function(root), "__closure__", None)
    if closure:
        for cell in closure:
            try:
                yield cell.cell_contents
            except ValueError:
                continue


def _capture_records_in(root: Any) -> tuple[_CallableRecord, ...]:
    found: dict[int, _CallableRecord] = {}
    for value in _callable_values(root):
        record = _record_for(value)
        if record is not None:
            found[id(record)] = record
    return tuple(found.values())


@contextlib.contextmanager
def _patched_capture_globals(root: Any) -> Iterator[None]:
    namespace = _callable_globals(root)
    if namespace is None:
        yield
        return
    _global_patch_lock.acquire()
    try:
        patches: list[tuple[str, Any, Any]] = []
        for name, value in list(namespace.items()):
            record = _record_for(value)
            if record is None:
                continue
            adapter = _capture_adapter(record)
            if value is adapter:
                continue
            namespace[name] = adapter
            patches.append((name, value, adapter))
        try:
            yield
        finally:
            for name, previous, adapter in reversed(patches):
                if namespace.get(name) is adapter:
                    namespace[name] = previous
    finally:
        _global_patch_lock.release()


def _unwrap_callable(value: Any) -> tuple[Any, _CallableRecord | None]:
    record = _record_for(value)
    if record is None:
        return value, None
    if record.mode == "substitute":
        return _capture_adapter(record), record
    return record.original, record


def _capture_root(root: Any) -> Any:
    if not any(_record_for(value) is not None for value in _callable_values(root)):
        return root

    target = _callable_function(root)
    bound_instance = None
    if inspect.ismethod(root):
        bound_instance = root.__self__
    if not builtins.isinstance(target, types.FunctionType):
        return root

    namespace = _callable_globals(root)
    patched_namespace = dict(namespace) if namespace is not None else {}
    for name, value in list(patched_namespace.items()):
        record = _record_for(value)
        if record is not None:
            patched_namespace[name] = _capture_adapter(record)
    closure = target.__closure__
    if closure is not None:
        patched_cells = []
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                patched_cells.append(cell)
                continue
            record = _record_for(value)
            if record is not None:
                patched_cells.append(types.CellType(_capture_adapter(record)))
            else:
                patched_cells.append(cell)
        closure = tuple(patched_cells)
    captured = types.FunctionType(
        target.__code__,
        patched_namespace,
        target.__name__,
        target.__defaults__,
        closure,
    )
    captured.__kwdefaults__ = target.__kwdefaults__
    captured.__annotations__ = dict(getattr(target, "__annotations__", {}))
    captured.__dict__.update(getattr(target, "__dict__", {}))
    captured.__module__ = target.__module__
    captured.__qualname__ = target.__qualname__
    captured.__doc__ = target.__doc__
    if bound_instance is not None:
        return types.MethodType(captured, bound_instance)
    return captured


def _wrap_compiled_callable(
    compiled: Callable[..., Any],
    root: Any,
) -> Callable[..., Any]:
    records = _capture_records_in(root)
    target = root.__func__ if inspect.ismethod(root) else root
    namespace = _callable_globals(root)
    isolated = False
    if namespace is not None and builtins.isinstance(target, types.FunctionType):
        isolated = True
        for value in namespace.values():
            record = _record_for(value)
            if record is not None and value is not _capture_adapter(record):
                isolated = False
                break
    if not records or isolated:
        return compiled

    @functools.wraps(compiled)
    def run(*args: Any, **kwargs: Any) -> Any:
        with _patched_capture_globals(root):
            return compiled(*args, **kwargs)

    run.__dict__.update(getattr(compiled, "__dict__", {}))
    run._tensorplay_compiler_inner = compiled  # type: ignore[attr-defined]
    return run


def compile(
    model: Callable[..., Any] | None = None,
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str | Callable[..., Any] | None = None,
    mode: str | None = None,
    options: dict[str, Any] | None = None,
    name: str | None = None,
    disable: bool = False,
    recompile_limit: int | None = None,
    strict_native: bool = False,
    dynamic_shapes: Any = None,
) -> Callable[..., Any]:
    """Compile a callable through the TensorPlay capture and lowering pipeline."""

    if dynamic_shapes is not None:
        if dynamic is not None:
            raise RuntimeError("dynamic and dynamic_shapes cannot both be specified")
        if not builtins.isinstance(dynamic_shapes, bool):
            # A per-input specification names the dimensions that may vary;
            # it is validated against the program's arguments and examples by
            # the capture frontend.  With no program yet, the specification is
            # kept for the decorated call, which knows the argument names.
            if model is not None:
                from ._core.shape_policy import normalize_dynamic_shapes
                from ._core.shape_policy import parameter_names

                normalize_dynamic_shapes(dynamic_shapes, parameter_names(model))
    if mode is not None and options is not None:
        raise RuntimeError("Either mode or options can be specified, but not both")
    if options is not None and not builtins.isinstance(options, dict):
        raise TypeError(f"options must be a dict, got {type(options)!r}")
    if not builtins.isinstance(config.assume_static_by_default, bool):
        raise TypeError("config.assume_static_by_default must be a bool")
    if not builtins.isinstance(config.verbose, bool):
        raise TypeError("config.verbose must be a bool")
    if not builtins.isinstance(config.fail_on_recompile_limit_hit, bool):
        raise TypeError("config.fail_on_recompile_limit_hit must be a bool")
    if not builtins.isinstance(config.force_disable_caches, bool):
        raise TypeError("config.force_disable_caches must be a bool")
    if config.dynamic_shapes is not None and not builtins.isinstance(
        config.dynamic_shapes, bool
    ):
        raise TypeError("config.dynamic_shapes must be a bool or None")
    accumulated_limit = config.accumulated_recompile_limit
    if (
        not builtins.isinstance(accumulated_limit, int)
        or isinstance(accumulated_limit, bool)
        or accumulated_limit < 1
    ):
        raise ValueError(
            "config.accumulated_recompile_limit must be a positive integer"
        )
    configured_limit = (
        config.recompile_limit if recompile_limit is None else recompile_limit
    )
    if configured_limit is not None:
        if not builtins.isinstance(configured_limit, int) or isinstance(
            configured_limit, bool
        ):
            raise TypeError("recompile_limit must be an integer")
        if configured_limit < 1:
            raise ValueError("recompile_limit must be positive")

    if model is None:
        return functools.partial(
            compile,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            name=name,
            disable=disable,
            recompile_limit=recompile_limit,
            strict_native=strict_native,
            dynamic_shapes=dynamic_shapes,
        )
    if not callable(model):
        raise TypeError(f"compile() expected a callable, got {type(model)!r}")

    if disable:
        return model

    model, record = _unwrap_callable(model)
    if record is not None and record.mode == "disabled":
        return model

    if dynamic is None and dynamic_shapes is None:
        dynamic = config.dynamic_shapes
        if dynamic is None and not config.assume_static_by_default:
            dynamic = True
    if recompile_limit is None:
        recompile_limit = config.recompile_limit
    if recompile_limit is not None:
        if not builtins.isinstance(recompile_limit, int):
            raise TypeError("recompile_limit must be an integer")
        if recompile_limit < 1:
            raise ValueError("recompile_limit must be positive")

    from tensorplay.compiler import _core

    capture_root = _capture_root(model)
    compiled = _core.compile(
        capture_root,
        fullgraph=fullgraph,
        dynamic=dynamic,
        backend=backend,
        mode=mode,
        options=options,
        name=name,
        disable=False,
        recompile_limit=recompile_limit,
        strict_native=strict_native,
        dynamic_shapes=dynamic_shapes,
    )
    return _wrap_compiled_callable(compiled, capture_root)


def reset() -> None:
    """Clear compiler specializations and capture-time constant values."""

    from tensorplay.compiler import _core

    _core.reset()
    with _records_lock:
        for record in _records.values():
            record.constant_ready = False
            record.constant_value = None


def build_aot_package(
    fn: Callable[..., Any],
    example_inputs: list[Any],
    path: str,
    *,
    options: dict[str, bool] | None = None,
    overwrite: bool = False,
) -> str:
    """Write ``fn`` for ``example_inputs`` as a deployment package.

    The package is a directory holding the region's shared object, its
    constants, and the metadata a caller needs to run it without this
    compiler: ``load_aot_package`` reads it back, and a C++ caller includes
    the generated header instead.  Regions whose schedule has no C++ form
    cannot be packaged and raise :class:`PackageError`.
    """

    from .backends.stax.aot_package import build_package

    return build_package(
        fn, example_inputs, path, options=options, overwrite=overwrite
    )


def load_aot_package(path: str) -> Any:
    """Load the deployment package written under ``path``."""

    from .backends.stax.aot_package import load_package

    return load_package(path)


def list_backends(
    exclude_tags: tuple[str, ...] | list[str] | None = ("debug", "experimental"),
    *,
    include_unavailable: bool = False,
) -> list[str]:
    """Return registered backend names accepted by :func:`compile`.

    Backends whose optional dependencies are missing are hidden unless
    ``include_unavailable`` is set; selecting one by name produces an error
    naming what to install.
    """

    from tensorplay.compiler import _core

    return _core.list_backends(
        exclude_tags=exclude_tags, include_unavailable=include_unavailable
    )


def lookup_backend(backend: str | Callable[..., Any]) -> Callable[..., Any]:
    """Resolve a backend name or validate a backend callable."""

    from tensorplay.compiler import _core

    return _core.lookup_backend(backend)


def register_backend(*args: Any, **kwargs: Any) -> Any:
    """Register a backend in the TensorPlay compiler registry."""

    from tensorplay.compiler import _core

    return _core.register_backend(*args, **kwargs)


def register_debug_backend(*args: Any, **kwargs: Any) -> Any:
    """Register a backend tagged for diagnostics."""

    from tensorplay.compiler import _core

    return _core.register_debug_backend(*args, **kwargs)


def register_experimental_backend(*args: Any, **kwargs: Any) -> Any:
    """Register a backend tagged for experimental use."""

    from tensorplay.compiler import _core

    return _core.register_experimental_backend(*args, **kwargs)


def unregister_backend(name: str) -> None:
    """Remove a named backend from the compiler registry."""

    from tensorplay.compiler import _core

    _core.unregister_backend(name)


def set_default_backend(backend: str | Callable[..., Any] | None) -> None:
    """Set the backend used when :func:`compile` receives no backend name."""

    from tensorplay.compiler import _core

    _core.set_default_backend(backend)


def get_default_backend() -> str | Callable[..., Any]:
    """Return the currently selected default backend."""

    from tensorplay.compiler import _core

    return _core.get_default_backend()


def get_backend_capabilities(backend: str | Callable[..., Any]) -> Any:
    """Return the :class:`BackendCapabilities` a backend declares."""

    from tensorplay.compiler._core.registry import get_backend_capabilities as _get

    return _get(backend)


def list_mode_options(mode: str | None = None) -> dict[str, Any]:
    """Return the optimization options each compile ``mode`` selects.

    With ``mode`` set, returns that mode's option patch; with ``mode`` unset,
    returns the full mode-to-options mapping.  Unknown modes raise.
    """
    from tensorplay.compiler.backends.stax.backend import list_mode_options as _list

    return _list(mode)


def __getattr__(name: str) -> Any:
    # The registry module is loaded on demand so importing the facade stays
    # free of backend-host imports.
    if name == "InvalidBackend":
        from tensorplay.compiler._core.registry import InvalidBackend

        return InvalidBackend
    if name == "BackendCapabilities":
        from tensorplay.compiler._core.registry import BackendCapabilities

        return BackendCapabilities
    if name == "CORE_BACKEND_CONTRACT_VERSION":
        from tensorplay.compiler._core.registry import CORE_BACKEND_CONTRACT_VERSION

        return CORE_BACKEND_CONTRACT_VERSION
    if name == "declares_capabilities":
        from tensorplay.compiler._core.registry import declares_capabilities

        return declares_capabilities
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
