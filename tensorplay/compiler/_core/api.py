"""Public compiler orchestration for TensorPlay.

Capture, backend selection, and execution concerns are separate. A backend is
never asked to discover Python control flow; it only receives a captured
``GraphModule`` and example inputs.
"""

from __future__ import annotations

import contextlib
import dis
import functools
import inspect
import re
import threading
import types
from typing import Any, Callable
import weakref
from weakref import WeakSet

from tensorplay.graph import (
    GraphCaptureError,
    GraphModule,
    Tracer,
    compiler_context,
)
from tensorplay.graph._utils import _capture_disabled
from .guards import GuardChain, build_guard_chain, format_recompile_reasons
from tensorplay.graph.passes import (
    ConstFold,
    DeadCodeElimination,
    DecomposePass,
    NormalizeOperators,
    PassManager,
    PointwiseFusionHint,
    ShapeProp,
)
from tensorplay.graph.passes.dialect.common import CSEPass, get_CSE_banned_ops
from tensorplay.graph.passes.regional_compile_invoke_subgraph import (
    regional_compile_invoke_subgraph,
)
from .registry import CompilerFn, get_default_backend, lookup_backend
from .region_cache import (
    load_region,
    region_key as make_region_key,
    store_region,
)
from .shape_policy import shape_policy_signature


_compiled_wrappers: WeakSet[Any] = WeakSet()
_DEFAULT_RECOMPILE_LIMIT = 8


def _compiler_context() -> Any:
    """Capture context owned by the graph namespace."""

    return compiler_context(require_native=True)


def _stride_signature(value: Any, *, dynamic: bool) -> tuple[Any, ...]:
    stride = getattr(value, "stride", ())
    strides = tuple(stride() if callable(stride) else stride)
    if not dynamic:
        return strides
    sizes = value.shape
    candidates = {}
    inferred = [None] * len(strides)
    # Express dense strides through dimensions so changing sizes can reuse
    # a specialization while changing storage order cannot.
    for stride, neg_index in sorted((s, -i) for i, s in enumerate(strides)):
        index = -neg_index
        contiguous = (
            index + 1 < len(sizes)
            and stride == sizes[index + 1] * strides[index + 1]
        )
        if stride in (0, 1) and not contiguous:
            expression = stride
        else:
            expression = candidates.get(stride, ("stride", index, stride))
        inferred[index] = expression
        candidates[sizes[index] * stride] = ("size", index, expression)
    return tuple(inferred)


def _tensor_signature(value: Any, *, dynamic: bool) -> tuple[Any, ...] | None:
    module_name = type(value).__module__
    if not module_name.startswith("tensorplay"):
        return None
    shape = getattr(value, "shape", None)
    if callable(shape):
        shape = shape()
    try:
        shape = tuple(int(item) for item in shape)
        # Dynamic mode keeps rank specialization but removes concrete sizes.
        # shape policy; operations still receive the real runtime tensors.
        shape_key = ("dynamic", len(shape)) if dynamic else shape
    except (TypeError, ValueError):
        shape_key = repr(shape)
    dtype = getattr(value, "dtype", None)
    if callable(dtype):
        dtype = dtype()
    device = getattr(value, "device", None)
    if callable(device):
        device = device()
    requires_grad = getattr(value, "requires_grad", None)
    if callable(requires_grad):
        requires_grad = requires_grad()
    return (
        "tensor",
        type(value),
        shape_key,
        repr(dtype),
        repr(device),
        bool(requires_grad),
        _stride_signature(value, dynamic=dynamic),
    )


def _strides_follow_shape(shape: Any, strides: Any) -> bool:
    """Whether ``strides`` is the dense layout of ``shape``."""

    if not isinstance(shape, tuple) or not isinstance(strides, tuple):
        return False
    if len(shape) != len(strides):
        return False
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if stride != expected:
            return False
        expected *= size
    return True


def _same_layout(expected: Any, actual: Any) -> bool:
    """Whether two tensor signatures describe the same memory layout.

    A dense tensor's strides are decided by its sizes, so a signature whose
    strides are dense keeps the same layout as another dense one even when
    the sizes differ.  Anything else keeps its strides as the layout fact.
    """

    stride_index = 6
    expected_shape, expected_stride = expected[2], expected[stride_index]
    actual_shape, actual_stride = actual[2], actual[stride_index]
    if _strides_follow_shape(expected_shape, expected_stride) and (
        _strides_follow_shape(actual_shape, actual_stride)
    ):
        return True
    return expected_stride == actual_stride


def _differs_only_in_tensor_shapes(expected: Any, actual: Any) -> bool:
    """Whether two input signatures differ only in concrete tensor sizes."""

    if isinstance(expected, tuple) and isinstance(actual, tuple):
        if (
            expected
            and isinstance(expected[0], str)
            and expected[0] == "tensor"
            and len(expected) == len(actual)
        ):
            for index, (left, right) in enumerate(zip(expected, actual)):
                if index == 2:  # the shape slot
                    continue
                if index == 6:  # the stride slot, decided by the sizes above
                    if _same_layout(expected, actual):
                        continue
                    return False
                if not _differs_only_in_tensor_shapes(left, right):
                    return False
            return True
        if len(expected) != len(actual):
            return False
        return all(
            _differs_only_in_tensor_shapes(left, right)
            for left, right in zip(expected, actual)
        )
    return expected == actual


def _shape_wobble(chain: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Whether a stored specialization fails only on a changed tensor size."""

    checker = getattr(chain, "fails_only_on_tensor_sizes", None)
    if checker is None:
        return False
    return bool(checker(args, kwargs))


def _value_signature(value: Any, *, dynamic: bool) -> Any:
    tensor_key = _tensor_signature(value, dynamic=dynamic)
    if tensor_key is not None:
        return tensor_key
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value), value)
    if isinstance(value, tuple):
        return (tuple, tuple(_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, list):
        return (list, tuple(_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, dict):
        items = sorted(
            (
                (
                    _value_signature(key, dynamic=dynamic),
                    _value_signature(item, dynamic=dynamic),
                )
                for key, item in value.items()
            ),
            key=repr,
        )
        return (dict, tuple(items))
    return (type(value), id(value))


def _input_signature(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, dynamic: bool
) -> Any:
    return (
        tuple(_value_signature(item, dynamic=dynamic) for item in args),
        tuple(
            sorted(
                (key, _value_signature(value, dynamic=dynamic))
                for key, value in kwargs.items()
            )
        ),
    )


_grad_enabled: Any = None


def _grad_state_component() -> tuple[Any, ...]:
    """The ambient autograd state at call time.

    A region captured with autograd enabled wires its outputs for backward;
    replaying it with the state flipped (or the reverse) would hand the
    caller an output under the wrong autograd contract, so the state is
    part of every specialization key and is read on every call.
    """
    global _grad_enabled
    if _grad_enabled is None:
        import tensorplay

        _grad_enabled = tensorplay.is_grad_enabled
    return (_grad_enabled(),)


def _quick_value_signature(value: Any, *, dynamic: bool) -> Any:
    """Build the hot-path guard key without repr-heavy metadata formatting."""

    if type(value).__module__.startswith("tensorplay"):
        shape = getattr(value, "shape", None)
        try:
            shape_key = ("dynamic", len(shape)) if dynamic else tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            shape_key = repr(shape)
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        device_type = getattr(device, "type", None)
        if device_type is None:
            device_type = repr(device)
        device_key = (
            device_type,
            getattr(device, "index", None),
        )
        requires_grad = getattr(value, "requires_grad", False)
        return (
            type(value), shape_key, dtype, device_key, bool(requires_grad),
            _stride_signature(value, dynamic=dynamic),
        )
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value), value)
    if isinstance(value, tuple):
        return (tuple, tuple(_quick_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, list):
        return (list, tuple(_quick_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, dict):
        return (
            dict,
            tuple(
                sorted(
                    (
                        key,
                        _quick_value_signature(item, dynamic=dynamic),
                    )
                    for key, item in value.items()
                )
            ),
        )
    return (type(value), id(value))


def _quick_input_signature(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, dynamic: bool
) -> Any:
    return (
        tuple(_quick_value_signature(item, dynamic=dynamic) for item in args),
        tuple(
            sorted(
                (key, _quick_value_signature(value, dynamic=dynamic))
                for key, value in kwargs.items()
            )
        ),
    )


def _arg_fingerprint(value: Any) -> Any:
    """Cheap per-call identity probe for the hot-path key memo.

    ``(id, version)`` for tensors: in-place mutation bumps ``_version`` so a
    cached key component is never reused across mutated inputs; fresh tensors
    have fresh ids.  Inference tensors carry no version counter and are
    immutable, so their identity alone keys the entry.  Other inputs without
    a version counter use tensor metadata.  Scalars compare by value.  This
    replaces per-call shape/dtype/device reads and tuple rebuilding, which
    profiling showed at ~40% of steady-state compiled-call time.
    """

    module = type(value).__module__
    if module.startswith("tensorplay"):
        try:
            version = value._version
        except RuntimeError:
            if getattr(value, "is_inference", lambda: False)():
                return ("t", id(value), None)
            version = ("metadata", _quick_value_signature(value, dynamic=False))
        return (
            "t",
            id(value),
            version,
            bool(getattr(value, "requires_grad", False)),
        )
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return ("v", type(value).__name__, value)
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_arg_fingerprint(item) for item in value))
    if isinstance(value, dict):
        return (
            dict,
            tuple(sorted(
                ((_arg_fingerprint(key), _arg_fingerprint(item)) for key, item in value.items()),
                key=repr,
            )),
        )
    return ("o", id(value))


def _call_fingerprint(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple:
    items = [_arg_fingerprint(item) for item in args]
    if kwargs:
        items.extend(
            (key, _arg_fingerprint(kwargs[key])) for key in sorted(kwargs)
        )
    return tuple(items)


def _region_label(model: Any, name: str | None) -> str:
    """How diagnostics name the region being compiled.

    A caller-supplied name is the region's identifier; without one the
    program's own name stands in for it.
    """

    program = getattr(model, "__name__", None)
    if name is None:
        return repr(program if program is not None else model)
    return f"{name!r} ({program})" if program is not None else repr(name)


def _backend_kwargs(
    *,
    mode: str | None,
    options: dict[str, Any] | None,
    dynamic: bool | None,
    strict_native: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if mode is not None and mode != "default":
        kwargs["mode"] = mode
    if options:
        kwargs["options"] = dict(options)
    if dynamic is not None:
        kwargs["dynamic"] = dynamic
    if strict_native:
        # Capture may run in Python, but the returned callable must be a
        # native backend executable when this contract is requested.
        kwargs["strict_native"] = True
    return kwargs


def _log_recompiles(config_verbose: bool) -> bool:
    import os

    return config_verbose or os.environ.get("TP_LOG_RECOMPILES", "") not in ("", "0")


def compile(
    model: Callable[..., Any] | None = None,
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str | CompilerFn | None = None,
    mode: str | None = None,
    options: dict[str, Any] | None = None,
    name: str | None = None,
    disable: bool = False,
    recompile_limit: int | None = None,
    strict_native: bool = False,
    dynamic_shapes: Any = None,
) -> Callable[..., Any]:
    """Compile a callable through the TensorPlay compiler frontend.

    ``backend`` may be a registered name or a callable with the contract
    ``backend(graph_module, example_inputs, **kwargs) -> callable``.  The
    frontend caches specializations by input metadata; capture and backend
    failures are surfaced as compiler errors.

    The shape policy is chosen by ``dynamic`` or ``dynamic_shapes``:

    * ``dynamic=None`` (the default) captures concrete sizes.  When a later
      call changes only a tensor size, the region is captured again with
      every extent that is not 0 or 1 symbolic, provided the backend declares
      ``supports_symbolic_shapes`` and
      :data:`tensorplay.compiler.config.automatic_dynamic_shapes` is true.
    * ``dynamic=True`` captures extents symbolically from the first call.
    * ``dynamic_shapes`` names the axes that may vary, per argument:
      ``{"x": {0: Dim("batch")}, "y": (Dim("batch"), None)}``.  Unnamed axes
      keep the example's size and are checked before a cached region runs.
      The specification needs a backend that declares
      ``supports_symbolic_shapes``.

    A tensor marked with :func:`mark_static` keeps the example's extent even
    when the policy would make it symbolic.

    ``name`` identifies the region in recompilation diagnostics and is kept on
    the returned callable as ``_tensorplay_region_name``; it does not change
    what the region computes, so it is never part of a cache key.
    """

    normalized_dynamic_shapes = dynamic_shapes
    shape_spec: dict[str, Any] | None = None
    if dynamic_shapes is not None:
        if dynamic is not None:
            raise RuntimeError("dynamic and dynamic_shapes cannot both be specified")
        if isinstance(dynamic_shapes, bool):
            dynamic = dynamic_shapes

    if mode is not None and options is not None:
        raise RuntimeError("Either mode or options can be specified, but not both")
    if mode is None and options is None:
        mode = "default"
    if options is not None and not isinstance(options, dict):
        raise TypeError(f"options must be a dict, got {type(options)!r}")

    from tensorplay.compiler import config as compiler_config

    configured_dynamic = compiler_config.dynamic_shapes
    if dynamic is None and normalized_dynamic_shapes is None:
        if configured_dynamic is not None:
            dynamic = configured_dynamic
        elif not compiler_config.assume_static_by_default:
            dynamic = True
    if not isinstance(compiler_config.assume_static_by_default, bool):
        raise TypeError("config.assume_static_by_default must be a bool")
    if not isinstance(compiler_config.verbose, bool):
        raise TypeError("config.verbose must be a bool")
    if not isinstance(compiler_config.fail_on_recompile_limit_hit, bool):
        raise TypeError("config.fail_on_recompile_limit_hit must be a bool")
    if not isinstance(compiler_config.force_disable_caches, bool):
        raise TypeError("config.force_disable_caches must be a bool")
    if not isinstance(compiler_config.automatic_dynamic_shapes, bool):
        raise TypeError("config.automatic_dynamic_shapes must be a bool")
    accumulated_limit = compiler_config.accumulated_recompile_limit
    if (
        not isinstance(accumulated_limit, int)
        or isinstance(accumulated_limit, bool)
        or accumulated_limit < 1
    ):
        raise ValueError("config.accumulated_recompile_limit must be a positive integer")

    if model is None:
        # A per-input specification is resolved against the program's
        # arguments, which are unknown until the decorator is applied.
        decorated_shapes = (
            None if isinstance(dynamic_shapes, bool) else dynamic_shapes
        )
        return lambda actual_model: compile(
            actual_model,
            fullgraph=fullgraph,
            backend=backend,
            dynamic=dynamic if decorated_shapes is None else None,
            mode=mode,
            options=options,
            name=name,
            disable=disable,
            recompile_limit=recompile_limit,
            strict_native=strict_native,
            dynamic_shapes=decorated_shapes,
        )
    if not callable(model):
        raise TypeError(f"compile() expected a callable, got {type(model)!r}")
    if disable:
        return model
    if recompile_limit is not None and recompile_limit < 1:
        raise ValueError("recompile_limit must be positive")
    if name is not None and not isinstance(name, str):
        raise TypeError(f"name must be a string or None, got {type(name)!r}")

    if dynamic_shapes is not None and not isinstance(dynamic_shapes, bool):
        from .shape_policy import normalize_dynamic_shapes, parameter_names

        shape_spec = normalize_dynamic_shapes(dynamic_shapes, parameter_names(model))
        normalized_dynamic_shapes = shape_spec
        dynamic = True

    lifted = _lift_free_tensors(model)
    if lifted is not None:
        program, read_free = lifted
        return _WithFreeTensors(
            model,
            compile(
                program,
                fullgraph=fullgraph,
                backend=backend,
                dynamic=None if shape_spec is not None else dynamic,
                mode=mode,
                options=options,
                name=name,
                recompile_limit=recompile_limit,
                strict_native=strict_native,
                dynamic_shapes=shape_spec,
            ),
            read_free,
        )

    backend_spec = get_default_backend() if backend is None else backend
    compiler_fn = lookup_backend(backend_spec)
    if shape_spec is not None and not getattr(
        compiler_fn, "supports_symbolic_shapes", False
    ):
        raise RuntimeError(
            f"backend {backend_spec!r} cannot capture symbolic shapes; "
            "dynamic_shapes needs a backend that declares "
            "supports_symbolic_shapes"
        )
    automatic_dynamic = (
        dynamic is None
        and normalized_dynamic_shapes is None
        and compiler_config.automatic_dynamic_shapes
        and getattr(compiler_fn, "supports_symbolic_shapes", False)
    )
    dynamic_generalized = False
    backend_kwargs = _backend_kwargs(
        mode=mode,
        options=options,
        dynamic=dynamic,
        strict_native=strict_native,
    )
    specialization_dynamic = dynamic is True
    specialization_limit = (
        _DEFAULT_RECOMPILE_LIMIT if recompile_limit is None else recompile_limit
    )
    if recompile_limit is None:
        specialization_limit = compiler_config.recompile_limit
    if (
        not isinstance(specialization_limit, int)
        or isinstance(specialization_limit, bool)
        or specialization_limit < 1
    ):
        raise ValueError("recompile_limit must be a positive integer")
    cache_enabled = not compiler_config.force_disable_caches
    compile_attempts = 0
    cache: dict[Any, Callable[..., Any]] = {}
    backend_variants: dict[Any, list[Callable[..., Any]]] = {}
    guard_chains: dict[Any, GuardChain] = {}
    lock = threading.RLock()
    last_quick_key: Any = object()
    last_compiled_fn: Callable[..., Any] | None = None
    guard_param_names: tuple[str, ...] = ()
    gate_evaluator: Callable[..., tuple] | None = None
    target_cache = model.forward if _is_module_like(model) else model
    try:
        target_signature: Any = inspect.signature(target_cache)
    except (TypeError, ValueError):
        target_signature = None
    last_call_fp: Any = None
    last_quick_parts: tuple[Any, ...] | None = None
    last_module_state: Any = None
    #: The entry the native certificate was last installed on, the places it
    #: names, and the ambient mode it was taken under.  Held so the install
    #: (a pass over every site) only happens when one of them moved.
    bound_certificate: tuple[Any, ...] | None = None

    def _guard_component(
        args_: tuple[Any, ...],
        kwargs_: dict[str, Any],
        builder: Callable[..., Any],
    ) -> tuple[Any, ...]:
        if not guard_param_names:
            return ()
        if target_signature is None:
            return ("shape-guards", "unbound")
        try:
            bound = target_signature.bind_partial(*args_, **kwargs_)
            bound.apply_defaults()
        except (TypeError, ValueError):
            return ("shape-guards", "unbound")
        return (
            "shape-guards",
            tuple(
                builder(bound.arguments.get(name), dynamic=False)
                for name in guard_param_names
            ),
        )

    def _bind_dispatcher_fast(fast: Any, nargs: int, bound_module_state: Any) -> None:
        """Hand a resolved steady-state entry to the outer C dispatcher.

        The dispatcher re-enters it directly once its argument memo passes;
        any slow call routes through here again, so the binding stays
        current across recompiles and specialization switches.

        The argument memo cannot see ambient autograd or module state, so
        both are checked before the entry runs.  Where the entry is a native
        trampoline, the places the capture read from are handed over as a
        certificate it re-checks in one C pass; otherwise the entry is a
        Python gate that reads the state the long way.  Either way a state
        change returns to specialization lookup rather than reusing an
        artifact captured under a different mode or module hierarchy.
        """

        nonlocal bound_certificate
        bound_grad_state = _grad_state_component()
        # A Python entry has no trampoline state of its own, so the plan has
        # nothing to attach to and the gate below stays in charge.
        if not hasattr(fast, "__code__"):
            plan = _state_certificate_plan(model)
            if plan is None:
                # The entry reads state the argument fingerprint cannot see,
                # and no certificate describes those places.  Installing it
                # anyway would replay a recording whose state may have moved,
                # so the caller keeps the path that reads that state itself.
                dispatcher.tpx_set_fast(None, nargs)
                bound_certificate = None
                return
            # The certificate attaches to the bound entry, so the entry is
            # bound first; a slow call re-enters here and rebinds.
            dispatcher.tpx_set_fast(fast, nargs)
            grad_enabled = bound_grad_state[0]
            if (
                bound_certificate is None
                or bound_certificate[0] is not fast
                or bound_certificate[1] is not plan
                or bound_certificate[2] != grad_enabled
            ):
                # Installing the certificate is a pass over every site it
                # names, so it is only redone when the entry, the places or
                # the mode it certifies are no longer the bound ones.
                dispatcher.tpx_set_state_plan(plan, grad_enabled)
                bound_certificate = (fast, plan, grad_enabled)
            return

        dispatcher.tpx_set_fast(fast, nargs)

        def _fast(*args: Any) -> Any:
            if (
                _grad_state_component() == bound_grad_state
                and _module_state_component(model) == bound_module_state
            ):
                return fast(*args)
            return optimized(*args)

        dispatcher.tpx_set_fast(_fast, nargs)

    @functools.wraps(model)
    def optimized(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_quick_key, last_compiled_fn
        nonlocal guard_param_names, gate_evaluator
        nonlocal last_call_fp, last_quick_parts, last_module_state
        nonlocal compile_attempts, specialization_dynamic, dynamic_generalized
        if _capture_disabled.get():
            return model(*args, **kwargs)
        module_state = _module_state_component(model)
        if module_state != last_module_state:
            # Input identity does not certify gates that read module state.
            last_call_fp = None
            last_quick_parts = None
            last_module_state = module_state
        # Steady-state memo: identical objects with unchanged versions (or
        # unchanged scalars) cannot produce different signatures or gate
        # outcomes -- skip metadata reads and evaluator replay entirely.
        # Tensor-positional calls ask the C dispatcher for a read-only
        # certificate against the trampoline fingerprint (one probe loop in
        # C replaces the Python (id, version) walk); every other call shape
        # falls back to the Python fingerprint below.
        if dispatcher is not None:
            matched = not kwargs and dispatcher.tpx_fingerprint_matches(*args)
            if not matched:
                # Refresh the certificate for this call up front so it stays
                # paired with the quick memo built below even when resolution
                # raises.  Calls the fingerprint cannot describe (scalars,
                # kwargs) clear it instead of leaving a stale certificate.
                if not kwargs:
                    dispatcher.tpx_fingerprint_store(*args)
                else:
                    dispatcher.tpx_fingerprint_clear()
        else:
            matched = False
        if last_quick_parts is not None and matched:
            input_signature, shape_component, data_component = last_quick_parts
        else:
            call_fp = _call_fingerprint(args, kwargs)
            if last_quick_parts is not None and call_fp == last_call_fp:
                input_signature, shape_component, data_component = last_quick_parts
            else:
                input_signature = _quick_input_signature(
                    args, kwargs, dynamic=specialization_dynamic
                )
                shape_component = _guard_component(
                    args, kwargs, _quick_value_signature
                )
                data_component = gate_evaluator(args, kwargs) if gate_evaluator else ()
            last_quick_parts = (
                input_signature,
                shape_component,
                data_component,
            )
            last_call_fp = call_fp
        grad_state = _grad_state_component()
        quick_key = (
            input_signature, shape_component, data_component, grad_state, module_state
        )
        with lock:
            if cache_enabled and cache and last_compiled_fn is not None and quick_key == last_quick_key:
                compiled_fn = last_compiled_fn
            else:
                key = (
                    _input_signature(args, kwargs, dynamic=specialization_dynamic),
                    _guard_component(args, kwargs, _value_signature),
                    data_component,
                    grad_state,
                    module_state,
                )
                compiled_fn = cache.get(key) if cache_enabled else None
            if compiled_fn is not None:
                check = getattr(compiled_fn, "_tensorplay_guard", None)
                if check is not None and not check(*args, **kwargs):
                    key = (
                        _input_signature(args, kwargs, dynamic=specialization_dynamic),
                        _guard_component(args, kwargs, _value_signature),
                        data_component,
                        grad_state,
                        module_state,
                    )
                    compiled_fn = next((
                        variant for variant in backend_variants.get(key, ())
                        if variant is not compiled_fn and getattr(variant, "_tensorplay_guard")(*args, **kwargs)
                    ), None)
            if not cache:
                backend_variants.clear()
            store_compiled = cache_enabled
            if compiled_fn is None:
                variant_count = len(cache) + sum(
                    max(0, len(variants) - 1) for variants in backend_variants.values()
                )
                if variant_count >= specialization_limit:
                    if fullgraph or compiler_config.fail_on_recompile_limit_hit:
                        raise RuntimeError(
                            "TensorPlay compile specialization limit reached in "
                            f"{_region_label(model, name)}"
                        )
                    store_compiled = False
                # Explain the miss against every stored specialization before
                # recompiling, retaining each guard mismatch.
                reasons: list[Any] = []
                for chain in guard_chains.values():
                    reasons.extend(chain.explain(args, kwargs))
                if reasons:
                    optimized._tensorplay_last_recompile_reasons = tuple(reasons)
                    if _log_recompiles(compiler_config.verbose):
                        import warnings

                        warnings.warn(
                            "recompiling "
                            f"{_region_label(model, name)}: "
                            + format_recompile_reasons(reasons),
                            stacklevel=2,
                        )
                if (
                    automatic_dynamic
                    and not dynamic_generalized
                    and any(
                        _shape_wobble(chain, args, kwargs)
                        for chain in guard_chains.values()
                    )
                ):
                    # The concrete policy just failed because a size moved.
                    # From here the region is captured with every extent that
                    # is not 0 or 1 symbolic, which serves the sizes already
                    # seen and the ones that would have followed each of
                    # them as separate specializations.
                    dynamic_generalized = True
                    specialization_dynamic = True
                    backend_kwargs["dynamic"] = True
                    optimized._tensorplay_dynamic = True  # type: ignore[attr-defined]
                    if cache_enabled:
                        cache.clear()
                        guard_chains.clear()
                        backend_variants.clear()
                    last_compiled_fn = None
                    last_call_fp = None
                    last_quick_parts = None
                    input_signature = _quick_input_signature(
                        args, kwargs, dynamic=specialization_dynamic
                    )
                    quick_key = (
                        input_signature,
                        shape_component,
                        data_component,
                        grad_state,
                        module_state,
                    )
                    key = (
                        _input_signature(args, kwargs, dynamic=specialization_dynamic),
                        _guard_component(args, kwargs, _value_signature),
                        data_component,
                        grad_state,
                        module_state,
                    )
                if compile_attempts >= accumulated_limit:
                    raise RuntimeError(
                        "TensorPlay accumulated recompilation limit reached in "
                        f"{_region_label(model, name)}"
                    )
                compile_attempts += 1
                compiled_fn, captured_gm = _compile_region(
                    model,
                    compiler_fn,
                    args,
                    kwargs,
                    backend_kwargs=backend_kwargs,
                    shape_spec=shape_spec,
                    region_key=make_region_key(
                        target_cache,
                        model if _is_module_like(model) else None,
                        key if shape_spec is None else (key, shape_policy_signature(shape_spec)),
                    ),
                )
                # Keys gain a guard component once capture reveals metadata
                # reads or control-flow gates; invalidate so entries are stored
                # uniformly.
                promoted = _extract_shape_guard_params(captured_gm)
                replay = captured_gm.meta.get("guard_replay")
                if promoted - set(guard_param_names) or (
                    replay is not None and gate_evaluator is None
                ):
                    guard_param_names = tuple(sorted({*guard_param_names, *promoted}))
                    if replay is not None:
                        gate_target = model.forward if _is_module_like(model) else model
                        gate_evaluator = _make_gate_evaluator(replay, gate_target)
                    if cache_enabled:
                        cache.clear()
                        guard_chains.clear()
                    last_compiled_fn = None
                    last_call_fp = None
                    last_quick_parts = None
                    if gate_evaluator is not None:
                        data_component = gate_evaluator(args, kwargs)
                key = (
                    _input_signature(args, kwargs, dynamic=specialization_dynamic),
                    _guard_component(args, kwargs, _value_signature),
                    data_component,
                    grad_state,
                    module_state,
                )
                if store_compiled:
                    cache[key] = compiled_fn
                    if getattr(compiled_fn, "_tensorplay_guard", None) is not None:
                        backend_variants.setdefault(key, []).append(compiled_fn)
                    guard_chains[key] = build_guard_chain(
                        key,
                        args=args,
                        kwargs=kwargs,
                        dynamic=specialization_dynamic,
                        target=model.forward if _is_module_like(model) else model,
                        gate_evaluator=gate_evaluator,
                        module_evaluator=lambda: _module_state_component(model),
                    )
            last_quick_key = quick_key
            last_compiled_fn = compiled_fn
        if not kwargs:
            make_entry = getattr(compiled_fn, "_steady_state_entry", None)
            fast = make_entry(len(args)) if make_entry is not None else None
            if fast is None:
                fast = getattr(compiled_fn, "_fast_call", None)
            if fast is not None:
                _bind_dispatcher_fast(fast, len(args), module_state)
                return compiled_fn(*args)
        return compiled_fn(*args, **kwargs)

    from tensorplay._C import _stax as _stax_native

    dispatcher: Any = _stax_native.make_call_dispatcher(optimized)

    optimized._tensorplay_backend = backend_spec  # type: ignore[attr-defined]
    optimized._tensorplay_cache = cache  # type: ignore[attr-defined]
    optimized._tensorplay_guard_chains = guard_chains  # type: ignore[attr-defined]
    optimized._tensorplay_last_recompile_reasons = ()  # type: ignore[attr-defined]
    optimized._tensorplay_original = model  # type: ignore[attr-defined]
    optimized._tensorplay_dynamic = (  # type: ignore[attr-defined]
        True if dynamic_generalized else dynamic
    )
    optimized._tensorplay_dynamic_shapes = normalized_dynamic_shapes  # type: ignore[attr-defined]
    optimized._tensorplay_region_name = name  # type: ignore[attr-defined]
    optimized._tensorplay_recompile_limit = specialization_limit  # type: ignore[attr-defined]
    _compiled_wrappers.add(optimized)
    if dispatcher is not None:
        dispatcher._tensorplay_dispatcher = optimized  # type: ignore[attr-defined]
        return dispatcher
    return optimized


@functools.lru_cache(maxsize=1024)
def _loaded_global_names_of_code(code: types.CodeType) -> frozenset[str]:
    """The names ``code``, and code defined inside it, loads as globals.

    This is a fixed property of the code itself -- which names it loads does
    not change when what they resolve to does -- so the walk over it is done
    once per code object rather than on every call that reads the state a
    function can reach through its globals.
    """

    loaded: set[str] = set()
    pending = [code]
    while pending:
        current = pending.pop()
        for instruction in dis.get_instructions(current):
            if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
                loaded.add(instruction.argval)
        pending.extend(
            c for c in current.co_consts if isinstance(c, types.CodeType)
        )
    return frozenset(loaded)


def _loaded_global_names(function: types.FunctionType) -> set[str]:
    """The names ``function``'s code, and code defined inside it, loads as globals."""

    return set(_loaded_global_names_of_code(function.__code__))


def _state_roots(program: Any) -> tuple[list[Any], list[types.FunctionType]]:
    """The modules and Python functions whose state a call can read.

    Walks closure cells, the globals a function loads, the children of the
    modules it reaches and the methods defined on their classes, so a value
    reached through a helper is found too.  Every function and module is
    visited once, which also stops a reference cycle from looping.
    """

    import tensorplay

    module_type = tensorplay.nn.Module
    modules: list[Any] = []
    functions: list[types.FunctionType] = []
    seen_modules: set[int] = set()
    seen_functions: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, module_type):
            if id(value) in seen_modules:
                return
            seen_modules.add(id(value))
            modules.append(value)
            for function in _module_running_functions(value):
                visit(function)
            for child in getattr(value, "_modules", {}).values():
                if child is not None:
                    visit(child)
            return
        if isinstance(value, (staticmethod, classmethod)):
            visit(value.__func__)
            return
        if isinstance(value, types.MethodType):
            visit(value.__self__)
            visit(value.__func__)
            return
        if not callable(value):
            return
        if isinstance(value, types.FunctionType) and "__wrapped__" not in value.__dict__:
            function = value
        else:
            try:
                function = inspect.unwrap(value)
            except (TypeError, ValueError):
                return
        if not isinstance(function, types.FunctionType):
            return
        if id(function) in seen_functions:
            return
        seen_functions.add(id(function))
        functions.append(function)
        for cell in function.__closure__ or ():
            try:
                visit(cell.cell_contents)
            except ValueError:
                continue
        scope = function.__globals__
        for name in _loaded_global_names(function):
            visit(scope.get(name))

    visit(program)
    return modules, functions


def _modules_reached(program: Any) -> list[Any]:
    """The modules whose state running ``program`` can update.

    The program itself or the module a bound method belongs to; for a plain
    function, the modules it reaches through its closure or the globals it
    loads -- a lambda calling a model it closes over updates that model.  A
    helper a reached function calls is followed as well.
    """

    return _state_roots(program)[0]


_MISSING = object()

#: Types whose values are described by themselves: an object of one of them
#: that is still the object a site held last time cannot describe a change, so
#: the steady-state scan does not need the general reuse test for it.
_IMMUTABLE_SITE_TYPES = frozenset({type(None), bool, int, float, str})

#: Frozen descriptions of immutable values, keyed by identity.  A value that
#: cannot change is described once; the entry keeps the value alive, so the
#: identity cannot be handed to a different object while the entry is held.
_constant_entry_cache: dict[int, tuple[Any, Any]] = {}
_MAX_CONSTANT_ENTRY_CACHE = 4096


def _is_immutable_constant(value: Any) -> bool:
    """Whether a value can be a literal in captured code and cannot change.

    Tensors, modules and callables are handled elsewhere.  Numbers, text and
    containers of them are what a capture may have fixed while tracing, and
    an immutable one can be compared by value on every call.
    """

    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_is_immutable_constant(item) for item in value)
    return False


def _freeze_immutable_constant(value: Any) -> Any:
    """A hashable, exact description of an immutable baked constant.

    Floats are described by their hexadecimal spelling so that a sign change
    on zero and a ``nan`` still compare as themselves; containers keep their
    type, because a list and a tuple of the same items need not produce the
    same trace.
    """

    kind = type(value).__name__
    if isinstance(value, float):
        return (kind, value.hex())
    if isinstance(value, complex):
        return (kind, value.real.hex(), value.imag.hex())
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return (kind, value)
    if isinstance(value, tuple):
        return (kind, tuple(_freeze_immutable_constant(item) for item in value))
    if isinstance(value, frozenset):
        return (
            kind,
            tuple(sorted(
                (_freeze_immutable_constant(item) for item in value), key=repr
            )),
        )
    return (kind, id(value))


def _constant_entry(value: Any) -> Any:
    """A hashable description of a value a capture may have baked, or None.

    Immutable values are compared by content.  A mutable container is
    described by a snapshot of its contents: an in-place edit (`list.append`,
    a dict update, an element write) changes the snapshot, so the next call
    builds a new specialization instead of reusing one that baked the old
    contents.  A callable is described by the object itself: a rebinding to
    another function changes which operations the captured program would have
    recorded.  A container the captured call fills in makes the key change on
    every call; that is the same recompile pressure the guard-based approach
    accepts, bounded by the recompile limit.
    """

    if _is_immutable_constant(value):
        cached = _constant_entry_cache.get(id(value))
        if cached is not None and cached[0] is value:
            return cached[1]
        frozen = _freeze_immutable_constant(value)
        if len(_constant_entry_cache) >= _MAX_CONSTANT_ENTRY_CACHE:
            _constant_entry_cache.clear()
        _constant_entry_cache[id(value)] = (value, frozen)
        return frozen
    if isinstance(value, (list, tuple)):
        return (
            "contents",
            type(value).__name__,
            tuple(_constant_entry(item) for item in value),
        )
    if isinstance(value, dict):
        return (
            "contents",
            "dict",
            tuple(sorted(
                ((_constant_entry(key), _constant_entry(item))
                 for key, item in value.items()),
                key=repr,
            )),
        )
    if isinstance(value, set):
        return (
            "contents",
            "set",
            tuple(sorted(
                (_constant_entry(item) for item in value), key=repr
            )),
        )
    if callable(value):
        return ("reference", id(value))
    return None


def _reusable_by_identity(value: Any) -> bool:
    """Whether the same object at a site means the same described value."""

    if isinstance(value, (list, dict, set)):
        return False
    if isinstance(value, tuple):
        return _is_immutable_constant(value)
    return True


def _dependency_names(
    code: types.CodeType,
    opnames: tuple[str, ...],
) -> frozenset[str]:
    """Names one kind of instruction mentions in ``code`` and code inside it."""

    names: set[str] = set()
    pending = [code]
    while pending:
        current = pending.pop()
        for instruction in dis.get_instructions(current):
            if instruction.opname in opnames:
                names.add(instruction.argval)
        pending.extend(
            c for c in current.co_consts if isinstance(c, types.CodeType)
        )
    return frozenset(names)


@functools.lru_cache(maxsize=1024)
def _assigned_names_of_code(code: types.CodeType) -> frozenset[str]:
    """Global and closure-cell names ``code`` writes."""

    return _dependency_names(
        code,
        ("STORE_GLOBAL", "STORE_NAME", "DELETE_GLOBAL", "DELETE_NAME"),
    ) | _dependency_names(code, ("STORE_DEREF", "DELETE_DEREF"))


@functools.lru_cache(maxsize=1024)
def _assigned_attribute_names_of_code(code: types.CodeType) -> frozenset[str]:
    """Attribute names ``code`` and code inside it write."""

    return _dependency_names(code, ("STORE_ATTR", "DELETE_ATTR"))


@functools.lru_cache(maxsize=1024)
def _loaded_attribute_names_of_code(code: types.CodeType) -> frozenset[str]:
    """The attribute names ``code``, and code defined inside it, reads."""

    return _dependency_names(code, ("LOAD_ATTR", "LOAD_METHOD"))


_IN_PLACE_METHODS = frozenset({
    "append", "extend", "insert", "pop", "remove", "clear", "reverse", "sort",
    "update", "add", "discard", "setdefault", "popitem", "increment_",
})


@functools.lru_cache(maxsize=1024)
def _in_place_mutated_names_of_code(code: types.CodeType) -> tuple[
    frozenset[str], frozenset[str], frozenset[str],
]:
    """Names whose object ``code`` (and code inside it) mutates in place.

    Returns (globals, closure-cell names, attributes).  A mutating method
    call attributes itself to the nearest name or attribute load that
    precedes it: ``_CALLS.append(1)`` reads the global ``_CALLS`` and calls
    ``append`` on it, so ``_CALLS`` is reported as mutated.  The contents of
    a container the captured call itself fills in are part of the call's
    effect, not of its key; such names are left out of the key entirely.
    """

    globals_: set[str] = set()
    cells: set[str] = set()
    attrs: set[str] = set()

    def walk(current: types.CodeType) -> None:
        window: list[tuple[str, Any]] = []
        for instruction in dis.get_instructions(current):
            # 3.12+ has no LOAD_METHOD; a method load is a LOAD_ATTR whose
            # arg may carry a "method" flag bit, so one opcode covers both.
            is_method_load = (
                instruction.opname == "LOAD_METHOD"
                or (
                    instruction.opname == "LOAD_ATTR"
                    and instruction.argval in _IN_PLACE_METHODS
                )
            )
            if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF", "LOAD_ATTR"):
                if is_method_load and instruction.argval in _IN_PLACE_METHODS:
                    for opname, argval in reversed(window):
                        if opname == "LOAD_GLOBAL" or opname == "LOAD_NAME":
                            globals_.add(argval)
                            break
                        if opname == "LOAD_DEREF":
                            cells.add(argval)
                            break
                        if opname == "LOAD_ATTR":
                            attrs.add(argval)
                            break
                    window.clear()
                    continue
                window.append((instruction.opname, instruction.argval))
                if len(window) > 8:
                    window.pop(0)
        for const in current.co_consts:
            if isinstance(const, types.CodeType):
                walk(const)

    walk(code)
    return frozenset(globals_), frozenset(cells), frozenset(attrs)


def _module_running_functions(module: Any) -> list[types.FunctionType]:
    """The functions whose code runs when ``module`` is called.

    ``forward`` and the methods it reaches from the module's own class are
    followed, so both values read directly and values read by a helper are
    part of what a call depends on.  Code that is not reached this way --
    ``__init__`` or an unused method -- is not.

    Which functions run is a property of the class, not of the instance, so
    the answer is worked out once per class.  An instance that rebinds its own
    ``forward`` is the one case where the class alone does not say what runs,
    and that call is answered directly rather than through the cache.  A class
    method rebound in place after a region was compiled is not watched: the
    class identity, not the contents of its namespace, is what a captured
    region is keyed on.
    """

    instance_forward = getattr(module, "__dict__", {}).get("forward")
    if instance_forward is not None:
        return _running_functions(type(module), instance_forward)
    return list(_running_functions_of_type(type(module)))


@functools.lru_cache(maxsize=1024)
def _running_functions_of_type(module_type: type) -> tuple[types.FunctionType, ...]:
    """The functions a call on an instance of ``module_type`` runs."""

    return tuple(_running_functions(module_type, None))


def _running_functions(
    module_type: type, forward_override: Any
) -> list[types.FunctionType]:
    helpers: dict[str, types.FunctionType] = {}
    for name, member in module_type.__dict__.items():
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        try:
            member = inspect.unwrap(member)
        except (TypeError, ValueError):
            continue
        if isinstance(member, types.FunctionType):
            helpers[name] = member
    forward = forward_override
    if forward is None:
        forward = getattr(module_type, "forward", None)
    if isinstance(forward, types.MethodType):
        forward = forward.__func__
    try:
        forward = inspect.unwrap(forward)
    except (TypeError, ValueError):
        forward = None
    running: list[types.FunctionType] = []
    seen: set[int] = set()
    pending = [forward] if isinstance(forward, types.FunctionType) else []
    while pending:
        function = pending.pop()
        if id(function) in seen:
            continue
        seen.add(id(function))
        running.append(function)
        for name in _loaded_attribute_names_of_code(function.__code__):
            helper = helpers.get(name)
            if helper is not None:
                pending.append(helper)
    return running


@functools.lru_cache(maxsize=1024)
def _attribute_names_of_type(module_type: type, assigned: bool) -> frozenset[str]:
    """Attribute names running code of ``module_type`` reads (or writes)."""

    collector = (
        _assigned_attribute_names_of_code
        if assigned
        else _loaded_attribute_names_of_code
    )
    names: set[str] = set()
    for function in _running_functions_of_type(module_type):
        names.update(collector(function.__code__))
    return frozenset(names)


@functools.lru_cache(maxsize=1024)
def _constant_reads_of_function(
    function: types.FunctionType,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """The globals and closure cells a capture may have baked from a function.

    Which names are read, assigned in place, or written is a property of the
    code, so the list is worked out once per function and only the values are
    read on each call.
    """

    code = function.__code__
    written = _assigned_names_of_code(code)
    mutated_globals, mutated_cells, _ = _in_place_mutated_names_of_code(code)
    globals_ = tuple(
        sorted(_loaded_global_names(function) - written - mutated_globals)
    )
    freevars = code.co_freevars
    cells = tuple(
        index
        for index in range(len(function.__closure__ or ()))
        if not (
            index < len(freevars)
            and (freevars[index] in written or freevars[index] in mutated_cells)
        )
    )
    return globals_, cells


@functools.lru_cache(maxsize=1024)
def _constant_attribute_names_of_type(module_type: type) -> tuple[str, ...]:
    """Attribute names of ``module_type`` whose values a capture may bake."""

    readable = _attribute_names_of_type(module_type, False)
    assigned = _attribute_names_of_type(module_type, True)
    mutated: set[str] = set()
    for function in _running_functions_of_type(module_type):
        _, _, mutated_attrs = _in_place_mutated_names_of_code(function.__code__)
        mutated |= mutated_attrs
    return tuple(sorted(readable - assigned - mutated))


def _constant_attribute_names(
    module_type: type, forward: Any
) -> tuple[str, ...]:
    """Attribute names to read when an instance rebinds its own ``forward``."""

    readable = _attribute_names_of_type(module_type, False)
    assigned = _attribute_names_of_type(module_type, True)
    mutated: set[str] = set()
    for function in _running_functions(module_type, forward):
        _, _, mutated_attrs = _in_place_mutated_names_of_code(function.__code__)
        mutated |= mutated_attrs
    return tuple(sorted(readable - assigned - mutated))


def _module_state_part(modules: list[Any]) -> tuple[Any, ...]:
    """Module identities, training modes and child bindings, in tree order."""

    return tuple(
        (
            index,
            id(module),
            module.training,
            tuple(
                (child_name, id(child))
                for child_name, child in module._modules.items()
            ),
        )
        for index, module in enumerate(modules)
    )


def _module_state_part_unchanged(
    modules: list[Any], recorded: tuple[Any, ...]
) -> bool:
    """Whether the tree still reads as it was recorded, without naming it again.

    Naming the tree builds a tuple per module and one per child binding, and
    this question is asked on every call where nothing has moved.  Walking the
    live tree against what was recorded answers it without building anything.
    """

    if len(modules) != len(recorded):
        return False
    for module, entry in zip(modules, recorded):
        _index, module_id, training, children = entry
        if module_id != id(module) or training is not module.training:
            return False
        live = module._modules
        if len(live) != len(children):
            return False
        for position, (child_name, child) in enumerate(live.items()):
            recorded_name, child_id = children[position]
            if child_id != id(child) or recorded_name != child_name:
                return False
    return True


def _module_state_component(program: Any) -> tuple[Any, ...]:
    """Read live module identities, training modes and child bindings.

    Closure cells and globals can be rebound, and children can be replaced
    without changing the root's training flag, so the places to read are
    revalidated on every call.  Discovering them -- walking the module tree
    and the functions it runs -- is the expensive half, and it only changes
    when a binding it walked does: the recorded modules are re-read first and
    the tree is walked again only when those readings moved.  Values read from
    outside the arguments are compared too, since a capture bakes them into the
    program it produces.

    The set of places to read is fixed once the part of the program's world
    that the call reaches has the same shape as last time, so the per-call
    work is reading those places rather than working out which they are.
    """

    roots = _cached_state_roots(program)
    if roots is not None:
        modules, functions, shape, recorded_part, recorded_component, pair = roots
        if _module_state_part_unchanged(modules, recorded_part):
            plan = _state_plan(program, shape, modules, functions)
            component = _planned_constant_component(plan)
            if component is recorded_component:
                return pair
            # A value a capture baked moved, and with it the module tree the
            # walk reached (a rebound closure cell or global is how a program
            # is pointed at another tree): discover the roots again.

    modules, functions = _state_roots(program)
    module_part = _module_state_part(modules)
    shape = (tuple(id(module) for module in modules), tuple(id(f) for f in functions))
    plan = _state_plan(program, shape, modules, functions)
    component = _planned_constant_component(plan)
    pair = (module_part, component)
    _store_state_roots(program, modules, functions, shape, module_part, component, pair)
    return pair


#: The modules and functions a program's call reaches, with the readings that
#: certify the walk is still current.  A dropped program drops its entry.
_state_roots_cache: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()


def _cached_state_roots(program: Any) -> Any:
    try:
        return _state_roots_cache.get(program)
    except TypeError:  # an unhashable program keeps no cache
        return None


def _store_state_roots(
    program: Any,
    modules: list[Any],
    functions: list[Any],
    shape: Any,
    module_part: tuple[Any, ...],
    component: tuple[Any, ...],
    pair: tuple[Any, ...],
) -> None:
    try:
        _state_roots_cache[program] = (
            modules, functions, shape, module_part, component, pair
        )
    except TypeError:
        pass


#: Read plans and the last value seen at each site, per compiled program.  A
#: weak key keeps a program that is dropped from holding its plan alive.
_state_plan_cache: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()

#: The same places, arranged for the native certificate, and the component they
#: were taken from.  A component is rebuilt whenever a recorded value moves, so
#: its identity is what says the arrangement is still describing the state the
#: capture saw.
_state_certificate_cache: "weakref.WeakKeyDictionary[Any, Any]" = (
    weakref.WeakKeyDictionary()
)


def _state_certificate_plan(program: Any) -> Any:
    """The captured state a call depends on, as the native check reads it.

    A site is described by where it lives -- a module attribute, a global, a
    closure cell -- and by the object it held when the capture ran.  The
    native side re-reads those places in one pass and answers whether every
    answer is still the same object with the same version, which is the
    condition under which the recording still describes the call.

    Answered with None when the program's state cannot be certified by
    identity alone (a site holding a value whose description moves without the
    object moving), so the caller keeps reading it the long way.
    """

    roots = _cached_state_roots(program)
    if roots is None:
        return None
    modules, functions, shape, _module_part, component, _pair = roots
    try:
        cached = _state_certificate_cache.get(program)
    except TypeError:
        cached = None
    if cached is not None and cached[0] is component:
        return cached[1]
    plan = _build_state_certificate(program, shape, modules, functions)
    try:
        _state_certificate_cache[program] = (component, plan)
    except TypeError:
        pass
    return plan


def _build_state_certificate(
    program: Any, shape: Any, modules: list[Any], functions: list[Any]
) -> Any:
    """Arrange a program's read sites as the native certificate's plan."""

    plan = _state_plan(program, shape, modules, functions)
    sites = plan[:-1]
    values, _cached_entries, last, reusable_flags = plan[-1][4]
    if last is None or len(values) < len(sites):
        return None
    entries: list[tuple[Any, ...]] = []
    for index, site in enumerate(sites):
        kind, owner, _label, name, _ = site
        value = values[index]
        present = value is not _MISSING
        if present and not reusable_flags[index]:
            # An unchanged object does not always mean an unchanged value for
            # this kind of site, and a certificate that cannot say the value
            # held still would be certifying the wrong thing.
            return None
        if kind == "attribute":
            code = 0
        elif kind == "global":
            code = 1
        elif kind == "closure":
            code = 2
        else:  # pragma: no cover - the plan only builds the three kinds
            return None
        entries.append((code, owner, name, value if present else None, present))
    module_part = [
        (
            module,
            bool(getattr(module, "training", True)),
            tuple(getattr(module, "_modules", {}).items()),
        )
        for module in modules
    ]
    return (tuple(module_part), tuple(entries))


def _state_plan(program: Any, shape: Any, modules: list[Any], functions: list[Any]) -> Any:
    """The sites a call reads constants from, rebuilt only when the shape moves."""

    try:
        cached = _state_plan_cache.get(program)
    except TypeError:
        cached = None
    if cached is not None and cached[0] == shape:
        return cached[1]
    plan = _build_read_plan(modules, functions)
    try:
        _state_plan_cache[program] = (shape, plan)
    except TypeError:
        pass
    return plan


def _build_read_plan(modules: list[Any], functions: list[Any]) -> tuple[Any, ...]:
    """Every site a capture may have read a value from, in a fixed order."""

    plan: list[Any] = []
    for function in functions:
        label = f"{function.__module__}.{function.__qualname__}"
        global_names, cell_indices = _constant_reads_of_function(function)
        for name in global_names:
            plan.append(("global", function, label, name, None))
        for index in cell_indices:
            plan.append(("closure", function, label, index, None))
    for module in modules:
        if getattr(module, "__dict__", {}).get("forward") is not None:
            names = _constant_attribute_names(type(module), module.forward)
        else:
            names = _constant_attribute_names_of_type(type(module))
        for name in names:
            plan.append(("attribute", module, None, name, None))
    plan.append(("values", None, None, None, [[], [], None, []]))
    return tuple(plan)


def _planned_constant_component(plan: tuple[Any, ...]) -> tuple[Any, ...]:
    """Read every planned site, reusing the description of an unchanged value."""

    values, cached_entries, last, reusable_flags = plan[-1][4]
    sites = plan[:-1]
    # Steady state: every site still holds the object it held last time, which
    # for a reusable value is the same value, so the description cannot have
    # moved.  Establishing that costs one comparison per site and nothing else.
    if last is not None and len(values) >= len(sites):
        for index, (kind, owner, label, name, _) in enumerate(sites):
            if kind == "attribute":
                value = owner.__dict__.get(name, _MISSING)
                if value is _MISSING:
                    value = type(owner).__dict__.get(name, _MISSING)
            elif kind == "global":
                value = owner.__globals__.get(name)
            else:
                try:
                    value = owner.__closure__[name].cell_contents
                except ValueError:
                    value = _MISSING
            # A site that reads nothing contributes no entry, but the slot
            # records that, so the scan can certify it like any other site.
            if value is not values[index]:
                break
            if value is _MISSING:
                continue
            if not reusable_flags[index]:
                # A value described by its contents can be edited in place, so
                # staying the same object is not enough; only its own
                # description has to be read again, not every site's.
                if _constant_entry(value) != cached_entries[index]:
                    break
        else:
            return last

    entries: list[tuple[Any, ...]] = []
    for index, (kind, owner, label, name, _) in enumerate(sites):
        if kind == "attribute":
            value = owner.__dict__.get(name, _MISSING)
            if value is _MISSING:
                value = type(owner).__dict__.get(name, _MISSING)
        elif kind == "global":
            value = owner.__globals__.get(name)
        else:
            try:
                value = owner.__closure__[name].cell_contents
            except ValueError:
                value = _MISSING
        if len(values) <= index:
            values.extend([_MISSING] * (index + 1 - len(values)))
            cached_entries.extend([None] * (index + 1 - len(cached_entries)))
            reusable_flags.extend([False] * (index + 1 - len(reusable_flags)))
        if value is _MISSING:
            # A name that reads nothing here stays recorded, so the
            # steady-state scan can certify it the same way as any other site.
            values[index] = _MISSING
            cached_entries[index] = None
            reusable_flags[index] = False
            continue
        reusable = _reusable_by_identity(value)
        if values[index] is value and reusable:
            entry = cached_entries[index]
        else:
            entry = _constant_entry(value)
            values[index] = value
            cached_entries[index] = entry
            # Certified here once: while the slot keeps this object, the flag
            # describes it, and the steady-state scan reads nothing else.
            reusable_flags[index] = bool(reusable)
        if entry is None:
            continue
        if kind == "attribute":
            entries.append(("attribute", id(owner), name, entry))
        elif kind == "global":
            entries.append(("global", label, name, entry))
        else:
            entries.append(("closure", label, name, entry))
    component = tuple(entries)
    plan[-1][4][2] = component
    return component


def _free_tensor_sources(
    function: types.FunctionType,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """The global names and closure cells through which ``function`` reads tensors.

    Only names the code loads as globals count -- an attribute that happens
    to share a global's name is not a read of it -- and functions defined
    inside the body are searched too, since they read the same globals.
    """

    import tensorplay

    scope = function.__globals__
    global_names = tuple(
        sorted(
            name
            for name in _loaded_global_names(function)
            if isinstance(scope.get(name), tensorplay.Tensor)
        )
    )
    cell_indices = []
    for index, cell in enumerate(function.__closure__ or ()):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, tensorplay.Tensor):
            cell_indices.append(index)
    return global_names, tuple(cell_indices)


def _module_state_slots(modules: list[Any]) -> list[tuple[Any, str, str, str]]:
    """Every parameter and buffer of ``modules``, as (module, table, name, label).

    A tensor held under several names is listed once, under the first.
    """

    slots: list[tuple[Any, str, str, str]] = []
    seen: set[int] = set()
    for index, root in enumerate(modules):
        for qualname, module in root.named_modules(remove_duplicate=True):
            for table in ("_parameters", "_buffers"):
                for name, value in getattr(module, table, {}).items():
                    if value is None or id(value) in seen:
                        continue
                    seen.add(id(value))
                    label = ".".join(p for p in (str(index), qualname, name) if p)
                    slots.append((module, table, name, label))
    return slots


def _lift_free_tensors(
    model: Callable[..., Any],
) -> tuple[Callable[..., Any], Callable[[], tuple[Any, ...]]] | None:
    """Make the tensors a function reads from outside into inputs of the region.

    A tensor a function reaches through a global or a closure -- directly, or
    as a parameter or buffer of a module it reaches that way -- is read when
    the function runs, not when it was compiled: an in-place update between
    calls, or a name rebound to another tensor, must reach the next call, and
    work done on such a tensor alone belongs to the call rather than to the
    capture.  So the region is captured from a copy of the function that takes
    those tensors as leading positional inputs, and every call reads them from
    where the function would.  Returns ``None`` when the function reads none.
    """

    if not isinstance(model, types.FunctionType) or getattr(model, "_tensorplay_lifted", False):
        return None
    try:
        signature = inspect.signature(model)
    except (TypeError, ValueError):
        return None
    global_names, cell_indices = _free_tensor_sources(model)
    slots = _module_state_slots(_modules_reached(model))
    if not global_names and not cell_indices and not slots:
        return None

    taken = set(signature.parameters)
    free_names = []
    for name in [
        *global_names,
        *(model.__code__.co_freevars[i] for i in cell_indices),
        *(label for *_, label in slots),
    ]:
        candidate = "free_" + re.sub(r"\W", "_", name)
        while candidate in taken:
            candidate = f"_{candidate}"
        taken.add(candidate)
        free_names.append(candidate)
    num_globals = len(global_names)
    num_direct = num_globals + len(cell_indices)
    count = len(free_names)
    missing = object()

    def program(*args: Any, **kwargs: Any) -> Any:
        free, rest = args[:count], args[count:]
        scope = dict(model.__globals__)
        scope.update(zip(global_names, free[:num_globals]))
        closure = model.__closure__
        if cell_indices:
            cells = list(closure)
            for index, value in zip(cell_indices, free[num_globals:num_direct]):
                cells[index] = types.CellType(value)
            closure = tuple(cells)
        body = types.FunctionType(
            model.__code__, scope, model.__name__, model.__defaults__, closure
        )
        body.__kwdefaults__ = model.__kwdefaults__
        # A module's state is read through the module, so the values handed
        # in stand in its attributes while the body runs, as a traced module's
        # own state does.
        patched = []
        try:
            for (module, _, name, _), value in zip(slots, free[num_direct:]):
                patched.append((module, name, module.__dict__.get(name, missing)))
                module.__dict__[name] = value
            return body(*rest, **kwargs)
        finally:
            for module, name, previous in reversed(patched):
                if previous is missing:
                    module.__dict__.pop(name, None)
                else:
                    module.__dict__[name] = previous

    functools.update_wrapper(program, model)
    program._tensorplay_lifted = True  # type: ignore[attr-defined]
    program.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY)
            for name in free_names
        ]
        + list(signature.parameters.values())
    )

    def read_free() -> tuple[Any, ...]:
        scope = model.__globals__
        closure = model.__closure__
        return (
            *(scope.get(name) for name in global_names),
            *(closure[index].cell_contents for index in cell_indices),
            *(getattr(module, table).get(name) for module, table, name, _ in slots),
        )

    return program, read_free


class _WithFreeTensors:
    """A compiled region handed the tensors its function reads from outside.

    Calls go to the region compiled from the lifted copy, with the current
    value of every such tensor ahead of the caller's arguments; what the
    compiled callable reports about itself is read from that region.
    """

    def __init__(
        self,
        model: Callable[..., Any],
        compiled: Callable[..., Any],
        read_free: Callable[[], tuple[Any, ...]],
    ) -> None:
        self._compiled = compiled
        self._read_free = read_free
        self._tensorplay_original = model
        functools.update_wrapper(self, model)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._compiled(*self._read_free(), *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_tensorplay_"):
            return getattr(self._compiled, name)
        raise AttributeError(name)


def _bind_sample_arguments(
    model: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind example call arguments to parameter names for the tracer.

    Metadata reads on placeholders (``x.shape[0] > 2``, ``range(x.ndim)``)
    then specialize statically during capture; the compile signature already
    keys on these fields, so no additional recompile conditions appear.
    """

    target = model.forward if _is_module_like(model) else model
    try:
        signature = inspect.signature(target)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError):
        return None
    return dict(bound.arguments)


def _is_module_like(value: Any) -> bool:
    return hasattr(value, "named_modules") and callable(
        getattr(value, "forward", None)
    )


def _region_is_training(
    example_inputs: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    graph_module: GraphModule | None = None,
) -> bool:
    """Whether any example input carries an autograd requirement.

    A region needs the autograd pipeline when any example input requires
    grad, when the captured module root is in training mode, or when any
    captured attribute (module parameter or buffer) requires grad.
    """

    import tensorplay

    def touches_autograd(value: Any) -> bool:
        if isinstance(value, tensorplay.Tensor):
            return bool(value.requires_grad)
        if isinstance(value, (tuple, list)):
            return any(touches_autograd(item) for item in value)
        if isinstance(value, dict):
            return any(touches_autograd(item) for item in value.values())
        return False

    if any(touches_autograd(item) for item in example_inputs) or any(
        touches_autograd(item) for item in example_kwargs.values()
    ):
        return True
    if graph_module is None:
        return False
    # A module left in training mode is a training region even when a call's
    # inputs carry no gradient requirement: its parameters still do, and the
    # backward must be compiled alongside the forward.
    root = getattr(graph_module, "root", None)
    if isinstance(root, tensorplay.nn.Module) and root.training:
        return True
    for node in graph_module.graph.nodes:
        if node.op != "get_attr":
            continue
        try:
            value = graph_module._get_attr(node.target)
        except (AttributeError, KeyError, IndexError, TypeError):
            continue
        if touches_autograd(value):
            return True
    return False


def _adapt_backend_to_region(
    compiler_fn: CompilerFn,
    example_inputs: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    graph_module: GraphModule | None = None,
    backend_kwargs: dict[str, Any] | None = None,
) -> CompilerFn:
    """Match a backend to the region being compiled.

    Training regions are routed through ahead-of-time autograd so parameters
    are lifted into explicit inputs before the backend sees them. Inference
    regions normally receive an operator trace when the backend requests it.
    A backend supporting symbolic capture can use the canonical graph directly
    to preserve shape expressions; container inputs still require flattening.
    """

    symbolic_inference = (
        getattr(graph_module, "meta", {}).get("symbolic_shapes")
        and getattr(compiler_fn, "supports_symbolic_shapes", False)
        and not graph_module.meta.get("static_shape_params")
    )
    if not _region_is_training(
        example_inputs, example_kwargs, graph_module
    ) and (not getattr(compiler_fn, "lowers_operator_graphs", False) or symbolic_inference):
        return compiler_fn
    from .aot_autograd import min_cut_rematerialization_partition
    from .common import aot_autograd

    forward_compiler = compiler_fn
    if backend_kwargs:
        kwargs = dict(backend_kwargs)

        def forward_compiler(graph, inputs):
            return compiler_fn(graph, inputs, **kwargs)

        forward_compiler.__name__ = getattr(compiler_fn, "__name__", "compiler_fn")

    from tensorplay._decomp import get_decompositions
    import tensorplay
    from .exact_narrowing import exact_narrowing_rules

    _NORM_DECOMPOSITION_OPS = (
        tensorplay.ops.tp.native_group_norm,
        tensorplay.ops.tp.native_group_norm_backward,
    )

    # A backend that lowers operator graphs says which operations it would
    # rather see as their parts; the trace writes those in their parts.
    backend_decompositions = getattr(compiler_fn, "trace_decompositions", None)
    adapted = aot_autograd(
        dynamic_shapes=(backend_kwargs or {}).get("dynamic") is True,
        fw_compiler=forward_compiler,
        bw_compiler=forward_compiler,
        # Keep the cheapest cut of forward values and let the backward
        # compute the rest again inside its own loops, rather than have the
        # forward write every value the backward reads.  A backend that has
        # rewrites for the whole joint graph brings its own cut, which runs
        # them before the split.
        partition_fn=getattr(
            compiler_fn, "partition_fn", min_cut_rematerialization_partition
        ),
        decompositions={
            **(backend_decompositions() if backend_decompositions is not None else {}),
            **get_decompositions(_NORM_DECOMPOSITION_OPS),
            **exact_narrowing_rules(),
        },
    )
    # The backend options are already bound into the compilers above; the
    # same options handed to the call again are not extra ones.
    adapted.options_bound = True
    return adapted


@contextlib.contextmanager
def _preserve_module_state(program: Any) -> Any:
    """Undo the state updates an executing capture makes.

    Capture runs the program to record it; in-place updates it performs on
    the parameters and buffers of the modules it reaches (running statistics,
    counters) must not count as a call, so their values are restored
    afterwards.
    """

    import tensorplay

    modules = _modules_reached(program)
    if not modules:
        yield
        return
    seen: set[int] = set()
    saved = []
    with tensorplay.no_grad():
        for module in modules:
            for t in (*module.parameters(), *module.buffers()):
                if id(t) not in seen:
                    seen.add(id(t))
                    saved.append((t, t.clone()))
    try:
        yield
    finally:
        with tensorplay.no_grad():
            for tensor, value in saved:
                if tuple(tensor.shape) == tuple(value.shape):
                    tensor.copy_(value)


@contextlib.contextmanager
def _preserve_input_values(*values: Any) -> Any:
    """Undo the input mutations an executing capture makes.

    Capture, propagation and lowering all execute the captured program to
    record or measure it; in-place updates it performs on its input tensors
    must not count as a call, so the input values are restored afterwards.
    The same values stand in for every compile-time run, which is what lets a
    program that mutates one of its arguments behave identically at compile
    time and at runtime without rewriting the caller's tensor.
    """

    import tensorplay

    with tensorplay.no_grad():
        saved: list[tuple[Any, Any]] = []

        def _save(value: Any) -> None:
            if isinstance(value, tensorplay.Tensor):
                saved.append((value, value.clone()))
            elif isinstance(value, (tuple, list)):
                for item in value:
                    _save(item)
            elif isinstance(value, dict):
                for item in value.values():
                    _save(item)

        _save(values)
    try:
        yield
    finally:
        with tensorplay.no_grad():
            for tensor, value in saved:
                if tuple(tensor.shape) == tuple(value.shape):
                    tensor.copy_(value)


def _shape_spec_for_examples(
    spec: dict[str, Any], samples: dict[str, Any]
) -> dict[str, Any]:
    """Drop the axes a caller marked static before the capture started.

    A mark overrides the declared policy for its axis, so an entry naming
    that axis is not a constraint the examples have to satisfy.
    """

    filtered: dict[str, Any] = {}
    for name, entry in spec.items():
        value = samples.get(name)
        marked = set(getattr(value, "_tensorplay_static_dims", ()))
        if not marked or entry is None:
            filtered[name] = entry
            continue
        if isinstance(entry, dict):
            filtered[name] = {
                axis: dim for axis, dim in entry.items() if axis not in marked
            }
        else:
            filtered[name] = tuple(
                None if axis in marked else dim
                for axis, dim in enumerate(entry)
            )
    return filtered


def _overlapping_input_groups(
    inputs: list[Any],
) -> list[tuple[int, int]]:
    """Pairs of tensor inputs whose storages share a byte range.

    Identity, not shape: a view and its base, or two slices of one tensor,
    overlap; two independent tensors never do, whatever their contents.
    """

    import tensorplay

    ranges: list[tuple[int, int, int, int]] = []
    for index, value in enumerate(inputs):
        if not isinstance(value, tensorplay.Tensor):
            continue
        try:
            pointer = value.untyped_storage().data_ptr()
            item = value.element_size()
            start = value.storage_offset() * item
            end = start + value.numel() * item
        except RuntimeError:
            continue
        ranges.append((index, pointer, start, end))
    pairs: list[tuple[int, int]] = []
    for position, (i, pi, si, ei) in enumerate(ranges):
        for j, pj, sj, ej in ranges[position + 1:]:
            if pi == pj and si < ej and sj < ei:
                pairs.append((i, j))
    return pairs


def _graph_writes_a_value(graph_module: Any) -> bool:
    """Whether the graph contains any in-place operation.

    A view-like chain from a placeholder to an in-place target cannot be
    ruled out cheaply, so an overlap plus *any* in-place operation is enough
    to refuse: refusing a compilable region is a lost optimization, while
    compiling one that shares a written buffer can produce the wrong answer.
    """

    for node in graph_module.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        target = node.target
        name = getattr(target, "__name__", None) or (target if isinstance(target, str) else "")
        if isinstance(name, str) and name.endswith("_") and not name.endswith("__"):
            return True
    return False


def _reject_overlapping_written_inputs(
    graph_module: Any, backend_inputs: list[Any]
) -> None:
    """Refuse a region whose aliased inputs it may write through."""

    pairs = _overlapping_input_groups(backend_inputs)
    if not pairs or not _graph_writes_a_value(graph_module):
        return
    first, second = pairs[0]
    raise GraphCaptureError(
        "inputs "
        f"#{first} and #{second} share storage and the region contains "
        "in-place operations; a write through one alias would change what "
        "the other reads, which this capture cannot order. Make the "
        "arguments independent or functionalize the writes."
    )


def _compile_region(
    model: Callable[..., Any],
    compiler_fn: CompilerFn,
    example_inputs: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    *,
    backend_kwargs: dict[str, Any],
    shape_spec: dict[str, Any] | None = None,
    region_key: str | None = None,
) -> tuple[Callable[..., Any], GraphModule]:
    devices = _generator_devices(example_inputs, example_kwargs)
    stored = load_region(region_key, model)
    if stored is not None:
        graph_module = stored
    else:
        import tensorplay

        with _preserve_input_values(example_inputs, example_kwargs), \
                tensorplay.random.fork_rng(devices=devices):
            try:
                with _compiler_context(), _preserve_module_state(model):
                    samples = _bind_sample_arguments(model, example_inputs, example_kwargs)
                    if shape_spec is not None:
                        # The declared dimensions are checked against this
                        # call's examples before capture reads them.
                        from tensorplay.export.dynamic_shapes import (
                            check_dynamic_shapes,
                        )

                        if samples is None:
                            raise RuntimeError(
                                "dynamic_shapes needs a program whose argument "
                                "names can be read"
                            )
                        check_dynamic_shapes(
                            samples, _shape_spec_for_examples(shape_spec, samples)
                        )
                    if (backend_kwargs or {}).get("dynamic") is True and getattr(
                        compiler_fn, "supports_symbolic_shapes", False
                    ):
                        from tensorplay.graph.experimental._symbolic_trace import SymbolicShapeTracer

                        tracer = SymbolicShapeTracer(samples or {}, dim_spec=shape_spec)
                    else:
                        if shape_spec is not None:
                            raise RuntimeError(
                                "the selected backend cannot capture symbolic "
                                "shapes; dynamic_shapes needs one that declares "
                                "supports_symbolic_shapes"
                            )
                        tracer = Tracer(execute=True)
                    # An operator that stands for a region of the program -- rather
                    # than for one operation -- has to be told that a capture is
                    # running, or it runs itself and the region it stands for is
                    # inlined into the graph as the operations that implement it.
                    # That inlined form is correct, and it is exactly what a backend
                    # holding a schedule for that region can never be handed.  The
                    # state is entered around the trace alone: the passes that
                    # follow rewrite the graph that came out, and one of them
                    # evaluates values, which must not be recorded as part of the
                    # region.
                    with tracer.proxy_mode:
                        graph_module = tracer.trace(
                            model,
                            sample_inputs=samples,
                        )
                    # Default capture pipeline: canonicalize operators, then constant
                    # folding, then decomposition, common-subexpression elimination
                    # and dead code elimination; fusion hints are stamped last so
                    # they see the final graph.  CSE runs after decomposition so
                    # shared sub-chains across rewritten composites collapse too
                    # (two gelu sites share one erf chain).  Backends always receive
                    # a folded, linted, hint-annotated graph; ShapeProp below
                    # additionally annotates tensor shapes.
                    pass_result = PassManager(
                        [
                            NormalizeOperators(),
                            ConstFold(),
                            DecomposePass(),
                            CSEPass(get_CSE_banned_ops()),
                            DeadCodeElimination(),
                            PointwiseFusionHint(),
                        ]
                    )(graph_module)
                    graph_module = pass_result.graph_module
            except GraphCaptureError as exc:
                raise GraphCaptureError(
                    "TensorPlay could not capture the requested compiler region: "
                    f"{exc}"
                ) from exc
            store_region(region_key, graph_module)
            tracer._node_samples.clear()
            tracer.sample_inputs.clear()
            tracer._samples.clear()
            _release_recorded_values(graph_module)

    # Capture, propagation and lowering all execute the program to record or
    # measure it.  The generator they advance belongs to the caller, so the
    # whole region is compiled inside a forked RNG: the draws the graph reads
    # at runtime start from where the caller left the generator, not from
    # where the compiler's own sample pass stopped.
    import tensorplay

    with _preserve_input_values(example_inputs, example_kwargs):
        with tensorplay.random.fork_rng(devices=devices):
            # Backend failures are compiler failures, not graph breaks.  In
            # particular, a Stax lowering error must not silently turn a requested
            # compiled region into an uncompiled call.
            # Registered backends receive graph inputs in placeholder order, including
            # values supplied through keywords and defaults.  Passing only
            # positional arguments makes a keyword-only/scalar placeholder appear
            # to be missing and is especially harmful for native Stax lowering.
            bound = graph_module.signature.bind_partial(*example_inputs, **example_kwargs)
            bound.apply_defaults()
            # Numeric-gate placeholders ride the contract as synthetic inputs; their
            # trace-time values stand in at lowering so kernels see real 0-d tensors.
            backend_inputs = []
            for node in graph_module.graph.placeholders:
                parameter_name = node.target if isinstance(node.target, str) else node.name
                try:
                    backend_inputs.append(bound.arguments[parameter_name])
                except KeyError:
                    if node.name not in bound.arguments:
                        raise GraphCaptureError(
                            f"missing sample value for graph placeholder {node.name!r}"
                        ) from None
                    backend_inputs.append(bound.arguments[node.name])

            _reject_overlapping_written_inputs(graph_module, backend_inputs)

            # Advisory shape/value metadata for backends and visualization; never a
            # reason to reject an otherwise compilable region.
            # Propagation executes the graph: it must not count as a call either.
            # It is also the second time the region runs before the backend has
            # looked at it, and the whole of that run is what it records, so it
            # is run against stand-ins that carry the geometry and hold nothing.
            # A region whose stand-ins cannot carry it is answered by running it
            # as written rather than being handed over undecided.
            advisory_device = next(
                (value.device for value in backend_inputs
                 if isinstance(value, tensorplay.Tensor)),
                None,
            )
            from tensorplay._subclasses import FakeTensorMode

            try:
                with _preserve_module_state(model), tensorplay.no_grad():
                    ShapeProp(backend_inputs, FakeTensorMode(advisory_device))(
                        graph_module
                    )
            except Exception:  # noqa: BLE001 - the run below is the fallback
                try:
                    with _preserve_module_state(model), tensorplay.no_grad():
                        ShapeProp(backend_inputs)(graph_module)
                except (GraphCaptureError, RuntimeError):
                    pass

            compiler_fn = _adapt_backend_to_region(
                compiler_fn,
                example_inputs,
                example_kwargs,
                graph_module,
                backend_kwargs,
            )

            regional_compile_invoke_subgraph(
                graph_module,
                compiler=compiler_fn,
                compiler_kwargs=backend_kwargs,
            )

            with _compiler_context():
                compiled = compiler_fn(graph_module, backend_inputs, **backend_kwargs)

    if not callable(compiled):
        raise TypeError(
            f"compiler backend returned {type(compiled)!r}; expected a callable"
        )
    if graph_module.meta.get("symbolic_shapes"):
        from tensorplay.compiler.backends.stax.shape_guards import build_capture_guard

        capture_guard = build_capture_guard(graph_module)
        backend_guard = getattr(compiled, "_tensorplay_guard", None)

        def check(*args, **kwargs):
            values = graph_module.signature.bind(*args, **kwargs)
            values.apply_defaults()
            ordered = [
                values.arguments[node.target] for node in graph_module.graph.placeholders
            ]
            return capture_guard(ordered) and (
                backend_guard is None or backend_guard(*args, **kwargs)
            )

        compiled._tensorplay_guard = check
    _release_recorded_values(graph_module)
    return compiled, graph_module


def _generator_devices(
    example_inputs: tuple[Any, ...], example_kwargs: dict[str, Any]
) -> list[int]:
    """The CUDA devices whose generators compiling a region may advance.

    Every device an input lives on, and the current device whenever CUDA is
    present, since a region can draw on it without reading anything there.
    Reading a device's generator state does not open a context on it.
    """

    import tensorplay

    devices: list[int] = []

    def add(index: int) -> None:
        if index not in devices:
            devices.append(index)

    for value in (*example_inputs, *example_kwargs.values()):
        if isinstance(value, tensorplay.Tensor) and value.device.type == "cuda":
            add(value.device.index or 0)
    cuda = getattr(tensorplay, "cuda", None)
    if cuda is not None and cuda.is_available():
        add(cuda.current_device())
    return devices


class _RecordedTensorMetadata:
    """Small tensor description retained for diagnostics after compilation."""

    __slots__ = (
        "shape", "dtype", "device", "requires_grad", "_stride", "_defined",
        "_storage_id", "_storage_offset",
    )

    def __init__(self, value: Any) -> None:
        from tensorplay._higher_order_ops._hop_base import value_device

        shape = getattr(value, "shape", ())
        self.shape = tuple(shape() if callable(shape) else shape)
        self.dtype = getattr(value, "dtype", None)
        # A description made while propagation runs on stand-ins has to name
        # where the value would live rather than where the stand-in sits: what
        # reads it later places kernels by that device.
        self.device = value_device(value)
        self.requires_grad = bool(getattr(value, "requires_grad", False))
        stride = getattr(value, "stride", ())
        self._stride = tuple(stride() if callable(stride) else stride)
        defined = getattr(value, "defined", None)
        self._defined = not callable(defined) or bool(defined())
        try:
            self._storage_id = value.untyped_storage().data_ptr()
            self._storage_offset = value.storage_offset()
        except (AttributeError, RuntimeError):
            self._storage_id = None
            self._storage_offset = 0

    def stride(self, dim: int | None = None) -> Any:
        return self._stride if dim is None else self._stride[dim]

    def defined(self) -> bool:
        return self._defined

    def storage_offset(self) -> int:
        return self._storage_offset

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def dim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        result = 1
        for size in self.shape:
            result *= int(size)
        return result

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def _device_type(self) -> str:
        return str(getattr(self.device, "type", "")) or ""

    @property
    def is_cpu(self) -> bool:
        return self._device_type() == "cpu"

    @property
    def is_cuda(self) -> bool:
        return self._device_type().startswith("cuda")

    @property
    def is_xpu(self) -> bool:
        return self._device_type() == "xpu"

    def is_floating_point(self) -> bool:
        return bool(getattr(self.dtype, "is_floating_point", False))

    def is_contiguous(self, memory_format: Any = None) -> bool:
        """Whether the strides are the dense row-major ones.

        Read off the shape and the strides kept in the description, so that a
        question about the layout can be answered after the value is gone.  A
        description written for another memory format, and one holding a size
        that is not known yet, cannot answer, so they say no: a layout is only
        reported when it is known to be that layout.
        """
        if memory_format is not None and "contiguous" not in str(memory_format):
            return False
        expected = 1
        try:
            for size, stride in zip(reversed(self.shape), reversed(self._stride)):
                if int(size) != 1 and int(stride) != expected:
                    return False
                expected *= int(size)
        except (TypeError, ValueError):
            return False
        return True


def _release_recorded_values(graph_module: Any) -> None:
    """Drop execution artifacts that capture-time propagation parked on nodes.

    Shape propagation executes the region on real tensors and leaves every
    intermediate on its node for the lowering pipeline to inspect.  Once the
    backend has produced its artifact those tensors are dead weight: one
    region can hold the entire forward's activation set, and every cached
    specialization would keep its own copy alive.  The cheap metadata each
    node carries (shape, dtype, device) stays behind, and propagation on a
    later compile of the same region regenerates whatever values the
    pipeline consumes.
    """
    try:
        nodes = graph_module.graph.nodes
    except AttributeError:
        return

    _release_recorded_values_of(nodes)
    samples = getattr(graph_module, "meta", {}).get("sample_inputs")
    if samples is not None:
        graph_module.meta["sample_inputs"] = _strip_recorded_value(samples)


def _release_recorded_values_of(nodes: Any) -> None:
    """Drop the execution artifacts parked on a set of nodes.

    A graph a transformation replaced is not the graph a later step reads,
    and the nodes it dropped are reachable through the references each node
    keeps to the nodes it reads and to its readers -- a cycle, which
    reference counting alone never breaks.  What those nodes recorded would
    therefore stay in device memory until the next cycle collection happened
    to run, so a caller that knows a graph is no longer read names its nodes
    here.
    """
    for node in nodes:
        _release_node_recorded_value(node)


def _release_node_recorded_value(node: Any) -> None:
    """Replace what a node recorded for one run with a description of it."""

    meta = getattr(node, "meta", None)
    if not meta:
        return
    meta.pop("widened_source", None)
    value = meta.get("val")
    if value is None:
        return
    meta["val"] = _strip_recorded_value(value)


def _strip_recorded_value(value: Any) -> Any:
    from tensorplay import Tensor

    if isinstance(value, Tensor):
        return _RecordedTensorMetadata(value)
    if isinstance(value, tuple):
        return tuple(_strip_recorded_value(item) for item in value)
    if isinstance(value, list):
        return [_strip_recorded_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_recorded_value(item) for key, item in value.items()}
    return value


_SHAPE_GUARD_ATTRS = frozenset({"shape", "len", "ndim"})


def _make_gate_evaluator(replay: dict[str, Any], target: Any) -> Callable[..., tuple]:
    """Build the per-call gate re-evaluator for one specialization (L1-D1).

    Replays the extracted condition subgraph on live inputs and returns the
    branch-deciding outcomes (``bool``/``iter`` gates) as the cache-key tail.
    Numeric-gate values never fragment the cache: they stay live inside the
    captured graph itself (GateValue proxies keep the condition subgraph
    reachable from the output), so the artifact recomputes them per call.
    """

    from tensorplay.graph import GraphModule, gate_outcome

    # The condition reads state off the module a bound method belongs to --
    # a buffer counting batches, say -- so the module is what it is read from;
    # the method only says how the call's arguments are named.
    owner = getattr(target, "__self__", None)
    import tensorplay

    root = owner if isinstance(owner, tensorplay.nn.Module) else target
    mini = GraphModule(
        root,
        replay["graph"],
        inspect.Signature(
            [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in replay["placeholders"]
            ]
        ),
    )
    gates = replay["gates"]
    target_signature = None
    try:
        target_signature = inspect.signature(target)
    except (TypeError, ValueError):
        pass

    def evaluate(args_: tuple[Any, ...], kwargs_: dict[str, Any]) -> tuple:
        if target_signature is not None:
            try:
                bound = target_signature.bind_partial(*args_, **kwargs_)
                bound.apply_defaults()
            except (TypeError, ValueError):
                return ("gates", "unbound")
            feeds_src = {
                name: bound.arguments.get(name) for name in replay["placeholders"]
            }
        else:
            feeds_src = dict(zip(replay["placeholders"], args_))
        values = mini._interpret(**feeds_src)
        outputs = values if isinstance(values, tuple) else (values,)
        # graph.gate() nodes stay symbolic inside the captured graph, so
        # their concrete values never fragment reuse; plain int()/float()
        # consumption bakes constants into the artifact and MUST key.
        symbolic = set(replay.get("symbolic") or ())
        key_tail: list[Any] = ["gates"]
        for output, (node_name, kind) in zip(outputs, gates):
            if kind in ("int", "float", "index") and node_name in symbolic:
                continue
            key_tail.append(gate_outcome(kind, output))
        return tuple(key_tail)

    return evaluate


def _extract_shape_guard_params(graph_module: GraphModule) -> frozenset[str]:
    """Parameters whose captured metadata reads require exact-shape guards.

    Branching on ``x.shape[0]`` bakes one side of the branch into the graph,
    so a dynamic-mode cache entry may only be reused while that placeholder's
    shape stays identical.  dtype/device/reads need no extra guards: they are
    already part of every specialization signature.
    """

    touches = getattr(graph_module, "meta", {}).get("metadata_touches") or ()
    if graph_module.meta.get("symbolic_shapes"):
        return frozenset(graph_module.meta.get("static_shape_params", ()))
    names = {name for name, attr in touches if attr in _SHAPE_GUARD_ATTRS}
    if not names or graph_module.signature is None:
        return frozenset()
    return frozenset(
        name for name in graph_module.signature.parameters if name in names
    )


def reset() -> None:
    """Clear all per-wrapper compiler specializations and backend state."""

    from .registry import reset_backends

    reset_backends()
    for wrapper in list(_compiled_wrappers):
        cache = getattr(wrapper, "_tensorplay_cache", None)
        if cache is not None:
            cache.clear()
        chains = getattr(wrapper, "_tensorplay_guard_chains", None)
        if chains is not None:
            chains.clear()
        try:
            wrapper._tensorplay_last_recompile_reasons = ()
        except Exception:
            pass
