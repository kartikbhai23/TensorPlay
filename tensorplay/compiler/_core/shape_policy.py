"""Per-input shape policies for one compiled region.

A compile call can ask for sizes to be treated in three ways.  The default is
the concrete policy: every extent read during capture is the one the example
had, and a different size is a different specialization.  ``dynamic=True``
reads every extent that is not 0 or 1 as a symbol, so one captured region
serves a range of sizes.  Between the two, a caller can name the dimensions
that may vary and pin the rest:

    compile(fn, dynamic_shapes={"x": {0: Dim("batch")}})

Only the named dimensions become symbols.  Every other axis keeps the
example's size and is checked before a cached region runs, so a call that
changes it recompiles rather than silently reusing a region sized for the
example.

The specification is validated in two steps.  Its structure and argument
names are checked when ``compile`` is called; the example sizes it mentions
are checked when the region is first captured, because only then are the
examples known.
"""

from __future__ import annotations

import inspect
from typing import Any

from tensorplay.export.dynamic_shapes import Dim, _DimHint

__all__ = [
    "normalize_dynamic_shapes",
    "parameter_names",
    "shape_policy_signature",
]


def parameter_names(model: Any) -> list[str] | None:
    """The named parameters of ``model``, or ``None`` if they cannot be read."""

    target = model.forward if _is_module_like(model) else model
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    names = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    return names


def _is_module_like(value: Any) -> bool:
    return callable(getattr(value, "forward", None)) and callable(
        getattr(value, "named_children", None)
    )


def normalize_dynamic_shapes(
    spec: Any,
    names: list[str] | None,
) -> dict[str, Any]:
    """Turn a public ``dynamic_shapes`` specification into a per-name mapping.

    A mapping is already keyed by argument name.  A sequence is positional,
    the way the call's positional arguments are.  Entries describe one
    input's axes as either a mapping from axis to dimension or a sequence
    whose non-``None`` entries are the dynamic axes.
    """

    if isinstance(spec, (list, tuple)):
        if names is None:
            raise TypeError(
                "a positional dynamic_shapes specification needs readable "
                "argument names"
            )
        if len(spec) > len(names):
            raise ValueError("dynamic_shapes has more entries than arguments")
        return {
            name: _validate_dimension_spec(value)
            for name, value in zip(names, spec)
            if value is not None
        }
    if not isinstance(spec, dict):
        raise TypeError("dynamic_shapes must be a mapping or a sequence")

    normalized: dict[str, Any] = {}
    for name, dims_spec in spec.items():
        if names is not None and name not in names:
            raise ValueError(
                f"dynamic_shapes key {name!r} does not match any argument; "
                f"expected one of {names}"
            )
        normalized[name] = _validate_dimension_spec(dims_spec)
    return normalized


def _validate_dimension_value(value: Any) -> Any:
    if value is None or isinstance(value, (Dim, _DimHint)):
        return value
    if type(value) is int:
        return value
    raise TypeError(
        "dimension spec must be int or Dim (a dim hint or None is also accepted), "
        f"got {type(value)!r}"
    )


def _validate_dimension_spec(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        result: dict[int, Any] = {}
        for index, item in value.items():
            if type(index) is not int or index < 0:
                raise TypeError(
                    f"dimension index must be a non-negative int, got {index!r}"
                )
            result[index] = _validate_dimension_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return tuple(_validate_dimension_value(item) for item in value)
    raise TypeError(
        "an input's dynamic_shapes entry must be a sequence of per-axis "
        "specifications or a mapping from axis to one, got "
        f"{type(value)!r}"
    )


def shape_policy_signature(spec: dict[str, Any] | None) -> Any:
    """A stable, hashable description of a shape specification.

    Capture results are cached by the program, its state and the region key;
    two compile calls that differ only in which dimensions may vary must not
    share a stored graph.
    """

    if spec is None:
        return None
    entries = []
    for name in sorted(spec):
        entry = spec[name]
        if entry is None:
            entries.append((name, None))
        elif isinstance(entry, dict):
            entries.append(
                (
                    name,
                    tuple(sorted((int(axis), repr(dim)) for axis, dim in entry.items())),
                )
            )
        else:
            entries.append((name, tuple(repr(dim) for dim in entry)))
    return tuple(entries)
