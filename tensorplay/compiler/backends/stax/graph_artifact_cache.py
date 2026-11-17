"""A cache of finished regions, keyed by what the region is made of.

A region's artifact is decided by the captured operations, the values the
region reads from its module, the example inputs it was specialized to, the
settings in force, and the toolchain that builds the result.  All of those are
known before lowering starts, so a region compiled before is recognised there
and its artifact is loaded instead of being lowered, scheduled, printed and
built again.

The payload holds plain Python data plus the constants the printed code reads
by name; nothing callable is stored, because a built callable belongs to the
process that built it.  A region whose constants cannot be written, or whose
graph reads a value that cannot be described, is simply not cached: every
failure here leaves the caller on the uncached path, and a directory that
belongs to another user is never written.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import os
import pickle
from typing import Any

from .cache_key import FX_GRAPH_CACHE_KEY_STRATEGY
from .codegen.cpp_wrapper import CppWrapperCode
from .runtime.cache_dir_utils import cache_dir
from .utils import ValueWithLineMap

#: Bumped whenever anything that shapes the stored artifact changes in a way
#: the key below cannot see.  A key that misses such a change would hand back
#: an artifact built by the old rules, so the version is the escape hatch:
#: raise it, and every stored entry becomes unreadable at once.
SCHEMA_VERSION = "3"

#: How deep a nested argument is rendered before the walk stops.  Captured
#: arguments are shallow: nodes, scalars, and short sequences of both.
_MAX_ARG_DEPTH = 8


def _arg_digest(value: object, depth: int = 0) -> object:
    """A stable, JSON-able stand-in for one captured argument.

    Nodes stand in by name -- two captures of the same program name their
    nodes the same way -- and sequences and maps are walked so that a list of
    nodes does not collapse into one opaque repr.  Anything else is rendered
    with ``repr``, which for scalars and the other values a captured graph
    carries is both complete and stable.
    """

    if depth >= _MAX_ARG_DEPTH:
        return repr(value)
    if hasattr(value, "op") and hasattr(value, "name"):
        return {"kind": "node", "name": value.name}
    if isinstance(value, (list, tuple)):
        return {"kind": "seq", "items": [_arg_digest(v, depth + 1) for v in value]}
    if isinstance(value, dict):
        return {
            "kind": "map",
            "items": sorted(
                (str(k), _arg_digest(v, depth + 1)) for k, v in value.items()
            ),
        }
    return {"kind": "value", "text": repr(value)}


def _node_digest(node: object) -> dict:
    args = [_arg_digest(v) for v in getattr(node, "args", ()) or ()]
    kwargs = {k: _arg_digest(v) for k, v in (getattr(node, "kwargs", None) or {}).items()}
    cache_key = getattr(node.target, "_tp_cache_key", None)
    return {
        "op": node.op,
        "target": {"lowering": cache_key} if cache_key else repr(node.target),
        "name": node.name,
        "args": args,
        "kwargs": kwargs,
    }


def _tensor_spec(tensor: object) -> list:
    return [
        list(tensor.shape),
        str(tensor.dtype),
        list(tensor.stride()),
        f"{tensor.device.type}{tensor.device.index or 0}",
    ]


def _scalar_digest(value: object) -> dict:
    """A stand-in for one scalar argument that keeps its type and its bits.

    ``True``, ``1`` and ``1.0`` print differently in the wrapper, and two
    floats that differ in the last bit print a different literal, so the type
    is named and floats are rendered by their exact hexadecimal form.
    """

    if isinstance(value, float):
        text = float.hex(value)
    elif isinstance(value, complex):
        text = f"{float.hex(value.real)}+{float.hex(value.imag)}j"
    else:
        text = repr(value)
    return {"kind": "scalar", "type": type(value).__name__, "text": text}


def _input_digest(value: object, depth: int = 0) -> object:
    """A stable, JSON-able stand-in for one example input.

    Every input takes part in the key, not only the tensors: a scalar is
    printed into the wrapper as a literal, so two regions that differ only in
    the scalar they were traced with produce different artifacts.  Tensors
    stand in by their spec -- shape, dtype, stride, device -- which is what the
    printed code is specialized to, and the containers around them are walked
    so that a tuple of tensors and a list of the same tensors stay apart.
    """

    if depth >= _MAX_ARG_DEPTH:
        return {"kind": "opaque", "type": type(value).__name__, "text": repr(value)}
    if hasattr(value, "shape") and hasattr(value, "stride"):
        return {"kind": "tensor", "spec": _tensor_spec(value)}
    if isinstance(value, (list, tuple)):
        return {"kind": "seq", "items": [_input_digest(v, depth + 1) for v in value]}
    if isinstance(value, dict):
        return {
            "kind": "map",
            "items": sorted(
                (repr(k), _input_digest(v, depth + 1)) for k, v in value.items()
            ),
        }
    if isinstance(value, (bool, int, float, complex, str, bytes)) or value is None:
        return _scalar_digest(value)
    return {"kind": "opaque", "type": type(value).__name__, "text": repr(value)}


def _host_bytes(tensor: object) -> bytes | None:
    """The tensor's contents as bytes, or ``None`` when they cannot be read.

    The artifact's printed code reads its constants by name, so two artifacts
    differ only if the values they read differ.  A value this process cannot
    read back -- a device tensor, a dtype without a host view -- is a value
    whose difference cannot be proven, so it makes the region uncachable
    rather than keying it as if it were unchanged.
    """

    device = getattr(tensor, "device", None)
    if device is not None and getattr(device, "type", None) not in (None, "cpu"):
        return None
    try:
        value = tensor.detach() if hasattr(tensor, "detach") else tensor
        return bytes(memoryview(value.contiguous().numpy()).cast("B"))
    except Exception:  # noqa: BLE001 - an unreadable value disables caching
        return None


def _value_digest(value: object, depth: int = 0) -> object:
    """A stable stand-in for one value the region reads from its module.

    Parameters and buffers stand in by their spec and their contents: the
    artifact is built for the values that were in place when it was lowered,
    so a value that changed has to produce a different key.  Plain scalars and
    containers of them are walked; anything else is a value whose stability
    cannot be proven, and ``None`` says so.
    """

    if depth >= _MAX_ARG_DEPTH:
        return None
    if hasattr(value, "shape") and hasattr(value, "stride"):
        contents = _host_bytes(value)
        if contents is None:
            return None
        return {
            "kind": "tensor",
            "spec": _tensor_spec(value),
            "digest": hashlib.sha256(contents).hexdigest(),
        }
    if isinstance(value, (list, tuple)):
        items = [_value_digest(v, depth + 1) for v in value]
        if any(item is None for item in items):
            return None
        return {"kind": "seq", "items": items}
    if isinstance(value, dict):
        items = sorted((str(k), _value_digest(v, depth + 1)) for k, v in value.items())
        if any(item is None for _, item in items):
            return None
        return {"kind": "map", "items": items}
    if isinstance(value, (bool, int, float, complex, str, bytes)) or value is None:
        return _scalar_digest(value)
    return None


def _module_state_digest(graph_module: object) -> dict | None:
    """What each value the region reads from its module is, by name.

    Only the attributes the graph actually reads take part: a module's other
    state says nothing about what this region computes, and including it would
    turn an unrelated change into a miss.  A read whose value cannot be
    described makes the whole region uncachable.
    """

    state: dict[str, object] = {}
    for node in graph_module.graph.nodes:
        if getattr(node, "op", None) != "get_attr":
            continue
        target = str(node.target)
        try:
            value = graph_module._get_attr(target)
        except Exception:  # noqa: BLE001 - a read that fails here may still lower
            return None
        digest = _value_digest(value)
        if digest is None:
            return None
        state[target] = digest
    return state


def _config_digest() -> dict:
    """Every setting of this module, private and sectioned alike.

    The scheduler and the code generators read settings from the module's
    top level and from the sections inside it, and both a top-level name and a
    sectioned one can change what is printed.  Rather than enumerating names
    that would fall behind, the portable snapshot the module already offers is
    used: it flattens the sections and writes callables by their qualified
    name, so it differs when behaviour differs.
    """

    from . import config
    from ... import config as frontend_config

    return {
        "stax": config.save_config_portable(ignore_private_configs=False),
        "frontend": {
            "cache_key_tag": frontend_config.cache_key_tag,
            "recompile_limit": frontend_config.recompile_limit,
        },
    }


def region_key(
    graph_module: object,
    example_inputs: object,
    *,
    is_inference: bool,
    is_backward: bool,
    aot_mode: bool,
    use_cuda_codegen: bool,
) -> str | None:
    """The name under which this region's finished artifact is stored.

    Everything the artifact depends on goes in: the captured program node by
    node, every example input the region is specialized to -- tensors by their
    spec and scalars by their exact value -- every value the region reads from
    its module by spec and content, the settings in force, the toolchain that
    will build the result, the mode the artifact is built for, and the version
    above.  ``None`` means the region is not cachable, because a value it
    depends on could not be described.
    """

    from .codecache import CacheBase, package_key
    from .cpp_builder import get_compiler_version_info, get_cpp_compiler

    state = _module_state_digest(graph_module)
    if state is None:
        return None
    components = {
        "schema": SCHEMA_VERSION,
        "graph": [_node_digest(node) for node in graph_module.graph.nodes],
        "inputs": [_input_digest(value) for value in example_inputs],
        "state": state,
        "flags": {
            "is_inference": is_inference,
            "is_backward": is_backward,
            "aot_mode": aot_mode,
            "cuda_codegen": use_cuda_codegen,
        },
        "config": _config_digest(),
        "compiler": get_compiler_version_info(get_cpp_compiler()),
        # The generator's own version and source, so an artifact written by an
        # older build is never read as one this build would have written.
        "package": package_key().hex(),
        "system": CacheBase.get_system()["hash"],
    }
    return FX_GRAPH_CACHE_KEY_STRATEGY.key_from_json(components)


@dataclasses.dataclass(frozen=True)
class RegionArtifact:
    """A region's finished artifact, as it comes back from the cache.

    The wrapper is the printed program the code generators produce, and the
    constants are what that program reads by name.  A caller turns the two
    back into a callable without lowering anything.
    """

    wrapper_code: CppWrapperCode | ValueWithLineMap
    constants: dict[str, Any]
    opaque_value_type_classes: dict[str, Any]
    single_output: bool


def _wrapper_payload(wrapper_code: object) -> dict:
    """The printed program, as plain data.

    A C++ wrapper carries the source plus the metadata the built module reads
    at call time -- names, indices, shapes, strides, all plain values.  A
    Python wrapper is nothing but its text; the line map is the bookkeeping of
    the printing itself (it points errors back at the region that produced
    each line) and is not part of the artifact, so a loaded wrapper reports
    from the wrapper's own lines instead.
    """

    if isinstance(wrapper_code, ValueWithLineMap):
        return {"kind": "py", "value": wrapper_code.value}
    return {
        "kind": "cpp",
        "value": wrapper_code.value,
        "buffer_meta": wrapper_code.buffer_meta,
        "output_names": wrapper_code.output_names,
        "tensor_index": wrapper_code.tensor_index,
        "num_tensors": wrapper_code.num_tensors,
        "num_sizevars": wrapper_code.num_sizevars,
        "graph_input_names": wrapper_code.graph_input_names,
        "view_meta": getattr(wrapper_code, "view_meta", None) or {},
    }


def _wrapper_from_payload(payload: dict) -> CppWrapperCode | ValueWithLineMap:
    if payload.get("kind") == "py":
        return ValueWithLineMap(value=payload["value"], line_map=[])
    return CppWrapperCode(
        value=payload["value"],
        line_map=[],
        buffer_meta=payload["buffer_meta"],
        output_names=payload["output_names"],
        tensor_index=payload["tensor_index"],
        sizevar_index={},
        num_tensors=payload["num_tensors"],
        num_sizevars=payload["num_sizevars"],
        graph_input_names=payload["graph_input_names"],
        view_meta=payload.get("view_meta") or {},
    )


def _artifact_path(key: str) -> str:
    from .codecache import CacheBase

    return os.path.join(
        cache_dir(), "cache", CacheBase.get_system()["hash"], "graph", f"{key}.pkl"
    )


def lookup(key: str | None) -> RegionArtifact | None:
    """The finished artifact of this region, if one is already stored.

    A region whose artifact is on disk comes back as the same wrapper the code
    generators would have produced, along with the constants it reads and can
    store.  Anything unreadable -- a missing file, a payload from an older
    schema, a damaged entry, a class that no longer imports -- is a miss, not
    an error.
    """

    if key is None or not _enabled():
        return None
    try:
        path = _artifact_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fp:
            payload = pickle.load(fp)
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA_VERSION:
            return None
        classes: dict[str, Any] = {}
        for name, (module_name, qualname) in payload["opaque_classes"].items():
            value: Any = importlib.import_module(module_name)
            for part in qualname.split("."):
                value = getattr(value, part)
            classes[name] = value
        return RegionArtifact(
            wrapper_code=_wrapper_from_payload(payload["wrapper"]),
            constants=payload["constants"],
            opaque_value_type_classes=classes,
            single_output=bool(payload["single_output"]),
        )
    except Exception:
        if os.environ.get("TP_STAX_CACHE_DEBUG"):
            import traceback

            traceback.print_exc()
        return None


def publish(
    key: str | None,
    wrapper_code: object,
    constants: dict[str, Any],
    opaque_value_type_classes: dict[str, Any],
    single_output: bool,
    torchbind_constants: dict[str, Any] | None = None,
) -> None:
    """Store the artifact this region just finished, under its own key.

    A region that carries a scripted object, or whose constants or classes
    cannot be written, is not stored: an artifact that came back missing part
    of what it reads would be worse than no artifact.  Failures are silent for
    the same reason a miss is silent -- the next run simply walks the uncached
    path again.
    """

    if key is None or not _enabled():
        return
    if not isinstance(wrapper_code, (CppWrapperCode, ValueWithLineMap)):
        return
    if torchbind_constants:
        return
    try:
        classes = {}
        for name, value in opaque_value_type_classes.items():
            module_name = getattr(value, "__module__", None)
            qualname = getattr(value, "__qualname__", None)
            if not module_name or not qualname:
                return
            classes[name] = (module_name, qualname)
        payload = {
            "schema": SCHEMA_VERSION,
            "wrapper": _wrapper_payload(wrapper_code),
            "constants": dict(constants),
            "opaque_classes": classes,
            "single_output": bool(single_output),
        }
        # The pickling is done before the directory is touched, so a value
        # that cannot be written leaves no entry behind.
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        path = _artifact_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as fp:
            fp.write(blob)
        os.replace(path + ".tmp", path)
    except Exception:
        if os.environ.get("TP_STAX_CACHE_DEBUG"):
            import traceback

            traceback.print_exc()


def _enabled() -> bool:
    from . import config

    return config.cache_graph_artifacts is not False and (
        not config.force_disable_caches
    )
