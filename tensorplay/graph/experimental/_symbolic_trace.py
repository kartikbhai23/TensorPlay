"""Capture shape reads and Python decisions as expressions with backed hints."""

from __future__ import annotations

import sympy

from .._utils import GraphCaptureError
from ..tracer import Tracer
from .sym_node import SymInt, SymNode


def _marked_static_axes(value):
    """Axes a caller pinned on this tensor before the capture started."""

    return tuple(getattr(value, "_tensorplay_static_dims", ()))


def _dynamic_axes(value, *, specialize_zero_one):
    """The automatic policy: every extent that is not 0 or 1 varies."""

    from tensorplay.export.dynamic_shapes import Dim

    marked = set(_marked_static_axes(value))
    return {
        axis: Dim.AUTO
        for axis, size in enumerate(value.shape)
        if axis not in marked
        and (not specialize_zero_one or size not in (0, 1))
    }


def _declared_axes(value, entry, *, specialize_zero_one):
    """The extents an explicit per-input specification leaves free."""

    from tensorplay.export.dynamic_shapes import Dim, _DimHint, _DimHintType

    if entry is None:
        return {}
    marked = set(_marked_static_axes(value))
    items = entry.items() if isinstance(entry, dict) else enumerate(entry)
    axes = {}
    for axis, dim in items:
        if axis >= len(value.shape):
            raise ValueError(
                f"dimension index {axis} is out of range for a "
                f"{len(value.shape)}-dimensional input"
            )
        if axis in marked:
            continue
        if dim is None or type(dim) is int:
            continue
        if isinstance(dim, _DimHint):
            if dim.type is _DimHintType.STATIC:
                continue
            if (
                dim.type is _DimHintType.AUTO
                and specialize_zero_one
                and value.shape[axis] in (0, 1)
            ):
                continue
        elif not isinstance(dim, Dim):
            raise TypeError(f"dimension spec must be int or Dim, got {type(dim)!r}")
        axes[axis] = dim
    return axes


class SymbolicShapeTracer(Tracer):
    def __init__(
        self,
        samples,
        *,
        specialize_zero_one=True,
        decomposition_table=None,
        dim_spec=None,
    ):
        super().__init__(execute=False)
        self.decomposition_table = dict(decomposition_table or {})
        from tensorplay.export._symbolic_dims import SymbolicDims

        if dim_spec is None:
            spec = {
                name: _dynamic_axes(
                    value, specialize_zero_one=specialize_zero_one
                )
                for name, value in samples.items()
                if hasattr(value, "shape")
            }
        else:
            spec = {
                name: _declared_axes(
                    value,
                    dim_spec.get(name),
                    specialize_zero_one=specialize_zero_one,
                )
                for name, value in samples.items()
                if hasattr(value, "shape")
            }
        self.dims = SymbolicDims.from_spec(samples, spec)
        self.static_dims = tuple(
            (name, axis, int(size))
            for name, value in samples.items()
            if hasattr(value, "shape")
            for axis, size in enumerate(value.shape)
            if axis not in spec[name]
        )
        from tensorplay.utils._pytree import tree_leaves

        self.static_params = tuple(
            name
            for name, value in samples.items()
            if not hasattr(value, "shape") and any(hasattr(leaf, "shape") for leaf in tree_leaves(value))
        )
        self.example_run = None
        self._shapes = {}
        #: value name -> axis -> the size a capture pinned it to
        self._static_extents = {}

    def trace(self, root, sample_inputs=None):
        from tensorplay.export._example_run import ExampleRun

        self.example_run = ExampleRun(self.dims, self._node_samples, lambda node: self._graph_attrs[node.target])
        try:
            graph = super().trace(root, sample_inputs)
            graph.meta["symbolic_shapes"] = True
            graph.meta["symbolic_dims"] = self.dims
            graph.meta["static_shape_dims"] = self.static_dims
            graph.meta["static_shape_params"] = self.static_params
            return graph
        finally:
            self.example_run.release()
            self.example_run = None

    def symbolic_shape(self, proxy):
        from tensorplay.export._shape_rules import Shaped

        value = self.example_run.value(proxy.node)
        if not isinstance(value, Shaped):
            return None
        if proxy.node.name not in self._shapes:
            pinned = self._static_extents.get(proxy.node.name, {})
            shape = []
            for axis, extent in enumerate(value.extents):
                if axis in pinned:
                    shape.append(pinned[axis])
                    continue
                if isinstance(extent, int):
                    shape.append(extent)
                    continue
                import tensorplay as tp

                item = self.create_proxy("call_function", tp.ops.tp.sym_size.int, (proxy, axis), {})
                self._node_samples[item.node.name] = SymInt(extent, None, int, self.dims.example(extent))
                shape.append(item)
            self._shapes[proxy.node.name] = tuple(shape)
        return self._shapes[proxy.node.name]

    def mark_static_dim(self, proxy, dim):
        """Pin one value's extents to the example sizes.

        A capture that sized its values symbolically can be told, while it
        runs, that an extent is in fact fixed for every call.  Reads made
        afterwards answer with the example size, and a condition on the
        expression the extent stood for is kept, so a call whose extent
        changed does not reuse the region.
        """

        from tensorplay.export._shape_rules import Shaped

        value = self.example_run.value(proxy.node)
        if not isinstance(value, Shaped):
            raise GraphCaptureError(
                "mark_static needs a tensor whose example extents are known"
            )
        rank = len(value.extents)
        if dim is None:
            axes = tuple(range(rank))
        else:
            if type(dim) is not int:
                raise TypeError(f"mark_static expects an int dim, got {type(dim)!r}")
            axis = dim + rank if dim < 0 else dim
            if axis < 0 or axis >= rank:
                raise IndexError(
                    f"dimension {dim} is out of range for a "
                    f"{rank}-dimensional value"
                )
            axes = (axis,)
        for axis in axes:
            self._pin_axis(proxy.node, value, axis)

    def _pin_axis(self, node, value, axis):
        known = self._static_extents.setdefault(node.name, {})
        if axis in known:
            return
        size = int(value.example.shape[axis])
        known[axis] = size
        extent = value.extents[axis]
        if isinstance(extent, int):
            if node.op == "placeholder":
                self.dims.static_axes[(str(node.target), axis)] = size
            return
        unknown = (
            getattr(extent, "free_symbols", set()) & self.dims.unknown.keys()
        )
        if unknown:
            raise GraphCaptureError(
                "mark_static cannot pin an extent that follows the data held "
                "by a tensor rather than the sizes of the inputs"
            )
        from tensorplay.export._symbolic_dims import Guard

        self.dims.guards.append(Guard(sympy.Eq(extent, size), node, size))
        if node.op == "placeholder":
            self.dims.static_axes[(str(node.target), axis)] = size
        shape = self._shapes.get(node.name)
        if shape is not None:
            self._shapes[node.name] = tuple(
                size if index == axis else item for index, item in enumerate(shape)
            )

    def known_metadata(self, proxy, name):
        from tensorplay.export._shape_rules import Shaped

        value = self.example_run.value(proxy.node)
        if isinstance(value, Shaped) and name in ("dtype", "device", "ndim"):
            return value.rank if name == "ndim" else getattr(value.example, name)
        return None

    def symbolic_gate(self, proxy, kind):
        sample = self._node_samples.get(proxy.node.name)
        if isinstance(sample, SymNode) and sample.hint is not None:
            return self.dims.decide(proxy.node, sample, kind)
        return None

    def create_proxy(self, kind, target, args, kwargs):
        decomposition = self.decomposition_table.get(target) if kind == "call_function" else None
        if decomposition is not None:
            result = decomposition(*args, **kwargs)
            if result is not NotImplemented:
                return result
        proxy = super().create_proxy(kind, target, args, kwargs)
        if kind in ("call_function", "call_method") and self.example_run is not None:
            from tensorplay.export._shape_rules import Shaped
            from ..node import map_aggregate

            leaves = []
            map_aggregate(self.resolve_sample((proxy.node.args, proxy.node.kwargs)), lambda value: leaves.append(value))
            if any(isinstance(value, SymNode) for value in leaves) and not any(
                hasattr(value, "shape") for value in leaves
            ):
                self._execute_node(proxy.node)
            if isinstance(self._node_samples.get(proxy.node.name), SymNode):
                return proxy

            value = self.example_run.record(proxy.node)
            if isinstance(value, sympy.Basic) and value.free_symbols:
                pytype = bool if isinstance(value, sympy.logic.boolalg.Boolean) else int
                from .sym_node import SymBool

                node_type = SymBool if pytype is bool else SymInt
                hint = value.xreplace(self.dims.hints)
                self._node_samples[proxy.node.name] = node_type(
                    value, None, pytype, bool(hint) if pytype is bool else int(hint)
                )
            elif isinstance(value, Shaped):
                proxy.node.meta["val"] = value.example
        return proxy
