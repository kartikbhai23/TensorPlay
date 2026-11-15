# tensorplay.compiler

## Shape policy

A compile call chooses how tensor extents are treated:

* With no shape argument, capture fixes the example sizes.  If a later call
  misses only because a size changed (and nothing else in the guard set did),
  the frontend clears the concrete specialization and captures the region
  again with every extent that is not 0 or 1 symbolic, so one artifact then
  serves the sizes already seen and the ones that follow.  This upgrade needs
  a backend that declares `supports_symbolic_shapes` and is controlled by
  `tensorplay.compiler.config.automatic_dynamic_shapes`.
* `dynamic=True` captures extents symbolically from the first call.  It keeps
  the rank in the specialization key and lets the backend accept the shape
  range.
* `dynamic_shapes` names the axes that may vary, per argument:

  ```python
  tensorplay.compile(
      fn,
      dynamic_shapes={"x": {0: tensorplay.export.Dim("batch")}},
  )
  ```

  Unnamed axes keep the example size and are checked before a cached region
  runs, so changing one creates a new specialization instead of silently
  reusing a region sized for the example.  A dimension may carry `min`/`max`
  bounds; an example outside them is rejected before capture.  The
  specification needs a backend that declares `supports_symbolic_shapes`.

`tensorplay.compiler.mark_static(tensor, dim=None)` pins a tensor's extents (or
one axis) to the example sizes.  Called before `compile`, it removes those axes
from the shape policy; called inside the captured function, it makes reads of
that extent answer with the example size and keeps a condition on it, so a
changed size recompiles rather than reusing a region that assumed the old one.

`tensorplay.compiler.disallow_in_graph(fn)` declares that a callable must not
be absorbed into a captured region.  A capture that reaches it raises
`GraphCaptureError` instead of recording the call: the region is one graph, so
there is no partial capture to resume after the call, and recording a call
whose effect the graph cannot carry would repeat that effect once per compiled
call.

## Autograd limitations

In v1.1.0, regions compiled with STAX or an AOT autograd backend support
first-order reverse-mode gradients. They do not support second-order gradients
(double backward). A first gradient computed with `create_graph=True` has the
correct value, but differentiating it again raises `RuntimeError` with an
explicit message about this limitation.

Run the original function without compilation when computing Hessians,
Hessian-vector products, or other higher-order derivatives. Operator support
for higher-order gradients in eager execution does not extend to compiled
regions.

## Classes

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.compiler.GraphCaptureError
    tensorplay.compiler.InvalidBackend
    tensorplay.compiler.NestedCompileRegionOptions
```

## Functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.compiler.allow_in_graph
    tensorplay.compiler.assume_constant_result
    tensorplay.compiler.compile
    tensorplay.compiler.disallow_in_graph
    tensorplay.compiler.disable
    tensorplay.compiler.disable_capture
    tensorplay.compiler.export
    tensorplay.compiler.get_backend_capabilities
    tensorplay.compiler.get_default_backend
    tensorplay.compiler.is_compiling
    tensorplay.compiler.is_exporting
    tensorplay.compiler.list_backends
    tensorplay.compiler.list_mode_options
    tensorplay.compiler.lookup_backend
    tensorplay.compiler.mark_static
    tensorplay.compiler.nested_compile_region
    tensorplay.compiler.nonstrict_trace
    tensorplay.compiler.overload_method
    tensorplay.compiler.register_backend
    tensorplay.compiler.register_debug_backend
    tensorplay.compiler.register_experimental_backend
    tensorplay.compiler.reset
    tensorplay.compiler.set_default_backend
    tensorplay.compiler.substitute_in_graph
    tensorplay.compiler.unregister_backend
    tensorplay.compiler.unused
```
