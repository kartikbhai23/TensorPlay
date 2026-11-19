"""Values a capture reads from outside its arguments stay part of the key.

A global number, a closure cell, a module attribute or a rebinding of the
callable a function applies are all fixed while a region is traced.  Changing
one must produce a new region instead of reusing a program that embeds the old
value.
"""

import sys

import tensorplay as tp

_THIS = sys.modules[__name__]

_SCALE = 2.0
_SHIFT = (1.0, 2.0)
_OP = tp.sin
_CALLS = []


def _scaled(x):
    return x * _SCALE


def _shifted(x):
    return x + _SHIFT[0]


def _applied(x):
    return _OP(x)


def _recorded(x):
    _CALLS.append(1)
    return x + 1


def test_global_scalar_change_recompiles_with_new_value(monkeypatch):
    value = tp.ones(3)
    monkeypatch.setattr(_THIS, "_SCALE", 2.0)
    compiled = tp.compile(_scaled, fullgraph=True)
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]

    monkeypatch.setattr(_THIS, "_SCALE", 3.0)
    assert compiled(value).tolist() == [3.0, 3.0, 3.0]
    assert len(compiled._tensorplay_cache) == 2


def test_tuple_constant_change_recompiles_with_new_value(monkeypatch):
    value = tp.ones(3)
    monkeypatch.setattr(_THIS, "_SHIFT", (1.0, 2.0))
    compiled = tp.compile(_shifted, fullgraph=True)
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]

    monkeypatch.setattr(_THIS, "_SHIFT", (4.0, 2.0))
    assert compiled(value).tolist() == [5.0, 5.0, 5.0]


def test_closure_scalar_change_recompiles_with_new_value():
    def make():
        scale = 2.0

        def fn(x):
            return x * scale

        def set_scale(next_value):
            nonlocal scale
            scale = next_value

        return fn, set_scale

    fn, set_scale = make()
    value = tp.ones(3)
    compiled = tp.compile(fn, fullgraph=True)
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]

    set_scale(5.0)
    assert compiled(value).tolist() == [5.0, 5.0, 5.0]


def test_rebound_global_callable_recompiles(monkeypatch):
    value = tp.ones(3)
    monkeypatch.setattr(_THIS, "_OP", tp.sin)
    compiled = tp.compile(_applied, fullgraph=True)
    assert tp.allclose(compiled(value), tp.sin(value))

    monkeypatch.setattr(_THIS, "_OP", tp.cos)
    assert tp.allclose(compiled(value), tp.cos(value))


def test_side_effecting_global_does_not_force_recapture(monkeypatch):
    record = []
    monkeypatch.setattr(_THIS, "_CALLS", record)
    compiled = tp.compile(_recorded, fullgraph=True)
    value = tp.ones(3)
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]
    assert len(compiled._tensorplay_cache) == 1


_LIST_SCALE = [2.0]


def _list_scaled(x):
    return x * _LIST_SCALE[0]


def test_global_list_edited_in_place_recompiles(monkeypatch):
    value = tp.ones(3)
    monkeypatch.setattr(_THIS, "_LIST_SCALE", [2.0])
    compiled = tp.compile(_list_scaled, fullgraph=True)
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]

    entry = compiled._tensorplay_cache
    before = len(entry)
    _LIST_SCALE[0] = 3.0
    assert compiled(value).tolist() == [3.0, 3.0, 3.0]
    _LIST_SCALE.append(9.0)
    assert compiled(value).tolist() == [3.0, 3.0, 3.0]
    assert len(entry) > before


_TABLE = {"scale": 2.0}


def _table_scaled(x):
    return x * _TABLE["scale"]


def test_global_dict_edited_in_place_recompiles(monkeypatch):
    table = {"scale": 2.0}
    monkeypatch.setattr(_THIS, "_TABLE", table)
    compiled = tp.compile(_table_scaled, fullgraph=True)
    value = tp.ones(3)
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]

    table["scale"] = 4.0
    assert compiled(value).tolist() == [4.0, 4.0, 4.0]


class _Scaled(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 2.0

    def forward(self, x):
        return x * self.scale


class _ScaledChild(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = _Scaled()

    def forward(self, x):
        return self.child(x)


class _ScaledHelper(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 2.0

    def factor(self):
        return self.scale

    def forward(self, x):
        return x * self.factor()


def test_module_attribute_change_recompiles_with_new_value():
    module = _Scaled()
    value = tp.ones(3)
    compiled = tp.compile(module, fullgraph=True)
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]

    module.scale = 4.0
    assert compiled(value).tolist() == [4.0, 4.0, 4.0]


def test_nested_module_attribute_change_recompiles_with_new_value():
    module = _ScaledChild()
    value = tp.ones(3)
    compiled = tp.compile(module, fullgraph=True)
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]

    module.child.scale = 5.0
    assert compiled(value).tolist() == [5.0, 5.0, 5.0]


def test_module_reached_through_a_closure_stays_guarded():
    module = _Scaled()

    def fn(x):
        return module(x)

    value = tp.ones(3)
    compiled = tp.compile(fn, fullgraph=True)
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]

    module.scale = 6.0
    assert compiled(value).tolist() == [6.0, 6.0, 6.0]


def test_helper_method_attribute_change_recompiles_with_new_value():
    module = _ScaledHelper()
    value = tp.ones(3)
    compiled = tp.compile(module, fullgraph=True)
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]

    module.scale = 7.0
    assert compiled(value).tolist() == [7.0, 7.0, 7.0]


class _ListScaledModule(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = [2.0]

    def forward(self, x):
        return x * self.scale[0]


def test_module_list_attribute_edited_in_place_recompiles():
    module = _ListScaledModule()
    value = tp.ones(3)
    compiled = tp.compile(module, fullgraph=True)
    assert compiled(value).tolist() == [2.0, 2.0, 2.0]

    entry = compiled._tensorplay_cache
    before = len(entry)
    module.scale[0] = 3.0
    assert compiled(value).tolist() == [3.0, 3.0, 3.0]
    assert len(entry) > before