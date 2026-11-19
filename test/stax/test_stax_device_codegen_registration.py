"""What a device's work is written through, including a device outside this repository.

A device outside the repository names itself and installs a module; the code
generator asks that module for its hooks by name.  The cases below are the
three answers that matter: a module that publishes all of them gets the device
registered, a module that publishes none leaves it unregistered rather than
failing, and a named hook that was never published is an error at the lookup.
"""
import types

import pytest

import tensorplay as tp
from tensorplay.compiler.backends.stax.codegen import common as codegen
from tensorplay.utils import backend_registration

OUT_OF_TREE = "outsidetree"


@pytest.fixture
def out_of_tree_device(monkeypatch):
    """The out-of-tree slot, named, with a module installed for it."""

    monkeypatch.setattr(
        backend_registration, "get_privateuse1_backend_name", lambda: OUT_OF_TREE
    )
    module = types.ModuleType(OUT_OF_TREE)
    monkeypatch.setattr(tp, OUT_OF_TREE, module, raising=False)
    yield module
    codegen.device_codegens.pop(OUT_OF_TREE, None)
    codegen.custom_backend_passes.pop(OUT_OF_TREE, None)
    codegen.custom_backend_codegen_configs.pop(OUT_OF_TREE, None)


def test_an_out_of_tree_device_publishes_what_its_work_is_written_through(
    out_of_tree_device,
):
    scheduling = object()
    wrapper_codegen = object()
    cpp_wrapper_codegen = object()
    fx_wrapper_codegen = object()
    out_of_tree_device.Scheduling = scheduling
    out_of_tree_device.PythonWrapperCodegen = wrapper_codegen
    out_of_tree_device.CppWrapperCodegen = cpp_wrapper_codegen
    out_of_tree_device.WrapperFxCodegen = fx_wrapper_codegen

    codegen.init_backend_registration()

    entry = codegen.device_codegens[OUT_OF_TREE]
    assert entry.scheduling is scheduling
    assert entry.wrapper_codegen is wrapper_codegen
    assert entry.cpp_wrapper_codegen is cpp_wrapper_codegen
    assert entry.fx_wrapper_codegen is fx_wrapper_codegen


def test_an_out_of_tree_device_without_hooks_stays_unregistered(out_of_tree_device):
    # A device that publishes no code generator is not this compiler's to
    # write; leaving it out is silence here and a refusal where its work is
    # actually reached.
    codegen.init_backend_registration()

    assert codegen.get_scheduling_for_device(OUT_OF_TREE) is None


def test_a_partial_set_of_hooks_leaves_the_device_unregistered(out_of_tree_device):
    # The hooks are taken as a set: a device that publishes a scheduler but no
    # way to hand results back is not half-registered.
    out_of_tree_device.Scheduling = object()
    out_of_tree_device.PythonWrapperCodegen = object()
    out_of_tree_device.CppWrapperCodegen = object()

    codegen.init_backend_registration()

    assert codegen.get_scheduling_for_device(OUT_OF_TREE) is None


def test_a_hook_the_device_module_did_not_publish_is_an_error(out_of_tree_device):
    with pytest.raises(RuntimeError, match="not available"):
        backend_registration._get_custom_mod_func("Scheduling")

    out_of_tree_device.Scheduling = object()
    assert (
        backend_registration._get_custom_mod_func("Scheduling")
        is out_of_tree_device.Scheduling
    )