"""The settings surface: every read resolves, and the snapshot is portable.

A setting a ported module reads has to exist in this module; an absent one is
an ``AttributeError`` on a path that may only run for one machine or one
configuration.  The snapshot is part of a cache key, so it also has to be
writable as text and to change when a setting does.
"""

import json

import pytest

from tensorplay.compiler.backends.stax import config
from tensorplay.compiler.backends.stax import comm_analysis
from tensorplay.compiler.backends.stax.graph_passes import bucketing
from tensorplay.compiler.backends.stax.runtime.caching import context


def test_settings_read_by_ported_modules_exist():
    assert config.compile_threads is None or isinstance(
        config.compile_threads, int
    )
    assert config.cudagraph_unsafe_unbacked_ops == []
    assert config.intra_node_bw is None
    assert config.inter_node_bw is None
    assert config.static_launch_user_defined_triton_kernels is False


def test_compile_threads_is_decided_on_first_use(monkeypatch):
    from tensorplay.compiler.backends.stax import async_compile

    monkeypatch.setattr(config, "compile_threads", None)
    threads = async_compile.get_compile_threads()
    assert threads >= 1
    assert config.compile_threads == threads


def test_bandwidth_settings_override_detection(monkeypatch):
    monkeypatch.setattr(config, "intra_node_bw", 123)
    monkeypatch.setattr(config, "inter_node_bw", 7)
    assert comm_analysis.get_intra_node_bw() == 123.0
    assert comm_analysis.get_inter_node_bw() == 7.0
    monkeypatch.setattr(config, "intra_node_bw", None)
    monkeypatch.setattr(config, "inter_node_bw", None)
    assert comm_analysis.get_intra_node_bw() > 0
    assert comm_analysis.get_inter_node_bw() > 0


def test_bucket_mode_comes_from_the_settings(monkeypatch):
    monkeypatch.setattr(
        config.tp_distributed_optimizations, "bucket_mode", "coalesced"
    )
    assert bucketing._default_bucket_mode() == "coalesced"
    monkeypatch.setattr(config.tp_distributed_optimizations, "bucket_mode", None)
    assert bucketing._default_bucket_mode() == "default"


def test_snapshot_writes_as_text_and_keeps_sections_apart():
    snapshot = config.save_config_portable()
    json.dumps(snapshot, sort_keys=True)

    assert snapshot["compile_threads"] in (None, config.compile_threads)
    assert "cudagraph_unsafe_unbacked_ops" in snapshot
    # A section keeps its own namespace, so two sections may share a name.
    assert "triton.cudagraphs" in snapshot
    assert "cuda" in snapshot or "cuda." in "".join(snapshot)


def test_snapshot_follows_a_setting(monkeypatch):
    before = config.save_config_portable()
    monkeypatch.setattr(config, "cudagraph_unsafe_unbacked_ops", ["lib::split"])
    after = config.save_config_portable()
    assert after["cudagraph_unsafe_unbacked_ops"] == ["lib::split"]
    assert before != after


def test_snapshot_leaves_private_names_out_unless_asked():
    public = config.save_config_portable()
    private = config.save_config_portable(ignore_private_configs=False)
    assert all(not name.startswith("_") for name in public)
    assert len(private) > len(public)


def test_snapshot_leaves_the_module_s_own_callables_out():
    snapshot = config.save_config_portable()
    # A helper defined here is surface, not a setting.
    assert "patch" not in snapshot
    assert "is_fbcode" not in snapshot
    assert "save_config_portable" not in snapshot


def test_isolation_key_composes_and_is_stable():
    expected = context._isolation_key()
    assert len(expected) == 32
    assert context._isolation_key() == expected
    forms = context._isolation_context()
    assert set(forms) == {"runtime_context", "compile_context"}
    assert "tp_configs" in forms["runtime_context"]


@pytest.mark.parametrize("selection", [True, False])
def test_isolation_key_respects_a_selection(selection):
    schema = {
        "runtime_context": selection,
        "compile_context": selection,
    }
    key = context._isolation_key(schema)
    assert len(key) == 32