"""What a generated launch records and how it reads its own fields.

A launch is written once and replayed many times, so what it records has to
say which settings it was written under, and how it finds the fields it needs
has to keep working when the runtime behind the kernel changes.
"""

import tensorplay as tp

from tensorplay.compiler.backends.stax import config
from tensorplay.compiler.backends.stax.codegen.triton import TritonKernel
from tensorplay.compiler.backends.stax.runtime.triton_heuristics import (
    _launch_field_schema,
    _schema_or_probed,
)


class _Binary:
    def __init__(self, schema=None, num_warps=4, shared=128):
        if schema is not None:
            self.launch_metadata_schema = schema
        self.num_warps = num_warps
        self.shared = shared


def test_the_record_says_whether_deterministic_arithmetic_was_asked_for(
    monkeypatch,
):
    """A reader filters configurations by this key, so it has to be written.

    Reduction configurations differ in how they order the same additions.
    When the run asked for deterministic arithmetic, the launch record has to
    carry that, because the configuration filter reads the record rather than
    the process.
    """

    monkeypatch.setattr(config, "write_are_deterministic_algorithms_enabled", True)
    monkeypatch.setattr(tp, "are_deterministic_algorithms_enabled", lambda: True)
    record = TritonKernel.tp_meta_common()
    assert record["are_deterministic_algorithms_enabled"] is True

    monkeypatch.setattr(tp, "are_deterministic_algorithms_enabled", lambda: False)
    assert TritonKernel.tp_meta_common()[
        "are_deterministic_algorithms_enabled"
    ] is False


def test_the_record_leaves_the_key_out_when_the_setting_is_off(monkeypatch):
    """With the setting off nothing is written, so nothing is filtered."""

    monkeypatch.setattr(config, "write_are_deterministic_algorithms_enabled", False)
    assert "are_deterministic_algorithms_enabled" not in TritonKernel.tp_meta_common()


def test_a_field_table_is_read_only_when_the_setting_asks_for_it(monkeypatch):
    """The table is the contract; probing is what is read without one."""

    binary = _Binary(schema={"num_warps": 8, "shared_mem": 4096}, num_warps=4, shared=128)

    monkeypatch.setattr(config, "use_launch_metadata_schema", True)
    schema = _launch_field_schema(binary)
    assert schema is not None
    assert _schema_or_probed(schema, "num_warps", lambda: binary.num_warps) == 8
    assert _schema_or_probed(schema, "shared_mem", lambda: binary.shared) == 4096

    monkeypatch.setattr(config, "use_launch_metadata_schema", False)
    assert _launch_field_schema(binary) is None
    assert _schema_or_probed(None, "num_warps", lambda: binary.num_warps) == 4


def test_a_field_the_table_does_not_list_is_probed(monkeypatch):
    """A partial table leaves the remaining fields to the old path."""

    binary = _Binary(schema={"num_warps": 8}, num_warps=4, shared=128)
    monkeypatch.setattr(config, "use_launch_metadata_schema", True)
    schema = _launch_field_schema(binary)
    assert _schema_or_probed(schema, "shared_mem", lambda: binary.shared) == 128


def test_a_binary_without_a_table_is_probed(monkeypatch):
    monkeypatch.setattr(config, "use_launch_metadata_schema", True)
    binary = _Binary(num_warps=4)
    assert _launch_field_schema(binary) is None