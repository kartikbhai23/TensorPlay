"""Weight download plumbing for the vision package.

The vision cache lives under ``TENSORPLAY_HOME`` (or
``~/.cache/tensorplay/datasets/vision``) so model weights stay separate from the
generic hub checkpoints. The transfer itself is delegated to
``tensorplay.hub``: staged writes, resume, retries, digest verification and
locking are implemented once, there.
"""

import os
from pathlib import Path

from tensorplay import hub

__all__ = ["load_state_dict_from_url"]

_HOME = Path.home() / ".cache" / "tensorplay" / "datasets" / "vision"


def _get_tensorplay_home() -> str:
    return os.environ.get("TENSORPLAY_HOME", str(_HOME))


def load_state_dict_from_url(
    url: str,
    model_dir: str | None = None,
    map_location=None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: str | None = None,
    weights_only: bool = True,
) -> dict:
    r"""Loads the Tensor serialized at ``url`` into a state dict.

    A cached file is reused only while it still matches the digest recorded
    when it was downloaded; otherwise it is fetched again. When ``check_hash``
    is True the file name must embed a ``<sha256>-<filename>`` prefix and the
    downloaded bytes are verified against it.

    ``weights_only=True`` is the default: checkpoint loading is always
    restricted to registered data types, and passing False only opts into a
    warning that says so.
    """
    if model_dir is None:
        model_dir = _get_tensorplay_home()
    return hub.load_state_dict_from_url(
        url,
        model_dir=model_dir,
        map_location=map_location,
        progress=progress,
        check_hash=check_hash,
        file_name=file_name,
        weights_only=weights_only,
    )