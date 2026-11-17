"""Shared setup for the stax tests.

Everything built while a test runs -- printed regions, kernels, autotuned
candidates -- is kept under one directory.  Left at its default it is the
working tree's own directory, which is shared by every run and by everyone
working in the tree, so a second run of a test can find the work its first run
did and never do it.  Tests that observe compilation would then observe a
cache hit instead, and tests that expect a build would pass or fail depending
on what ran before them.

Each session therefore points the whole cache at a directory of its own, and
leaves the whole-region artifact cache off.  A test that wants to pin the cache
somewhere specific still can: setting the directory itself takes precedence for
as long as it is set.  A test that is about the artifact cache turns it back on
for itself.
"""

import pytest

from tensorplay.compiler.backends.stax import config
from tensorplay.compiler.backends.stax.runtime.cache_dir_utils import (
    temporary_cache_dir,
)


@pytest.fixture(autouse=True, scope="session")
def _session_cache_directory(tmp_path_factory):
    with temporary_cache_dir(str(tmp_path_factory.mktemp("stax-cache"))):
        # A finished region is an answer without a build; tests that watch a
        # region being built need the build to happen.
        with config.patch({"cache_graph_artifacts": False}):
            yield
