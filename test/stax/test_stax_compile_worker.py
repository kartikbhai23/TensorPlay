"""The compile-worker sidecar's own pieces: timers, heartbeats and process role.

These run around every long compile, so they are tested where they can be
observed directly rather than through a pool: the timer's quiet-period
behaviour, the slot lifecycle the sidecar reads, and the flags that say whether
this process may start workers.
"""

import os
import time

import pytest

from tensorplay.compiler.backends.stax.compile_worker import utils
from tensorplay.compiler.backends.stax.compile_worker import watchdog
from tensorplay.compiler.backends.stax.compile_worker.timer import Timer


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_timer_calls_once_after_the_quiet_period():
    calls = []
    timer = Timer(0.05, lambda: calls.append(time.monotonic()))
    start = time.monotonic()

    timer.record_call()
    assert calls == []
    assert _wait_until(lambda: calls), "the timer never fired"
    assert len(calls) == 1
    assert calls[0] >= start

    # A fired timer is done: it does not fire again on its own.
    time.sleep(0.15)
    assert len(calls) == 1


def test_timer_restarts_its_quiet_period_on_each_call():
    calls = []
    timer = Timer(0.2, lambda: calls.append(time.monotonic()))

    timer.record_call()
    time.sleep(0.1)
    timer.record_call()
    first = time.monotonic()
    assert _wait_until(lambda: calls, timeout=2.0)
    # The second event reset the clock, so the call happened after it.
    assert calls[0] >= first


def test_timer_quit_cancels_a_pending_call():
    calls = []
    timer = Timer(0.1, lambda: calls.append("fired"))

    timer.record_call()
    timer.quit()
    time.sleep(0.3)
    assert calls == []


@pytest.fixture
def clean_heartbeat():
    saved = (
        watchdog._heartbeat,
        watchdog._counter,
        watchdog._nprocs,
        watchdog._slot,
    )
    yield
    (
        watchdog._heartbeat,
        watchdog._counter,
        watchdog._nprocs,
        watchdog._slot,
    ) = saved


def test_heartbeat_is_inactive_until_the_sidecar_allocates_it(clean_heartbeat):
    watchdog._heartbeat = None
    watchdog._counter = None
    watchdog._slot = None

    assert watchdog.enabled() is False
    assert watchdog.read_heartbeats() == {}
    # Every worker-side call is a no-op without a buffer.
    watchdog.init_worker_slot()
    watchdog.set_current_job(1)
    watchdog.report_phase(watchdog.Phase.COMPILING)
    watchdog.clear_current_job()
    assert watchdog.read_heartbeats() == {}


def test_heartbeat_reports_the_phase_a_busy_slot_is_in(clean_heartbeat):
    watchdog._heartbeat = None
    watchdog._counter = None
    watchdog._slot = None
    watchdog.create(2)
    assert watchdog.enabled() is True

    watchdog.init_worker_slot()
    watchdog.set_current_job(7)
    watchdog.report_phase(watchdog.Phase.QUERYING_CACHE)
    busy = watchdog.read_heartbeats()
    assert set(busy) == {7}
    phase, started, pid = busy[7]
    assert phase == watchdog.Phase.QUERYING_CACHE
    assert pid == os.getpid()
    assert started <= time.monotonic_ns()

    watchdog.report_phase(watchdog.Phase.COMPILING)
    assert watchdog.read_heartbeats()[7][0] == watchdog.Phase.COMPILING

    watchdog.clear_current_job()
    assert watchdog.read_heartbeats() == {}


def test_heartbeat_slots_are_recycled_by_reset(clean_heartbeat):
    watchdog._heartbeat = None
    watchdog._counter = None
    watchdog._slot = None
    watchdog.create(1)

    watchdog.init_worker_slot()
    watchdog.set_current_job(3)
    assert set(watchdog.read_heartbeats()) == {3}

    # A new worker generation reuses slot 0 rather than colliding with it.
    watchdog.reset()
    assert watchdog.read_heartbeats() == {}
    watchdog.init_worker_slot()
    watchdog.set_current_job(4)
    assert set(watchdog.read_heartbeats()) == {4}


def test_utils_reports_the_process_role(monkeypatch):
    monkeypatch.setattr(utils, "_IN_TOPLEVEL_PROCESS", True)
    assert utils.in_toplevel_process() is True
    monkeypatch.setattr(utils, "_IN_TOPLEVEL_PROCESS", False)
    assert utils.in_toplevel_process() is False


def test_utils_detects_a_reassigned_parent(monkeypatch):
    monkeypatch.setattr(utils, "_original_parent", os.getppid())
    assert utils.has_parent_changed() is False
    monkeypatch.setattr(utils, "_original_parent", os.getppid() + 1)
    assert utils.has_parent_changed() is True