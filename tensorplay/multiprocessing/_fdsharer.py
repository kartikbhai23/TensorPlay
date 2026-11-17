"""Descriptor sharing between processes over persistent local connections.

Handing a file descriptor to another process requires an out-of-band channel:
the descriptor travels as ``SCM_RIGHTS`` ancillary data on a local socket. A
one-shot channel pays for a fresh connection and the authentication handshake
of every message, which costs on the order of two hundred microseconds and
dwarfs the descriptor transfer itself (a single request/reply round trip costs
a few microseconds).

This module keeps one channel per peer: the sending process publishes a
``(address, key)`` handle with the pickled message, and the receiving process
connects to that address once and reuses the connection for every later
transfer to the same peer. Descriptors offered for transfer are duplicated at
registration time, so the sender's own descriptor can be closed immediately.

The listener lives in the Linux abstract socket namespace, so it is released by
the kernel when the owning process exits and leaves no socket file behind.
"""

import os
import signal
import struct
import sys
import threading
import uuid

from multiprocessing import process as _mp_process
from multiprocessing.connection import Client, Listener
from multiprocessing.reduction import recv_handle, send_handle

__all__ = ["SharedFd"]

# Request frame: transfer key followed by the receiving process id. A fixed
# binary frame keeps the request byte-for-byte on the wire, so no pickling
# machinery runs on either side of the hand-off.
_REQUEST = struct.Struct("=Ii")

# Registry of descriptors offered to peers, keyed by the handle published in the
# message. Each entry holds the duplicated descriptor until a peer takes it.
_pending = {}
# Connections to peers, keyed by their listener address.
_clients = {}
_lock = threading.Lock()
# Serialises request/reply pairs, since a connection is shared by every thread
# of the receiving process.
_transfer_lock = threading.Lock()
_listener = None
_address = None
_key = 0

if sys.platform == "linux":
    os.register_at_fork(after_in_child=lambda: _after_fork())


def _peer_authkey():
    return _mp_process.current_process().authkey


def _new_address():
    return "\0tensorplay-fd-sharer-" + str(os.getpid()) + "-" + uuid.uuid4().hex


def _ensure_listener():
    """Create this process's listener on first use."""
    global _listener, _address
    if _listener is not None:
        return
    _listener = Listener(
        _new_address(), authkey=_peer_authkey(), backlog=128
    )
    _address = _listener.address
    thread = threading.Thread(target=_accept_loop, daemon=True)
    thread.start()


def _accept_loop():
    # Signals belong to the thread that can handle them; keeping them off this
    # background thread leaves KeyboardInterrupt delivery to the main thread.
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
    while True:
        try:
            conn = _listener.accept()
        except (OSError, EOFError):
            return
        thread = threading.Thread(target=_serve_connection, args=(conn,), daemon=True)
        thread.start()


def _serve_connection(conn):
    """Answer transfer requests on one peer connection until it closes."""
    try:
        while True:
            request = conn.recv_bytes()
            if not request:
                return
            key, destination_pid = _REQUEST.unpack(request)
            entry = _pending.pop(key, None)
            if entry is None:
                return
            send, close = entry
            try:
                send(conn, destination_pid)
            finally:
                close()
    except (EOFError, OSError, ValueError, struct.error):
        # The peer closed the connection or it broke mid-request.
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def register(fd):
    """Duplicate *fd* and return the handle a peer uses to take the duplicate."""
    new_fd = os.dup(fd)

    def send(conn, destination_pid):
        send_handle(conn, new_fd, destination_pid)

    def close():
        try:
            os.close(new_fd)
        except OSError:
            pass

    global _key
    with _lock:
        _ensure_listener()
        _key += 1
        _pending[_key] = (send, close)
        return (_address, _key)


def _connection(address):
    """Return the cached connection to *address*, connecting on first use."""
    with _lock:
        conn = _clients.get(address)
    if conn is not None:
        return conn
    conn = Client(address, authkey=_peer_authkey())
    with _lock:
        existing = _clients.setdefault(address, conn)
        if existing is not conn:
            conn.close()
            return existing
    return conn


def _drop_connection(address, conn):
    with _lock:
        if _clients.get(address) is conn:
            del _clients[address]
    try:
        conn.close()
    except OSError:
        pass


def detach(handle):
    """Take the descriptor identified by *handle* from its owning process."""
    address, key = handle
    request = _REQUEST.pack(key, os.getpid())
    with _transfer_lock:
        conn = _connection(address)
        try:
            conn.send_bytes(request)
        except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
            # The cached connection went stale before the request left; asking
            # again on a fresh connection is safe because nothing was consumed.
            _drop_connection(address, conn)
            conn = _connection(address)
            conn.send_bytes(request)
        try:
            return recv_handle(conn)
        except BaseException:
            _drop_connection(address, conn)
            raise


class SharedFd:
    """Picklable wrapper around a descriptor offered to another process.

    The wrapper itself only carries the handle; ``detach`` retrieves the
    descriptor and may be called once.
    """

    def __init__(self, fd):
        self._id = register(fd)

    def detach(self):
        ident, self._id = self._id, None
        if ident is None:
            raise RuntimeError("descriptor handle already detached")
        return detach(ident)


def _after_fork():
    """Drop state inherited from the parent; it belongs to the parent's peers."""
    global _listener, _address, _key, _lock, _transfer_lock
    for _send, close in _pending.values():
        close()
    _pending.clear()
    for conn in _clients.values():
        try:
            conn.close()
        except OSError:
            pass
    _clients.clear()
    if _listener is not None:
        try:
            _listener.close()
        except OSError:
            pass
    _listener = None
    _address = None
    _key = 0
    # A lock inherited while another thread held it can never be released.
    _lock = threading.Lock()
    _transfer_lock = threading.Lock()