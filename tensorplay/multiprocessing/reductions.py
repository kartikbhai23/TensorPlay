# mypy: allow-untyped-defs
"""Serialization reducers that move tensors between processes through shared memory.

Registering these reducers with the multiprocessing pickling machinery makes
every tensor sent through a queue, pipe, or connection hand over a shared
segment instead of a byte copy: the sender moves its data into memory visible
to other processes (either a named segment or a passed file descriptor), and
each recipient rebuilds a tensor aliasing those same pages.

Two sharing strategies are supported:

- ``file_system``: segments are identified by their name in the shared-memory
  filesystem, so any process that knows the name can attach to them.
- ``file_descriptor``: segments are anonymous ``memfd`` regions passed as
  duplicated descriptors over the connection, leaving no named entries behind.
"""

import mmap
import os
import threading
from multiprocessing import reduction
from multiprocessing.resource_tracker import unregister as _tracker_unregister
from multiprocessing.shared_memory import SharedMemory

import tensorplay
from tensorplay._C import DType

from ._atfork import register_after_fork

# Early load resource_sharer to prevent a partially initialized instance from
# being inherited in a forked child process. The descriptor-based reducer needs
# this module indirectly through DupFd(), and the built-in mp.Queue pickles
# arguments in a background thread which may overlap with the fork.
try:
    import multiprocessing.resource_sharer  # noqa: F401
except ImportError:
    pass

__all__ = ["init_reductions"]

_MEMFD_PREFIX = "tensorplay-mp-segment-"

# Tensors at or below this many bytes travel inline with the pickled message
# instead of through a shared segment. Establishing a segment and exchanging a
# descriptor costs on the order of a hundred microseconds per message, far more
# than copying this much data: measured across a queue link, inline transfer is
# 2-4x faster up to this size and loses beyond 1 MiB.
_INLINE_TENSOR_BYTES = 1 << 19


class TensorWeakRef:
    r"""A cache entry for a tensor recently handed to another process.

    The reference is intentionally strong: it anchors the Python object that
    keeps a named segment alive (the tensor and its shared-memory owner) for as
    long as the cache holds the entry, so a recipient that attaches later still
    finds the segment registered. ``nbytes`` records how much memory the entry
    keeps alive, letting the cache bound itself by size and not only by count.
    """

    __slots__ = ["tensor", "nbytes"]

    def __init__(self, tensor, nbytes=0):
        self.tensor = tensor
        self.nbytes = nbytes


class SharedCache(dict):
    """Dictionary from segment names to TensorWeakRef objects.

    Entries keep the sender-side tensor alive until another process attaches to
    the segment, so the cache is bounded by an entry count and by a byte budget:
    a handful of large tensors must not be retained as long as a hundred small
    ones would be.
    """

    def __init__(self) -> None:
        self.limit = 128
        self.byte_limit = 1 << 28
        self.nbytes = 0
        # `fork` inherits lock state, so in case we fork while the lock is
        # held, reset the lock in the child to avoid a deadlock. Registering
        # through os.register_at_fork also covers children created by a bare
        # os.fork(), which the multiprocessing-only hook would miss.
        self._after_fork()
        register_after_fork(self._after_fork)

    def _after_fork(self):
        self.lock = threading.Lock()

    def get(self, key):  # type: ignore[override]
        with self.lock:
            return dict.get(self, key)

    def __setitem__(self, key, storage_ref):
        with self.lock:
            previous = dict.get(self, key)
            if previous is not None:
                self.nbytes -= previous.nbytes
            dict.__setitem__(self, key, storage_ref)
            self.nbytes += storage_ref.nbytes
            if len(self) > self.limit or self.nbytes > self.byte_limit:
                self._evict_oldest(key)

    def _evict_oldest(self, keep):
        # Entries hold strong references, so they only leave the cache when
        # evicted here. Dropping an entry is safe: the segment name stays valid
        # while the resource tracker keeps it registered for this process, so a
        # recipient attaching afterwards still sees the same live pages.
        for oldest in list(self.keys()):
            if len(self) <= self.limit and self.nbytes <= self.byte_limit:
                break
            if oldest == keep:
                continue
            self.nbytes -= dict.pop(self, oldest).nbytes


# mapping from segment names to TensorWeakRef objects
shared_cache = SharedCache()


def rebuild_event(device, handle):
    return tensorplay.cuda.Event.from_ipc_handle(device, handle)


def reduce_event(event):
    handle = event.ipc_handle()
    return (rebuild_event, (event.device, handle))


def rebuild_empty_tensor(cls, shape, dtype, requires_grad):
    # Zero-element tensors cannot back a shared segment (there are no bytes to
    # map), so they are rebuilt from metadata alone. Strides carry no
    # information without elements.
    t = tensorplay.empty(shape, dtype=dtype)
    t.requires_grad = requires_grad
    return t


def rebuild_shared_tensor(
    cls, name, shape, strides, dtype, device_type, device_index, requires_grad
):
    # Attach to the named segment and rebuild a tensor aliasing its pages.
    try:
        shm = SharedMemory(name=name, create=False, track=False)
    except TypeError:
        # Runtimes without per-instance tracking control register the segment
        # with the resource tracker on attach. Drop that registration again:
        # this process must not unlink a segment its sender still owns.
        shm = SharedMemory(name=name, create=False)
        try:
            _tracker_unregister(shm._name, "shared_memory")
        except (ValueError, KeyError):
            pass
    tensor = cls.__new__(cls)
    tensor.__setstate__(
        ("shm", shm, shape, strides, dtype, device_type, device_index, requires_grad)
    )
    return tensor


def rebuild_tensor_from_state(cls, state):
    # Rebuild from the byte payload carried by the pickle itself.
    tensor = cls.__new__(cls)
    tensor.__setstate__(state)
    return tensor


def rebuild_shared_tensor_fd(cls, df, nbytes, shape, dtype, requires_grad):
    # Map the received descriptor and rebuild a tensor aliasing the same
    # pages the sender filled. The mapping keeps the segment alive after the
    # descriptor is closed.
    fd = df.detach()
    try:
        view = mmap.mmap(fd, nbytes)
    finally:
        os.close(fd)
    tensor = tensorplay.frombuffer(view, dtype=DType(dtype))
    tensor = tensor.reshape(shape)
    tensor.requires_grad = requires_grad
    return tensor


def _reduce_tensor_shared(tensor):
    if not tensor.is_shared():
        if tensor.is_contiguous() and tensor.storage_offset() == 0:
            # Move this tensor's storage into a named shared segment in
            # place; the sending process keeps reading and writing the very
            # same pages the recipients will attach to.
            tensor.share_memory_()
        else:
            # Only contiguous, unoffset storage can be moved in place; share
            # a compacted copy of the data instead.
            compact = tensor.contiguous()
            if compact.storage_offset() != 0:
                compact = compact.clone()
            compact.share_memory_()
            tensor = compact
    state = tensor.__getstate__()
    shm = state[1]
    shared_cache[shm.name] = TensorWeakRef(tensor, shm.size)
    return (
        rebuild_shared_tensor,
        (type(tensor), shm.name) + tuple(state[2:]),
    )


def _reduce_tensor_inline(tensor):
    # Ship the values with the message. The recipient rebuilds a private copy
    # from the byte payload, which is what the descriptor path hands over too
    # (it copies into a private segment before duplicating the descriptor).
    return (rebuild_tensor_from_state, (type(tensor), tensor.__getstate__()))


def _reduce_tensor_fd(tensor):
    # Hand the segment over as a duplicated descriptor. The sender fills a
    # private memfd-backed region; the recipient maps the received descriptor
    # and rebuilds a tensor aliasing those pages. The sending tensor itself
    # keeps its original (private) storage.
    shape = tuple(tensor.shape)
    nbytes = tensor.numel() * tensor.itemsize()
    fd = os.memfd_create(_MEMFD_PREFIX + str(os.getpid()), flags=os.MFD_CLOEXEC)
    try:
        os.ftruncate(fd, nbytes)
        view = mmap.mmap(fd, nbytes)
        host = tensorplay.frombuffer(view, dtype=tensor.dtype)
        host = host.reshape(shape)
        host.copy_(tensor)
    except BaseException:
        os.close(fd)
        raise
    # DupFd duplicates the descriptor while it is being constructed, and the
    # duplicate holds the segment pages until the recipient detaches it.
    # Neither this descriptor nor the mapping has to outlive the call: the page
    # cache keeps the contents for the recipient to map.
    try:
        df = reduction.DupFd(fd)
    finally:
        os.close(fd)
    return (
        rebuild_shared_tensor_fd,
        (
            type(tensor),
            df,
            nbytes,
            shape,
            int(tensor.dtype),
            bool(tensor.requires_grad),
        ),
    )


def reduce_tensor(tensor):
    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError(
            "Cowardly refusing to serialize non-leaf tensor which requires_grad, "
            "since autograd does not support crossing process boundaries.  "
            "If you just want to transfer the data, call detach() on the tensor "
            "before serializing (e.g., putting it on the queue)."
        )

    if hasattr(tensor, "_backward_hooks"):
        from tensorplay.utils.hooks import warn_if_has_hooks

        warn_if_has_hooks(tensor)

    if tensor.device.type != "cpu":
        raise RuntimeError(
            f"Cannot pickle {tensor.device.type} tensor: only CPU tensors "
            "support cross-process shared memory"
        )

    if tensor.numel() == 0:
        return (
            rebuild_empty_tensor,
            (
                type(tensor),
                tuple(tensor.shape),
                tensor.dtype,
                bool(tensor.requires_grad),
            ),
        )

    from . import get_sharing_strategy

    if get_sharing_strategy() == "file_system" or tensor.is_shared():
        return _reduce_tensor_shared(tensor)
    if tensor.numel() * tensor.itemsize() <= _INLINE_TENSOR_BYTES:
        return _reduce_tensor_inline(tensor)
    return _reduce_tensor_fd(tensor)


def init_reductions():
    reduction.register(tensorplay.Tensor, reduce_tensor)

    try:
        from tensorplay._C import TensorBase

        if TensorBase is not tensorplay.Tensor:
            reduction.register(TensorBase, reduce_tensor)
    except ImportError:
        pass

    try:
        from tensorplay.nn.parameter import Parameter

        reduction.register(Parameter, reduce_tensor)
    except ImportError:
        pass

    try:
        cuda_module = getattr(tensorplay, "cuda", None)
        event_cls = getattr(cuda_module, "Event", None)
        if event_cls is not None and hasattr(event_cls, "ipc_handle"):
            reduction.register(event_cls, reduce_event)
    except ImportError:
        pass
