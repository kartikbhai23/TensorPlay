import multiprocessing
import os
import unittest

import tensorplay as tp
import tensorplay.multiprocessing as tmp


def _fs_child(q_in, q_out):
    t = q_in.get()
    assert t.is_shared()
    t[0] = 42.0
    q_out.put("ok")


def _fd_child(q_in, q_out, ack):
    t = q_in.get()
    t[1] = 7.0
    q_out.put(t)
    # Stay alive until the receiver detaches the duplicated descriptor: the
    # resource sharer thread of the sending process serves the fd transfer.
    ack.get(timeout=120)


def _fd_echo_child(q_in, q_out, rounds):
    for _ in range(rounds):
        t = q_in.get()
        q_out.put(float(t[0] + t[-1]))


def _reader_child(q_in, q_out):
    t = q_in.get()
    q_out.put(float(t[0]))


def _reader_empty_child(q_in, q_out):
    t = q_in.get()
    assert t.numel() == 0
    q_out.put("ok")


def _parameter_child(q_in, q_out):
    p = q_in.get()
    p[0] = 10.0
    q_out.put(p)


def _spawn_ok(index):
    with open(f"/tmp/tp_mp_spawn_ok_{index}", "w") as fh:
        fh.write("ok")


def _spawn_fail(index):
    raise ValueError(f"boom {index}")


def _pool_square(t):
    return t * t


class TestSharingStrategy(unittest.TestCase):
    def setUp(self):
        self._old = tmp.get_sharing_strategy()

    def tearDown(self):
        tmp.set_sharing_strategy(self._old)

    def test_get_set(self):
        for strategy in tmp.get_all_sharing_strategies():
            tmp.set_sharing_strategy(strategy)
            self.assertEqual(tmp.get_sharing_strategy(), strategy)

    def test_invalid(self):
        with self.assertRaises(AssertionError):
            tmp.set_sharing_strategy("bogus")

    def test_strategies_present(self):
        strategies = tmp.get_all_sharing_strategies()
        self.assertIn("file_system", strategies)


class TestQueueSharing(unittest.TestCase):
    def setUp(self):
        self._old = tmp.get_sharing_strategy()
        self.ctx = multiprocessing.get_context("fork")

    def tearDown(self):
        tmp.set_sharing_strategy(self._old)

    def _run_pair(self, target, args=()):
        q_in, q_out = self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=target, args=(q_in, q_out) + args)
        p.start()
        return p, q_in, q_out

    def test_file_system_zero_copy(self):
        tmp.set_sharing_strategy("file_system")
        t = tp.zeros(3)
        p, q_in, q_out = self._run_pair(_fs_child)
        try:
            q_in.put(t)
            self.assertEqual(q_out.get(timeout=60), "ok")
            # The sender moved its storage into the shared segment, so the
            # write performed by the child is visible here without any copy.
            self.assertEqual(float(t[0]), 42.0)
        finally:
            p.join(timeout=60)
        self.assertEqual(p.exitcode, 0)

    def test_segment_survives_child_exit(self):
        tmp.set_sharing_strategy("file_system")
        t = tp.zeros(2)
        t[0] = 5.0
        seen = []
        for _ in range(2):
            p, q_in, q_out = self._run_pair(_reader_child)
            try:
                q_in.put(t)
                seen.append(q_out.get(timeout=60))
            finally:
                p.join(timeout=60)
            self.assertEqual(p.exitcode, 0)
        # A child must not unlink the segment when it exits, otherwise the
        # second child could no longer attach to it.
        self.assertEqual(seen, [5.0, 5.0])

    def test_file_descriptor_roundtrip(self):
        tmp.set_sharing_strategy("file_descriptor")
        t = tp.arange(0, 4, dtype=tp.float32)
        q_in, q_out, ack = self.ctx.Queue(), self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=_fd_child, args=(q_in, q_out, ack))
        p.start()
        try:
            q_in.put(t)
            back = q_out.get(timeout=60)
            self.assertEqual(tuple(back.shape), (4,))
            self.assertEqual(float(back[1]), 7.0)
        finally:
            ack.put("done")
            p.join(timeout=60)
        self.assertEqual(p.exitcode, 0)
        # Under this strategy the sending tensor keeps its private storage.
        self.assertEqual(float(t[1]), 1.0)

    def test_file_descriptor_above_inline_limit(self):
        # Payloads over the inline limit still travel through a duplicated
        # descriptor; the mapped values must match the sender's byte for byte.
        tmp.set_sharing_strategy("file_descriptor")
        t = tp.arange(0, 300000, dtype=tp.float32)  # 1.2 MiB
        q_in, q_out, ack = self.ctx.Queue(), self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=_fd_child, args=(q_in, q_out, ack))
        p.start()
        try:
            q_in.put(t)
            back = q_out.get(timeout=120)
            self.assertEqual(tuple(back.shape), tuple(t.shape))
            self.assertEqual(float(back[1]), 7.0)
            self.assertEqual(float(back[-1]), float(t[-1]))
        finally:
            ack.put("done")
            p.join(timeout=120)
        self.assertEqual(p.exitcode, 0)
        self.assertEqual(float(t[1]), 1.0)

    def test_file_descriptor_keeps_no_sender_copy(self):
        # The descriptor is duplicated when it is handed over, so the sender's
        # private copy must not be retained by the segment cache.
        tmp.set_sharing_strategy("file_descriptor")
        from tensorplay.multiprocessing.reductions import shared_cache

        before = len(shared_cache)
        t = tp.zeros(300000)  # above the inline limit: descriptor path
        p, q_in, q_out = self._run_pair(_reader_child)
        try:
            q_in.put(t)
            self.assertEqual(q_out.get(timeout=60), 0.0)
        finally:
            p.join(timeout=60)
        self.assertEqual(p.exitcode, 0)
        self.assertEqual(len(shared_cache), before)

    def test_file_descriptor_reuses_peer_connection(self):
        # Every transfer after the first runs on the connection already open to
        # the sender, so a long-lived pair pays the connect and handshake once.
        tmp.set_sharing_strategy("file_descriptor")
        rounds = 20
        t = tp.arange(0, 300000, dtype=tp.float32)  # 1.2 MiB
        q_in, q_out = self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=_fd_echo_child, args=(q_in, q_out, rounds))
        p.start()
        try:
            for _ in range(rounds):
                q_in.put(t)
                self.assertEqual(q_out.get(timeout=120), float(t[0] + t[-1]))
        finally:
            p.join(timeout=120)
        self.assertEqual(p.exitcode, 0)

    def test_file_descriptor_two_receivers(self):
        # Two peers keep their own connection to the same sender open at the
        # same time, so the sender must serve them concurrently.
        tmp.set_sharing_strategy("file_descriptor")
        t = tp.full((300000,), 3.0)
        procs, queues = [], []
        for _ in range(2):
            q_in, q_out = self.ctx.Queue(), self.ctx.Queue()
            p = self.ctx.Process(target=_reader_child, args=(q_in, q_out))
            p.start()
            procs.append(p)
            queues.append((q_in, q_out))
        try:
            for q_in, _ in queues:
                q_in.put(t)
            for _, q_out in queues:
                self.assertEqual(q_out.get(timeout=120), 3.0)
        finally:
            for p in procs:
                p.join(timeout=120)
        for p in procs:
            self.assertEqual(p.exitcode, 0)

    def _roundtrip_through_child(self, t):
        q_in, q_out, ack = self.ctx.Queue(), self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=_fd_child, args=(q_in, q_out, ack))
        p.start()
        try:
            q_in.put(t)
            back = q_out.get(timeout=120)
        finally:
            ack.put("done")
            p.join(timeout=120)
        self.assertEqual(p.exitcode, 0)
        return back

    def test_file_descriptor_strided_payload(self):
        # A strided view cannot be copied out as one contiguous run, so the
        # sender compacts it before filling the segment.
        tmp.set_sharing_strategy("file_descriptor")
        t = tp.arange(0, 600000, dtype=tp.float32)[::2]  # 1.2 MiB strided view
        back = self._roundtrip_through_child(t)
        self.assertEqual(tuple(back.shape), tuple(t.shape))
        self.assertEqual(float(back[1]), 7.0)
        self.assertEqual(float(back[-1]), float(t[-1]))
        self.assertEqual(float(t[1]), 2.0)

    def test_file_descriptor_dtype_without_byte_view(self):
        # Dtypes that have no element-mapped array view still transfer through
        # the mapping fallback.
        tmp.set_sharing_strategy("file_descriptor")
        t = tp.zeros(300000, dtype=tp.bfloat16)  # 600 KiB
        back = self._roundtrip_through_child(t)
        self.assertEqual(back.dtype, t.dtype)
        self.assertEqual(float(back[1]), 7.0)

    def test_empty_tensor(self):
        t = tp.empty(0)
        q_in, q_out = self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=_reader_empty_child, args=(q_in, q_out))
        p.start()
        try:
            q_in.put(t)
            self.assertEqual(q_out.get(timeout=60), "ok")
        finally:
            p.join(timeout=60)
        self.assertEqual(p.exitcode, 0)

    def test_parameter_roundtrip(self):
        from tensorplay.nn.parameter import Parameter

        tmp.set_sharing_strategy("file_system")
        q_in, q_out = self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=_parameter_child, args=(q_in, q_out))
        p.start()
        try:
            q_in.put(Parameter(tp.arange(0, 3, dtype=tp.float32)))
            back = q_out.get(timeout=60)
        finally:
            p.join(timeout=60)
        self.assertEqual(p.exitcode, 0)
        self.assertIsInstance(back, Parameter)
        self.assertEqual(back.tolist(), [10.0, 1.0, 2.0])

    def test_non_leaf_requires_grad_rejected(self):
        from multiprocessing.reduction import ForkingPickler

        x = tp.zeros(2, requires_grad=True)
        y = x * 2
        # Queue.put pickles in a background feeder thread, so the rejection is
        # asserted directly on the pickling path used by every queue/pipe.
        with self.assertRaises(RuntimeError):
            ForkingPickler.dumps(y)


class TestSpawnMethod(unittest.TestCase):
    def setUp(self):
        self._old = tmp.get_sharing_strategy()
        tmp.set_sharing_strategy("file_system")
        self.ctx = multiprocessing.get_context("spawn")

    def tearDown(self):
        tmp.set_sharing_strategy(self._old)

    def test_spawn_zero_copy(self):
        t = tp.zeros(2)
        q_in, q_out = self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=_fs_child, args=(q_in, q_out))
        p.start()
        try:
            q_in.put(t)
            self.assertEqual(q_out.get(timeout=120), "ok")
            self.assertEqual(float(t[0]), 42.0)
        finally:
            p.join(timeout=120)
        self.assertEqual(p.exitcode, 0)

    def test_spawn_file_descriptor(self):
        tmp.set_sharing_strategy("file_descriptor")
        t = tp.full((2, 3), 1.5)
        q_in, q_out, ack = self.ctx.Queue(), self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=_fd_child, args=(q_in, q_out, ack))
        p.start()
        try:
            q_in.put(t)
            back = q_out.get(timeout=120)
            self.assertEqual(tuple(back.shape), (2, 3))
            # The child assigned row 1 of the shared region.
            self.assertEqual(float(back[1][0]), 7.0)
        finally:
            ack.put("done")
            p.join(timeout=120)
        self.assertEqual(p.exitcode, 0)

    def test_spawn_file_descriptor_above_inline_limit(self):
        # Large payloads take the descriptor path in both directions, so the
        # spawned child has to authenticate against the parent's sharer and
        # then publish a listener of its own for the reply.
        tmp.set_sharing_strategy("file_descriptor")
        t = tp.arange(0, 300000, dtype=tp.float32)  # 1.2 MiB
        q_in, q_out, ack = self.ctx.Queue(), self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=_fd_child, args=(q_in, q_out, ack))
        p.start()
        try:
            q_in.put(t)
            back = q_out.get(timeout=120)
            self.assertEqual(float(back[1]), 7.0)
            self.assertEqual(float(back[-1]), float(t[-1]))
        finally:
            ack.put("done")
            p.join(timeout=120)
        self.assertEqual(p.exitcode, 0)


class TestStartProcesses(unittest.TestCase):
    def test_success(self):
        tmp.start_processes(_spawn_ok, args=(), nprocs=2, join=True)

    def test_spawn_parallel_start(self):
        old = os.environ.get(tmp.ENV_VAR_PARALLEL_START)
        os.environ[tmp.ENV_VAR_PARALLEL_START] = "1"
        try:
            tmp.start_processes(_spawn_ok, args=(), nprocs=2, join=True)
        finally:
            if old is None:
                del os.environ[tmp.ENV_VAR_PARALLEL_START]
            else:
                os.environ[tmp.ENV_VAR_PARALLEL_START] = old

    def test_exception_forwarded(self):
        with self.assertRaises(tmp.ProcessRaisedException) as cm:
            tmp.start_processes(_spawn_fail, args=(), nprocs=1, join=True)
        self.assertIn("boom 0", str(cm.exception))

    def test_no_join_returns_context(self):
        context = tmp.start_processes(
            _spawn_ok, args=(), nprocs=1, join=False, start_method="fork"
        )
        try:
            self.assertIsInstance(context, tmp.ProcessContext)
            while not context.join(timeout=60):
                pass
        finally:
            for process in context.processes:
                process.join(timeout=60)


class TestPool(unittest.TestCase):
    def test_pool_map(self):
        tmp.set_sharing_strategy("file_system")
        pool = tmp.Pool(2)
        try:
            results = pool.map(
                _pool_square, [tp.tensor([1.0, 2.0]), tp.tensor([3.0, 4.0])]
            )
        finally:
            pool.close()
            pool.join()
        self.assertEqual(results[0].tolist(), [1.0, 4.0])
        self.assertEqual(results[1].tolist(), [9.0, 16.0])


if __name__ == "__main__":
    unittest.main()
