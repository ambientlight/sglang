"""CPU-only regressions for the 2026-09-20 strict-UTF8 log-pipe stall."""

import collections
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
import unittest


spec = importlib.util.spec_from_file_location('logging_boot', Path(__file__).with_name('boot.py'))
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)


class Fragments:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def read(self, size):
        return next(self.chunks, b'')


class FlushedBytes(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.flushed = threading.Event()

    def flush(self):
        super().flush()
        self.flushed.set()


class LogByteTests(unittest.TestCase):
    def forward(self, chunks, secret='fixture-key'):
        sink = io.BytesIO()
        boot.forward_log_bytes(Fragments(chunks), sink, secret)
        return sink.getvalue()

    def test_unicode_invalid_bytes_and_binary_content_are_preserved(self):
        data = 'English 中文 日本語 😀\n'.encode() + b'\xe4\xb8[interleaved]\xff\x00\n'
        for size in range(1, 40):
            with self.subTest(size=size):
                self.assertEqual(self.forward([data[i:i+size] for i in range(0, len(data), size)]), data)

    def test_redacts_all_chunk_splits_including_overlapping_prefixes(self):
        for secret in ('fixture-key', 'aba', 'aaaa', '钥匙', 'x'):
            data = ('prefix ' + secret * 3 + ' suffix ' + secret[:2] + '\n').encode()
            expected = data.replace(secret.encode(), b'[REDACTED]')
            for split in range(1, len(data)):
                with self.subTest(secret=secret, split=split):
                    self.assertEqual(self.forward([data[:split], data[split:]], secret), expected)
            self.assertEqual(self.forward([bytes([b]) for b in data], secret), expected)

    def test_random_fragmentation_matches_whole_stream_redaction(self):
        rng = random.Random(1749)
        for _ in range(100):
            secret = ''.join(rng.choice('abc') for _ in range(rng.randint(1, 16)))
            data = b''.join(rng.choice((secret.encode(), b'ababc', b'\xff', '中'.encode())) for _ in range(100))
            chunks = []
            while data[len(b''.join(chunks)):]:
                pos = sum(map(len, chunks)); chunks.append(data[pos:pos+rng.randint(1, 31)])
            self.assertEqual(self.forward(chunks, secret), data.replace(secret.encode(), b'[REDACTED]'))

    def test_partial_secret_at_eof_is_not_lost(self):
        self.assertEqual(self.forward([b'hello fixture-']), b'hello fixture-')

    def test_empty_key_does_not_insert_redactions(self):
        self.assertEqual(self.forward([b'abc', b'\xff'], ''), b'abc\xff')

    def test_partial_sink_writes_are_completed(self):
        class ShortSink(io.BytesIO):
            def write(self, data):
                return super().write(data[:3])
        sink = ShortSink()
        boot.forward_log_bytes(io.BytesIO(b'hello fixture-key world'), sink, 'fixture-key')
        self.assertEqual(sink.getvalue(), b'hello [REDACTED] world')

    def test_zero_or_none_sink_write_fails_instead_of_spinning(self):
        for result in (None, 0):
            class BrokenSink:
                def write(self, data): return result
            with self.assertRaisesRegex(OSError, 'stopped accepting'):
                boot.forward_log_bytes(io.BytesIO(b'hello'), BrokenSink(), '')

    def test_multi_megabyte_line_is_forwarded_in_bounded_chunks(self):
        data = b'a' * (4 * 1024 * 1024) + b'fixture-key\xff'
        class CountingSink:
            def __init__(self): self.digest = hashlib.sha256(); self.largest = 0
            def write(self, block):
                self.largest = max(self.largest, len(block)); self.digest.update(block); return len(block)
            def flush(self): pass
        sink = CountingSink()
        boot.forward_log_bytes(io.BytesIO(data), sink, 'fixture-key')
        self.assertLessEqual(sink.largest, 65536 + len('fixture-key'))
        self.assertEqual(sink.digest.digest(), hashlib.sha256(data.replace(b'fixture-key', b'[REDACTED]')).digest())


@unittest.skipUnless(sys.platform == 'linux', 'Real Linux subprocess/pipe regression')
class LogProcessTests(unittest.TestCase):
    def start(self, program, sink=None, **kwargs):
        process = subprocess.Popen([sys.executable, '-u', '-c', program],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
        sink = FlushedBytes() if sink is None else sink
        forwarder = boot.ServerLogForwarder(process, 'fixture-key', sink,
                                           shutdown_timeout=.15, kill_timeout=2, **kwargs)
        def cleanup():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
            forwarder.thread.join(timeout=3)
            process.stdout.close(); process.stdin.close()
        self.addCleanup(cleanup)
        forwarder.start()
        return process, forwarder, sink

    def test_exact_malformed_utf8_then_more_than_pipe_capacity_completes(self):
        program = "import os\nos.write(1,b'\\xe4\\xb8[invalid UTF8]\\n')\ndata=b'harmless diagnostic line\\n'*40000\nwhile data:\n n=os.write(1,data); data=data[n:]\n"
        process, forwarder, sink = self.start(program)
        self.assertEqual(forwarder.wait(), 0)
        self.assertEqual(sink.getvalue(), b'\xe4\xb8[invalid UTF8]\n' + b'harmless diagnostic line\n' * 40000)
        self.assertFalse(forwarder.failed.is_set())

    def test_small_line_is_flushed_before_next_write_or_eof(self):
        process, forwarder, sink = self.start("import os,sys\nos.write(1,b'ready\\n')\nsys.stdin.readline()\n")
        self.assertTrue(sink.flushed.wait(3))
        self.assertEqual(sink.getvalue(), b'ready\n')
        self.assertIsNone(process.poll())
        process.stdin.write(b'go\n'); process.stdin.flush()
        self.assertEqual(forwarder.wait(), 0)

    def test_concurrent_stdout_stderr_writers_preserve_all_bytes(self):
        program = "import os\npid=os.fork()\ndata=('漢'*100000).encode() if pid else b'ASCII status\\n'*30000\nfd=1 if pid else 2\nwhile data:\n n=os.write(fd,data); data=data[n:]\nif pid: os.waitpid(pid,0)\n"
        process, forwarder, sink = self.start(program)
        self.assertEqual(forwarder.wait(), 0)
        expected = ('漢' * 100000).encode() + b'ASCII status\n' * 30000
        self.assertEqual(collections.Counter(sink.getvalue()), collections.Counter(expected))

    def test_nonzero_server_exit_is_preserved(self):
        process, forwarder, sink = self.start("import sys\nprint('last message')\nsys.exit(7)\n")
        self.assertEqual(forwarder.wait(), 7)
        self.assertEqual(sink.getvalue(), b'last message\n')

    def test_sink_exception_stops_child_even_without_main_wait_loop(self):
        class BrokenSink:
            def write(self, data): raise OSError('private fixture-key must not escape')
        process, forwarder, sink = self.start("import time\nprint('ready',flush=True)\ntime.sleep(30)\n", BrokenSink())
        self.assertTrue(forwarder.failed.wait(3))
        forwarder.thread.join(3)
        self.assertIsNotNone(process.poll())
        with self.assertRaisesRegex(RuntimeError, r'forwarding failed \(OSError\)') as result:
            forwarder.check()
        self.assertNotIn('fixture-key', str(result.exception))

    def test_sink_failure_is_not_hidden_by_successful_server_exit(self):
        class BrokenSink:
            def write(self, data): raise BrokenPipeError('closed sink')
        process, forwarder, sink = self.start("print('done')", BrokenSink())
        with self.assertRaisesRegex(RuntimeError, 'forwarding failed'):
            forwarder.wait()

    def test_closed_log_pipe_with_live_server_is_fatal(self):
        process, forwarder, sink = self.start("import os,time\nos.close(1); os.close(2)\ntime.sleep(30)\n")
        self.assertTrue(forwarder.failed.wait(3))
        forwarder.thread.join(3)
        self.assertIsNotNone(process.poll())
        with self.assertRaisesRegex(RuntimeError, 'forwarding failed'):
            forwarder.wait()

    def test_sink_failure_escalates_when_child_ignores_sigterm(self):
        class BrokenSink:
            def write(self, data): raise OSError('sink unavailable')
        process, forwarder, sink = self.start("import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nprint('ready',flush=True)\ntime.sleep(30)\n", BrokenSink())
        self.assertTrue(forwarder.failed.wait(3))
        forwarder.thread.join(3)
        self.assertEqual(process.poll(), -signal.SIGKILL)

    def test_requested_shutdown_is_bounded_even_with_sigterm_ignored(self):
        process, forwarder, sink = self.start("import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nprint('ready',flush=True)\ntime.sleep(30)\n")
        self.assertTrue(sink.flushed.wait(3))
        forwarder.request_stop()
        self.assertEqual(forwarder.wait(), -signal.SIGKILL)
        self.assertFalse(forwarder.failed.is_set())


if __name__ == '__main__':
    unittest.main()
