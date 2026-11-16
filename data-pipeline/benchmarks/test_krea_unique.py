"""CPU protocol and artifact checks; mock timings are not GPU benchmarks."""
import base64
import contextlib
import errno
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image

spec = importlib.util.spec_from_file_location("krea_unique", Path(__file__).with_name("krea_unique.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class MockServer(BaseHTTPRequestHandler):
    bodies = []
    fail = False

    def log_message(self, *_args):
        pass

    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.bodies.append(body)
        self.reply(202, {"poll_url": "/sdcpp/v1/jobs/one"})

    def do_GET(self):
        if self.path.endswith("models/status"):
            self.reply(200, {"state": "ready"})
        elif self.fail:
            self.reply(200, {"status": "failed", "error": "secret must never escape"})
        else:
            stream = io.BytesIO()
            Image.new("RGB", (512, 512), "magenta").save(stream, format="PNG")
            self.reply(200, {"status": "completed", "result": {"images": [
                {"b64_json": base64.b64encode(stream.getvalue()).decode()}]}})


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prompts = self.root / "prompts.txt"
        self.prompts.write_text("\n".join(f"pixel art sheet subject {i}" for i in range(30)))
        self.holdout = self.root / "holdout.json"
        self.holdout.write_text(json.dumps([f"held phrase {i}" for i in range(50)]))

    def tearDown(self):
        self.temp.cleanup()

    def args(self, server):
        return bench.parser().parse_args(["--server", server, "--gpu", "0", "--prompts", str(self.prompts),
            "--blind50", str(self.holdout), "--eval50", str(self.holdout), "--lora", "test.safetensors",
            "--out", str(self.root / "new-run"), "--flash-attn-declared", "off", "--conditioning-cache", "off",
            "--trials", "2", "--sheets", "1", "--warmup", "1"])

    @contextlib.contextmanager
    def server(self):
        MockServer.bodies = []
        MockServer.fail = False
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockServer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_full_protocol_unique_prompts_and_new_folder(self):
        with self.server() as server, patch.object(bench.GpuLog, "run"), contextlib.redirect_stdout(io.StringIO()):
            args = self.args(server)
            self.assertEqual(bench.run(args), 0)
            summary = json.loads((args.out / "summary.json").read_text())
            config = json.loads((args.out / "config.json").read_text())
            self.assertEqual(summary["completed"], 7)
            self.assertEqual([t["inflight"] for t in summary["trials"]], [1, 1, 2, 3, 3, 2, 1])
            self.assertEqual(len({b["prompt"] for b in MockServer.bodies}), 7)
            self.assertEqual(len(list(args.out.glob("sheet-*/sheet.png"))), 7)
            self.assertFalse(config["settings_verified"])
            requests = [json.loads(line) for line in (args.out / "requests.jsonl").read_text().splitlines()]
            self.assertTrue(all(0 <= r["start_s"] <= r["request_end_s"] <= r["end_s"] for r in requests))
            self.assertNotIn(server, json.dumps(config))
            with self.assertRaises(FileExistsError):
                bench.run(args)

    def test_failures_are_counted_and_raw_messages_hidden(self):
        with self.server() as server, patch.object(bench.GpuLog, "run"), contextlib.redirect_stdout(io.StringIO()):
            MockServer.fail = True
            args = self.args(server)
            self.assertEqual(bench.run(args), 1)
            summary = json.loads((args.out / "summary.json").read_text())
            self.assertEqual(summary["errors"], 7)
            self.assertFalse(summary["comparison_valid"])
            self.assertNotIn("secret", (args.out / "requests.jsonl").read_text())

    def test_punctuation_holdouts_and_duplicates(self):
        self.prompts.write_text("pixel art: HELD-phrase 2!")
        with self.assertRaisesRegex(ValueError, "held-out"):
            bench.read_prompts(self.prompts, [self.holdout], 1)
        self.prompts.write_text("Sheet-one\nsheet one")
        with self.assertRaisesRegex(ValueError, "unique"):
            bench.read_prompts(self.prompts, [self.holdout], 1)

    def test_remote_url_and_cross_origin_poll_refused(self):
        for url in ["https://127.0.0.1:1", "http://example.com", "http://user:pass@localhost"]:
            with self.assertRaises(ValueError):
                bench.loopback_url(url)
        with patch.object(bench, "request", return_value=(202, {"poll_url": "http://example.com/job"})):
            with self.assertRaisesRegex(ValueError, "cross-origin"):
                bench.generate("http://127.0.0.1:123", {}, self.root / "sheet", 1)

    def test_server_log_only_copies_numeric_allowlist_after_offset(self):
        path = self.root / "server.log"
        prefix = "sampling completed in 99.0s\n"
        path.write_text(prefix + "private-path decoding completed in 0.85s secret\nother secret\n")
        output = self.root / "phases.jsonl"
        bench.safe_server_timings(path, len(prefix), output)
        self.assertEqual(json.loads(output.read_text()), {"phase": "vae", "seconds": 0.85})

    def test_malformed_holdout_fails(self):
        self.holdout.write_text('["only one"]')
        with self.assertRaisesRegex(ValueError, "50"):
            bench.read_prompts(self.prompts, [self.holdout], 1)

    def test_existing_server_cannot_claim_flash_was_applied(self):
        args = self.args("http://127.0.0.1:17890")
        args.flash_attn, args.flash_attn_declared = "off", None
        with patch.object(bench, "request") as send:
            with self.assertRaisesRegex(ValueError, "only with --sd-server"):
                bench.run(args)
            send.assert_not_called()
        self.assertFalse(args.out.exists())

    def test_flash_off_reaches_real_child_argv_and_only_child_is_stopped(self):
        import socket
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0))
            url = f"http://127.0.0.1:{port.getsockname()[1]}"
        script, record = self.root / "mock_server.py", self.root / "received.json"
        script.write_text('''import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"argv":sys.argv[2:],"gpu":os.environ.get("CUDA_VISIBLE_DEVICES")}))
class Handler(BaseHTTPRequestHandler):
 def log_message(self, *args): pass
 def do_GET(self):
  body = b'{"state":"ready"}'
  self.send_response(200); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
HTTPServer(("127.0.0.1",int(sys.argv[sys.argv.index("--listen-port")+1])),Handler).serve_forever()
''')
        argv = self.root / "server-args.json"
        argv.write_text(json.dumps([str(script), str(record)]))
        args = self.args(url)
        args.sd_server, args.server_args = Path(sys.executable), argv
        args.flash_attn, args.flash_attn_declared, args.startup_timeout = "off", None, 5
        children = []
        start = subprocess.Popen
        def launch(*a, **kw):
            child = start(*a, **kw)
            children.append(child)
            return child
        with patch.object(bench, "idle_gpu"), patch.object(bench.subprocess, "Popen", side_effect=launch):
            with self.assertRaisesRegex(RuntimeError, "test cleanup"):
                with bench.server_session(args, url) as settings:
                    received = json.loads(record.read_text())
                    self.assertEqual(received["argv"][-2:], ["--flash-attn", "off"])
                    self.assertEqual(received["gpu"], "0")
                    self.assertEqual(settings["flash_attn_startup_argument"], "off")
                    self.assertNotIn(str(script), json.dumps(settings))
                    self.assertIsNone(children[0].poll())
                    raise RuntimeError("test cleanup")
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    def test_dedicated_gpu_guard_refuses_busy_or_unknown_occupancy(self):
        with patch.dict(os.environ, {}, clear=True):
            for occupancy in ("123456\n", "N/A\n"):
                with patch.object(bench.subprocess, "run", return_value=type("R", (), {"stdout":occupancy})()):
                    with self.assertRaisesRegex(ValueError, "busy"):
                        bench.idle_gpu("0")

    def test_restart_waits_for_port_then_binds_without_reuse(self):
        probe = unittest.mock.MagicMock()
        probe.__enter__.return_value = probe
        probe.bind.side_effect = [OSError(errno.EADDRINUSE, "in use"), None]
        with patch.object(bench.socket, "socket", return_value=probe), \
             patch.object(bench.time, "monotonic", return_value=1), \
             patch.object(bench.time, "sleep") as sleep:
            bench.wait_port_free(17890, 2)
        self.assertEqual(probe.bind.call_count, 2)
        sleep.assert_called_once_with(.2)
        probe.setsockopt.assert_not_called()

    def test_port_wait_is_bounded_and_other_socket_errors_are_not_retried(self):
        probe = unittest.mock.MagicMock()
        probe.__enter__.return_value = probe
        with patch.object(bench.socket, "socket", return_value=probe), \
             patch.object(bench.time, "monotonic", return_value=2), \
             patch.object(bench.time, "sleep") as sleep:
            probe.bind.side_effect = OSError(errno.EADDRINUSE, "in use")
            with self.assertRaisesRegex(TimeoutError, "did not become free"):
                bench.wait_port_free(17890, 2)
            probe.bind.side_effect = OSError(errno.EACCES, "denied")
            with self.assertRaises(OSError) as error:
                bench.wait_port_free(17890, 3)
            self.assertEqual(error.exception.errno, errno.EACCES)
        sleep.assert_not_called()

    def test_server_arg_file_cannot_override_flash_or_loopback(self):
        args = self.args("http://127.0.0.1:17890")
        args.sd_server = Path(sys.executable)
        args.server_args = self.root / "server-args.json"
        args.flash_attn, args.flash_attn_declared = "off", None
        for option in ("--flash-attn=on", "--listen-ip", "--listen-port", "--diffusion-fa"):
            args.server_args.write_text(json.dumps([option, "on"]))
            with self.assertRaisesRegex(ValueError, "must not override"):
                bench.server_command(args, args.server)

    def test_gpu_telemetry_persists_only_selected_numeric_fields(self):
        gpu = bench.GpuLog(self.root / "gpu.csv", "2")
        def response(command, **kwargs):
            self.assertEqual(command[:3], ["nvidia-smi", "-i", "2"])
            gpu.stop.set()
            return type("Result", (), {"stdout": "92, 10414, 240.5, 61, 1700, 9500, P2\n"})()
        with patch.object(gpu, "reasons", return_value=[]), patch.object(bench.subprocess, "run", side_effect=response):
            gpu.run()
        self.assertEqual(gpu.samples, 1)
        self.assertEqual(gpu.errors, 0)
        self.assertIn("10414", gpu.out.read_text())
        self.assertNotIn("uuid", gpu.out.read_text())

    def test_gpu_reason_discovery_uses_supported_names_and_sanitizes_activity(self):
        gpu = bench.GpuLog(self.root / "gpu.csv", "2", origin=0)
        commands = []
        def response(command, **kwargs):
            commands.append(command)
            if "--help-query-gpu" in command:
                return type("Result", (), {"stdout": "clocks_event_reasons.hw_slowdown"})()
            gpu.stop.set()
            return type("Result", (), {"stdout": "92, 10414, 240.5, 61, 1700, 9500, P2, Active\n"})()
        with patch.object(bench.subprocess, "run", side_effect=response):
            gpu.run()
        self.assertIn("clocks_event_reasons.hw_slowdown", " ".join(commands[-1]))
        self.assertEqual(gpu.out.read_text().splitlines()[-1].split(",")[-1], "1")

    def test_gpu_optional_query_failure_keeps_base_telemetry(self):
        gpu = bench.GpuLog(self.root / "gpu.csv", "2", origin=0)
        def response(command, **kwargs):
            if "clocks_event_reasons" in " ".join(command):
                raise subprocess.CalledProcessError(1, command)
            gpu.stop.set()
            return type("Result", (), {"stdout": "92, 10414, 240.5, 61, 1700, 9500, P2\n"})()
        with patch.object(gpu, "reasons", return_value=["clocks_event_reasons.hw_slowdown"]), \
             patch.object(gpu.stop, "wait"), patch.object(bench.subprocess, "run", side_effect=response):
            gpu.run()
        self.assertEqual((gpu.samples, gpu.errors), (1, 1))
        self.assertEqual(gpu.out.read_text().splitlines()[-1].split(",")[-1], "N/A")

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"),
                         "3x3 extraction requires optional NumPy/SciPy dependencies")
    def test_real_extract_v2_writes_nine_rgba_cells(self):
        path = Path(__file__).parents[1] / "extract_v2.py"
        module_spec = importlib.util.spec_from_file_location("test_extract_v2", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        dest = self.root / "cells"
        dest.mkdir()
        accepted = bench.extract_v2(Image.new("RGB", (576, 576), "magenta"), dest, module)
        self.assertEqual(accepted, 0)
        files = list(dest.glob("cell-*.png"))
        self.assertEqual(len(files), 9)
        for path in files:
            with Image.open(path) as image:
                self.assertEqual(image.size, (32, 32))
                self.assertEqual(image.mode, "RGBA")


if __name__ == "__main__":
    unittest.main()
