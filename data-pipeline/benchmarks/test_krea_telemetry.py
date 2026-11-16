"""Synthetic event/telemetry checks do not measure GPU performance."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("krea_telemetry", Path(__file__).with_name("krea_telemetry.py"))
telemetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telemetry)


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_runner_events_drop_private_fields_and_keep_numeric_phases(self):
        event = {"t": 110, "gen_s": 6, "extract_s": 1, "sprites": 4, "usable": 3,
                 "cells": [{"subject": "private", "file": "private"}], "secret": "private"}
        row = telemetry.runner_event(event, 100, 2)
        self.assertEqual((row["start_s"], row["end_s"], row["request_end_s"]), (3, 10, 9))
        self.assertEqual((row["request_s"], row["extract_write_s"], row["usable"]), (6, 1, 3))
        self.assertNotIn("private", json.dumps(row))
        self.assertIsNone(telemetry.runner_event(event, 120, 2))
        row = telemetry.runner_event({"t": 110, "error": "private asset path"}, 100, 2)
        self.assertFalse(row["ok"])
        self.assertNotIn("private", json.dumps(row))

    def test_streaks_group_concurrency_join_gpu_and_count_errors(self):
        rows = []
        for i, seconds in enumerate([4, 4, 8, 8, 4]):
            rows.append({"phase": "measured", "inflight": 2, "ok": True,
                         "start_s": i * 10, "end_s": i * 10 + seconds, "request_s": seconds})
        rows.append({"phase": "measured", "inflight": 1, "ok": True,
                     "start_s": 60, "end_s": 72, "request_s": 12})
        rows.append({"phase": "measured", "ok": False})
        (self.root / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        with (self.root / "gpu.csv").open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["elapsed_s", "clocks.sm", "clocks_event_reasons.hw_slowdown", "unsafe"])
            writer.writerows([[1, 1800, 0, "secret"], [21, 900, 1, "secret"], [31, 900, 1, "secret"]])
        args = SimpleNamespace(run=self.root, out=self.root / "analysis.json", slow_ratio=1.75, baseline_seconds=None)
        self.assertEqual(telemetry.analyze(args), 1)
        report = json.loads(args.out.read_text())
        self.assertEqual((report["completed"], report["errors"]), (6, 1))
        group = report["groups"][1]
        self.assertEqual((group["inflight"], group["slow"], group["slow_fraction"]), (2, 2, .4))
        self.assertEqual(group["slow_streaks"][0]["requests"], 2)
        self.assertEqual(group["slow_gpu"]["fields"]["clocks.sm"]["median"], 900)
        self.assertNotIn("secret", args.out.read_text())
        with self.assertRaises(FileExistsError):
            telemetry.analyze(args)

    def test_old_unaligned_recording_is_refused(self):
        (self.root / "requests.jsonl").write_text(json.dumps({"phase": "measured", "ok": True, "request_s": 4}))
        with self.assertRaisesRegex(ValueError, "aligned"):
            telemetry.analyze(SimpleNamespace(run=self.root, out=self.root / "analysis.json", slow_ratio=1.75, baseline_seconds=None))

    def test_benchmark_trials_do_not_join_separate_streaks(self):
        rows = [{"phase": "measured", "inflight": 2, "ok": True, "trial": i,
                 "start_s": i * 20, "end_s": i * 20 + 8, "request_s": 8} for i in range(2)]
        (self.root / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (self.root / "gpu.csv").write_text("elapsed_s,clocks.sm\n")
        args = SimpleNamespace(run=self.root, out=self.root / "analysis.json", slow_ratio=1.75, baseline_seconds=4)
        self.assertEqual(telemetry.analyze(args), 0)
        streaks = json.loads(args.out.read_text())["groups"][0]["slow_streaks"]
        self.assertEqual([s["requests"] for s in streaks], [1, 1])

    def test_collector_copies_only_new_sanitized_events(self):
        events = self.root / "events.jsonl"
        events.write_text(json.dumps({"t": 90, "error": "old private error"}) + "\n")
        class Gpu:
            def __init__(self, *args):
                self.samples, self.errors = 1, 0
                def start():
                    with events.open("a") as out:
                        out.write(json.dumps({"t": 101, "gen_s": .5, "extract_s": .2, "cells": "private"}) + "\n")
                self.thread = SimpleNamespace(start=start, join=lambda **kwargs: None)
                self.stop = SimpleNamespace(set=lambda: None)
        args = SimpleNamespace(gpu="0", seconds=.02, events=events, inflight=2, out=self.root / "new")
        with patch.dict("sys.modules", {"krea_unique": SimpleNamespace(GpuLog=Gpu)}), patch.object(telemetry.time, "time", return_value=100):
            self.assertEqual(telemetry.record(args), 0)
        value = (args.out / "requests.jsonl").read_text()
        self.assertEqual(len(value.splitlines()), 1)
        self.assertNotIn("private", value)
        self.assertEqual(json.loads((args.out / "record-summary.json").read_text())["events"], 1)


if __name__ == "__main__":
    unittest.main()
