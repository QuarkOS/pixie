"""Bounded host/stream timing of completed updates, without phase synchronization."""
import statistics
import time

import torch


PHASES = ("data_setup", "clip_or_cache", "forward_loss", "backward", "clip_grad_adamw")


class StepProfile:
    def __init__(self, device, steps, warmup):
        self.device, self.steps, self.warmup = device, steps, warmup
        self.rows = []
        self.active = False

    def begin(self, segment_step, images):
        self.active = self.warmup <= segment_step < self.warmup + self.steps
        if not self.active:
            return
        self.segment_step, self.images = segment_step, images
        self.host = [time.perf_counter()]
        self.events = []
        self._event()

    def _event(self):
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                event = torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream(self.device))
            self.events.append(event)

    def mark(self):
        if self.active:
            self._event()
            self.host.append(time.perf_counter())

    def finish(self):
        """Call after the trainer's existing end-update device synchronization."""
        if not self.active:
            return
        if len(self.host) != len(PHASES) + 1:
            raise RuntimeError("step profile has incomplete phase boundaries")
        self.rows.append({
            "segment_step": self.segment_step, "images": self.images,
            "completed_update_seconds": time.perf_counter() - self.host[0],
            "host_phase_seconds": {name: end - start for name, start, end in
                                   zip(PHASES, self.host, self.host[1:])},
            "cuda_stream_phase_ms": {name: start.elapsed_time(end) for name, start, end in
                                     zip(PHASES, self.events, self.events[1:])}
                                    if self.events else None,
        })
        self.active = False

    def report(self):
        summary = {}
        for field in ("host_phase_seconds", "cuda_stream_phase_ms"):
            summary[field] = {}
            for phase in PHASES:
                values = [row[field][phase] for row in self.rows if row[field] is not None]
                summary[field][phase] = ({"median": statistics.median(values),
                                          "range": [min(values), max(values)]} if values else None)
        return {
            "scope": "completed updates only; excludes evaluation, sampling and checkpoint writes",
            "timing_note": "host phases include dispatch/blocking waits; CUDA stream intervals include dispatch gaps, not only kernel execution; do not add host and CUDA values",
            "perturbs_execution": True, "warmup_updates_excluded": self.warmup,
            "requested_updates": self.steps, "completed_profiled_updates": len(self.rows),
            "completed_profiled_images": sum(row["images"] for row in self.rows),
            "rows": self.rows, "summary": summary,
        }
