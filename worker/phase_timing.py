"""Exclusive wall-clock worker phases; CUDA synchronization is opt-in."""
from contextlib import contextmanager
import time


class PhaseTimer:
    def __init__(self, enabled=False, synchronize=lambda: None, clock=time.perf_counter):
        self.enabled = enabled
        self.synchronize = synchronize
        self.clock = clock
        self.seconds = {}
        self.active = False
        if enabled:
            synchronize()
            self.started = clock()

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        if self.active:
            raise RuntimeError("worker phases must not overlap")
        self.synchronize()
        began = self.clock()
        self.active = True
        try:
            yield
        finally:
            self.synchronize()
            self.seconds[name] = self.seconds.get(name, 0.) + self.clock() - began
            self.active = False

    def finish(self):
        if not self.enabled:
            return None
        if self.active:
            raise RuntimeError("cannot finish an active phase")
        self.synchronize()
        elapsed = self.clock() - self.started
        return {"scope": "synchronized generation wall time; excludes queue, polling, result serialization and HTTP; synchronization can perturb latency",
                "total_ms": elapsed * 1000,
                "phase_ms": {key: value * 1000 for key, value in self.seconds.items()},
                "unclassified_ms": (elapsed - sum(self.seconds.values())) * 1000}
