"""Random-read throughput of the CLIP feature cache (cold: file pages dropped with POSIX_FADV_DONTNEED, no sudo)."""
import json, os, sys, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
d = sys.argv[1]; meta = json.load(open(os.path.join(d, "meta.json")))
path = os.path.join(d, meta["file"]); row = meta["stored_positions"] * 768 * 2; C = meta["captions"]
fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
rng = np.random.default_rng(1)
for workers in (1, 6):
    buf = [bytearray(row) for _ in range(workers)]
    def batch(w, ids):
        for i in np.sort(ids):
            os.preadv(fd, [buf[w]], int(i) * row)
    n = 0; t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        while time.time() - t < 8:
            fs = [ex.submit(batch, w, rng.integers(0, C, 115)) for w in range(workers)]
            [f.result() for f in fs]; n += 115 * workers
    dt = time.time() - t
    print(f"workers {workers}: {n / dt:,.0f} rows/s = {n * row / dt / 1e6:.0f} MB/s random 26 KB reads = {n / dt / 115:.1f} batches(115 rows)/s")
