import numpy as np, hashlib, sys
a = np.load(sys.argv[1], mmap_mode='r'); h = hashlib.sha256()
for i in range(0, a.shape[0], 100000):
    h.update(np.ascontiguousarray(a[i:i+100000]).tobytes())
print(a.shape, h.hexdigest())
