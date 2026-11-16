import numpy as np
from PIL import Image
def edge_profile(img):
    a = img.astype(int)
    dx = np.abs(np.diff(a, axis=1)).max(-1)  # H x W-1  (edge between x and x+1)
    dy = np.abs(np.diff(a, axis=0)).max(-1)
    ex = (dx > 40).sum(0).astype(float); ey = (dy > 40).sum(1).astype(float)
    return ex, ey
def best_pitch(e, lo=4.0, hi=40.0):
    pos = np.arange(len(e)) + 1.0  # boundary between pixel i and i+1 is at i+1
    w = e / max(e.sum(), 1)
    best = (0, None, None)
    for p in np.arange(lo, hi, 0.05):
        ph = (pos % p) / p * 2 * np.pi
        z = np.abs((w * np.exp(1j * ph)).sum())   # phase coherence of edges at this pitch
        if z > best[0]: best = (z, p, np.angle((w * np.exp(1j * ph)).sum()))
    return best
def pitch_of(img):
    ex, ey = edge_profile(img)
    zx, px, _ = best_pitch(ex); zy, py, _ = best_pitch(ey)
    return px, py, zx, zy
if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        im = np.asarray(Image.open(f).convert("RGB"))
        px, py, zx, zy = pitch_of(im)
        print(f, im.shape[:2], f"pitch x={px:.2f} y={py:.2f} coh={zx:.2f},{zy:.2f}")
