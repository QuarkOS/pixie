"""Pixie v3 typo-caption variants (CPU only). Reads v3/data (unchanged), writes a NEW folder v3/data_typo.

For every train row: one deterministic typo copy of its main caption (1 typo 70%, 2 typos 30%) on word
tokens of >= 4 letters. Ops: drop / swap adjacent / double / adjacent-key (QWERTY). Stored as text_typo
[N,26] + has_typo [N] next to the original arrays; the trainer swaps it in with --p-typo per sample.
Eval/probe/iid arrays, prompts and the image cache are byte-identical to v3/data (same rows, same order).
Blind-50 / eval-50 / probe: train rows already exclude them; typo strings are re-checked against Blind-50
phrases and eval-50 exact captions and rejected if they match.
"""
import hashlib, json, os, random, re, shutil, sys, time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

def _v3_root():
    if os.environ.get("PIXIE_V3"):
        return Path(os.environ["PIXIE_V3"])
    if os.environ.get("PIXIE_ROOT"):
        return Path(os.environ["PIXIE_ROOT"]) / "v3"
    raise SystemExit("set PIXIE_V3 or PIXIE_ROOT")

V3 = _v3_root(); SRC = V3 / "data"; OUT = V3 / "data_typo"; TOKD = V3 / "tok"
BOS, EOS, L, TEXT_IDS, SEED = 49406, 49407, 26, 24, 0
P_TWO = 0.30
KB = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
POS = {c: (r, i) for r, row in enumerate(KB) for i, c in enumerate(row)}
def neigh(c):
    if c not in POS: return []
    r, i = POS[c]; out = []
    for dr, di in ((0, -1), (0, 1), (-1, 0), (-1, 1), (1, -1), (1, 0)):
        rr, ii = r + dr, i + di
        if 0 <= rr < 3 and 0 <= ii < len(KB[rr]): out.append(KB[rr][ii])
    return out
WORD = re.compile(r"[a-z]{4,}")

def typo_word(w, rng):
    for _ in range(10):
        op = rng.choice(("drop", "swap", "double", "key"))
        if op == "drop":
            i = rng.randrange(1, len(w)); n = w[:i] + w[i + 1:]
        elif op == "swap":
            i = rng.randrange(1, len(w) - 1); n = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        elif op == "double":
            i = rng.randrange(0, len(w)); n = w[:i + 1] + w[i] + w[i + 1:]
        else:
            i = rng.randrange(1, len(w)); nb = neigh(w[i])
            if not nb: continue
            n = w[:i] + rng.choice(nb) + w[i + 1:]
        if n != w: return n, op
    return None, None

def make_typo(cap, rng):
    spans = [m.span() for m in WORD.finditer(cap)]
    if not spans: return None, []
    k = 2 if (rng.random() < P_TWO and len(spans) >= 2) else 1
    pick = sorted(rng.sample(range(len(spans)), k), reverse=True)
    s = cap; ops = []
    for j in pick:
        a, b = spans[j]; n, op = typo_word(s[a:b], rng)
        if n is None: continue
        s = s[:a] + n + s[b:]; ops.append(op)
    return (s if s != cap else None), ops

def main():
    t0 = time.time()
    tok = Tokenizer.from_file(str(TOKD / "tokenizer.json"))
    blind = [l.strip().lower() for l in (TOKD / "prompts_blind50.txt").read_text().splitlines() if l.strip()]
    assert len(blind) == 50
    bl_re = re.compile(r"\b(" + "|".join(re.escape(p) for p in blind) + r")\b")
    eval_set = {p.strip().lower() for p in json.loads((SRC / "prompts_v3.json").read_text())["eval50"]}
    tr = np.load(SRC / "v3_train.npz"); arrs = {k: tr[k] for k in tr.files}
    TM = arrs["text_main"]; N = TM.shape[0]
    uniq, inv = np.unique(TM, axis=0, return_inverse=True); inv = inv.reshape(-1)
    def ids_of(row):
        r = row.tolist(); e = r.index(EOS, 1); return r[1:e]
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    byte_dec = {chr(c): b for b, c in zip(bs, cs)}
    vocab_inv = {v: k for k, v in tok.get_vocab().items()}
    def dec(ids):
        txt = "".join(vocab_inv[i] for i in ids).replace("</w>", " ")
        return bytearray(byte_dec[ch] if ch in byte_dec else ord(" ") for ch in txt).decode("utf-8", "replace").strip()
    caps = [dec(ids_of(u)) for u in uniq]
    def enc(strings):
        out = np.full((len(strings), L), EOS, np.uint16); lens = np.zeros(len(strings), np.int32)
        for k, e in enumerate(tok.encode_batch(list(strings), add_special_tokens=False)):
            ids = e.ids[:TEXT_IDS]; out[k, 0] = BOS; out[k, 1:1 + len(ids)] = ids; out[k, 1 + len(ids)] = EOS
            lens[k] = len(e.ids)
        return out, lens
    re_u, _ = enc(caps)
    rt_ok = (re_u == uniq).all(1)
    print(f"unique captions {len(uniq)} decode->encode exact {rt_ok.mean()*100:.3f}% {time.time()-t0:.0f}s", flush=True)
    typo = [None] * N; opsc = {"drop": 0, "swap": 0, "double": 0, "key": 0}; ntwo = 0
    for i in range(N):
        u = inv[i]
        if not rt_ok[u]: continue
        rng = random.Random(SEED * 1_000_003 + i)
        s, ops = make_typo(caps[u], rng)
        if s is None or bl_re.search(s) or s in eval_set: continue
        typo[i] = s
        for o in ops: opsc[o] += 1
        ntwo += len(ops) == 2
    idx = [i for i in range(N) if typo[i] is not None]
    tt, tl = enc([typo[i] for i in idx])
    ok = tl <= TEXT_IDS
    text_typo = TM.copy(); has_typo = np.zeros(N, bool)
    sel = np.array(idx, dtype=np.int64)[ok]; text_typo[sel] = tt[ok]; has_typo[sel] = True
    same = (text_typo[sel] == TM[sel]).all(1); has_typo[sel[same]] = False
    print(f"typo rows {int(has_typo.sum())} of {N} ({has_typo.mean()*100:.2f}%), two-typo {ntwo}, too long {int((~ok).sum())}, ops {opsc} {time.time()-t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(OUT / "v3_train.npz", **arrs, text_typo=text_typo, has_typo=has_typo)
    for f in ("v3_eval.npz", "prompts_v3.json"):
        shutil.copyfile(SRC / f, OUT / f)
    sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for f in ("v3_eval.npz", "prompts_v3.json"):
        assert sh(SRC / f) == sh(OUT / f)
    man = json.loads((SRC / "manifest_v3.json").read_text())
    ex_rng = random.Random(7); ex = []
    for i in ex_rng.sample(list(np.where(has_typo)[0]), 40):
        ex.append([caps[inv[i]], typo[i]])
    man["typo"] = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(SRC), "seed": SEED,
        "method": "one deterministic typo copy per train row (random.Random(seed*1000003+row)); 1 typo (70%) or 2 typos on different words (30%); words of >=4 letters; ops drop(not first letter)/swap adjacent(not first letter)/double/adjacent QWERTY key(not first letter); rejected if it matches a Blind-50 phrase or eval-50 caption or exceeds 24 CLIP ids",
        "applied_by_trainer": "--p-typo P: per sample, with prob P the caption becomes text_typo (if has_typo), after synonym and before subject/blank overrides",
        "rows": N, "typo_rows": int(has_typo.sum()), "two_typo_rows": ntwo, "ops": opsc,
        "roundtrip_exact_pct": round(float(rt_ok.mean() * 100), 4), "examples": ex,
        "eval_sha256_unchanged": sh(OUT / "v3_eval.npz"), "sha256_v3_train.npz": sh(OUT / "v3_train.npz")}
    (OUT / "manifest_v3.json").write_text(json.dumps(man, indent=1))
    print("examples:"); [print("  ", a, "->", b) for a, b in ex[:20]]
    print(f"done {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
