"""Pixie v3 data build (CPU only).

Set PIXIE_ENC, PIXIE_SPLITS, PIXIE_BLIND50, PIXIE_MANIFEST, PIXIE_TOKENIZER,
PIXIE_V3_OUT, and PIXIE_V2_CODE (the directory that contains v2_textlib.py).
Writes only PIXIE_V3_OUT. Does not read a machine-specific default path.

Restores the 829 subset150k dev/test subjects to training except a ~100-subject
probe set (picked only from those 829, so v1/v2 never saw them either), builds an
iid dev split, drops Blind-50 phrases and eval-50 exact captions, and pre-tokenizes
captions + variants with the CLIP BPE (BOS + <=24 ids + EOS, padded with EOS to 26).

Row ids index an external token cache (same bytes as the encoded shards,
sha256 8de0b0aa... verified when the cache was built).
"""
import glob, hashlib, json, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

def _need(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"set {name}")
    return Path(v)

ENC = _need("PIXIE_ENC")
SPLITS = _need("PIXIE_SPLITS")
BLIND50 = _need("PIXIE_BLIND50")
MANIFEST = _need("PIXIE_MANIFEST")
TOK = _need("PIXIE_TOKENIZER")
OUT = _need("PIXIE_V3_OUT")
sys.path.insert(0, str(_need("PIXIE_V2_CODE")))
from v2_textlib import clean_caption, allow_flip  # noqa: E402

BOS, EOS, L = 49406, 49407, 26
TEXT_IDS = 24
SEED = 0
N_PROBE = 100
PROBE_ROWS_PER_SUBJ = 30
IID_PER_BUCKET = 1000
P_SUBJ, P_SYN = 0.175, 0.10   # recorded in the manifest; applied by the trainer

# Concepts we want to test after the restore: never in the probe.
KEEP_OUT_OF_PROBE = re.compile(
    r"\b(anchor|candelabra|wizard|witch|hat|sunglasses|glasses|umbrella|trumpet|cactus|cacti|"
    r"barrel|compass|backpack|burger|grape|grapes|sunflower|cat|owl|butterfly|sword|potion|shield|"
    r"lamb|dragon|elemental|blobfish|carp|strongman|buffer|sauce|ring|mask|candle|candlestick)\b", re.I)

SYN = {  # small, conservative word-level synonym map (whole words, lowercase)
    "gray": "grey", "grey": "gray", "big": "large", "large": "big", "small": "little",
    "little": "small", "tiny": "small", "huge": "giant", "giant": "huge", "shiny": "glossy",
    "kitten": "cat", "bunny": "rabbit", "rabbit": "bunny", "rock": "stone", "cup": "mug",
    "mug": "cup", "blade": "sword", "closed": "shut", "glowing": "luminous", "ancient": "old",
}
SYN_RE = re.compile(r"\b(" + "|".join(SYN) + r")\b")


def t():
    return f"{time.time() - T0:.1f}s"


def head_word(s):
    w = re.findall(r"[a-z0-9]+", s.lower())
    return w[-1] if w else ""


def main():
    global T0
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)
    rid, subj, cap, sha, near = [], [], [], [], []
    for p in sorted(ENC.glob("train-*.parquet.meta.jsonl")):
        for line in p.open():
            a, b, c, d, e = json.loads(line)
            rid.append(a); subj.append(b); cap.append(c); sha.append(d); near.append(int(e))
    rid = np.array(rid, np.int64); near = np.array(near, np.uint64)
    N = len(rid)
    assert N == 1_405_308 and (np.diff(rid) > 0).all()  # dataset row ids (gapped); tokens.npy index = position
    print("meta", N, t(), flush=True)

    held = {}
    for sp in ("dev", "test"):
        for line in (SPLITS / f"{sp}.jsonl").open():
            r = json.loads(line); held[r["subject"]] = sp
    assert len(held) == 829
    blind = [l.strip().lower() for l in BLIND50.read_text().splitlines() if l.strip()]
    assert len(blind) == 50
    bl_re = re.compile(r"\b(" + "|".join(re.escape(p) for p in blind) + r")\b")
    eval50 = list(json.loads(MANIFEST.read_text())["coverage"])
    eval_set = {p.strip().lower() for p in eval50}

    clean = [clean_caption(c)[0] for c in cap]
    drop_blind = np.array([bool(bl_re.search(c.lower()) or bl_re.search(s.lower())) for c, s in zip(clean, subj)])
    drop_eval = np.array([c.lower() in eval_set for c in clean]) & ~drop_blind
    print("blind50 rows", int(drop_blind.sum()), "eval50 exact", int(drop_eval.sum()), t(), flush=True)

    sfreq = Counter(subj)
    # word document frequency (full corpus), for the concept-rarity buckets
    df = Counter()
    for c in clean:
        df.update(set(re.findall(r"[a-z0-9]+", c.lower())))
    subjects = sorted(sfreq)
    hdf = {s: df[head_word(s)] for s in subjects}

    def sbucket(s):
        f = sfreq[s]
        return "lt_50" if f < 50 else ("50_300" if f <= 300 else "gt_300")

    def cbucket(s):
        f = hdf[s]
        return "rare_lt1k" if f < 1000 else ("mid_1k_5k" if f <= 5000 else "common_gt5k")

    # ---- probe: 100 subjects from the 829 held-out ones, stratified by concept bucket
    cand = [s for s in sorted(held) if not KEEP_OUT_OF_PROBE.search(s) and not bl_re.search(s.lower())]
    byb = defaultdict(list)
    for s in cand:
        byb[cbucket(s)].append(s)
    probe = []
    quota = {"rare_lt1k": 34, "mid_1k_5k": 33, "common_gt5k": 33}
    for b, q in quota.items():
        pool = byb[b]
        pick = list(rng.choice(pool, size=min(q, len(pool)), replace=False))
        probe += pick
    if len(probe) < N_PROBE:  # top up from any bucket
        rest = [s for s in cand if s not in set(probe)]
        probe += list(rng.choice(rest, size=N_PROBE - len(probe), replace=False))
    probe = sorted(probe)
    probe_set = set(probe)
    print("probe subjects", len(probe), {b: sum(cbucket(s) == b for s in probe) for b in quota}, t(), flush=True)

    subj_arr = np.array(subj, dtype=object)
    is_probe_subj = np.array([s in probe_set for s in subj])
    base_ok = ~(drop_blind | drop_eval)

    # probe eval rows: up to 30 per subject
    probe_rows = []
    by_s = defaultdict(list)
    for i in np.where(is_probe_subj & base_ok)[0]:
        by_s[subj[i]].append(i)
    for s in probe:
        r = by_s[s]
        probe_rows += sorted(rng.choice(r, size=min(PROBE_ROWS_PER_SUBJ, len(r)), replace=False).tolist())
    probe_rows = np.array(probe_rows, np.int64)

    # iid rows: stratified by concept bucket over non-probe eligible rows
    elig = base_ok & ~is_probe_subj
    v2_seen_subj = {s for s in subjects if s not in held}
    iid_rows, iid_cb = [], []
    cb_row = np.array([cbucket(s) for s in subj])
    for b in quota:
        pool = np.where(elig & (cb_row == b))[0]
        pick = np.sort(rng.choice(pool, size=min(IID_PER_BUCKET, len(pool)), replace=False))
        iid_rows += pick.tolist(); iid_cb += [b] * len(pick)
    iid_rows = np.array(iid_rows, np.int64)
    print("probe rows", len(probe_rows), "iid rows", len(iid_rows), t(), flush=True)

    # ---- training rows: everything eligible except iid rows, exact-sha dupes of eval rows,
    # near-duplicates (Hamming <= 4) of any probe-subject row or iid row, and intra-train exact dupes.
    train = elig.copy()
    train[iid_rows] = False
    eval_sha = {sha[i] for i in np.concatenate([np.where(is_probe_subj)[0], iid_rows])}
    for i in np.where(train)[0]:
        if sha[i] in eval_sha:
            train[i] = False
    n_sha = int(elig.sum() - train.sum() - len(iid_rows))
    # Near-duplicates: the 64-bit near hash is loose (median ~90 Hamming<=4 matches per sprite, some
    # refs match >10k), so only refs whose hash is specific (<= NEAR_MAX_MATCH corpus matches) remove
    # their Hamming<=4 neighbours; everything else relies on exact-sprite (sha) and subject exclusion.
    NEAR_MAX_MATCH = 50
    refs = np.concatenate([probe_rows, iid_rows])
    nd = 0; n_specific = 0
    for r in refs:
        m = np.bitwise_count(np.bitwise_xor(near, near[r])) <= 4
        if int(m.sum()) - 1 > NEAR_MAX_MATCH:
            continue
        n_specific += 1
        hit = m & train
        nd += int(hit.sum()); train[hit] = False
    print("near-dup specific refs", n_specific, "of", len(refs), flush=True)
    first = {}
    nx = 0
    for i in np.where(train)[0]:
        if sha[i] in first:
            train[i] = False; nx += 1
        else:
            first[sha[i]] = i
    tr = np.where(train)[0]
    print("train rows", len(tr), "sha-dupe-of-eval", n_sha, "near-dupe", nd, "intra dupes", nx, t(), flush=True)

    # ---- tokenization
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOK))

    def enc(strings):
        out = np.full((len(strings), L), EOS, np.uint16)
        lens = np.zeros(len(strings), np.int16)
        for k, e in enumerate(tok.encode_batch(list(strings), add_special_tokens=False)):
            ids = e.ids[:TEXT_IDS]
            out[k, 0] = BOS; out[k, 1:1 + len(ids)] = ids; out[k, 1 + len(ids)] = EOS
            lens[k] = len(e.ids)
        return out, lens

    ucap = sorted(set(clean[i] for i in tr) | set(clean[i] for i in probe_rows) | set(clean[i] for i in iid_rows))
    cidx = {c: k for k, c in enumerate(ucap)}
    utext, ulen = enc(ucap)
    print("unique captions", len(ucap), "max len", int(ulen.max()), "truncated", int((ulen > TEXT_IDS).sum()), t(), flush=True)

    def syn(c):
        s = SYN_RE.sub(lambda m: SYN[m.group(1)], c.lower())
        return s if s != c.lower() and not bl_re.search(s) and s not in eval_set else None
    usyn = [syn(c) for c in ucap]
    has_syn_u = np.array([s is not None for s in usyn])
    stext_u, _ = enc([s if s is not None else "" for s in usyn])
    stext_u[~has_syn_u] = utext[~has_syn_u]

    usub = sorted(set(subj[i] for i in tr))
    sidx = {s: k for k, s in enumerate(usub)}
    sub_clean = [clean_caption(s)[0] for s in usub]
    assert not any(bl_re.search(s.lower()) for s in sub_clean)
    subj_text, _ = enc(sub_clean)

    ci = np.array([cidx[clean[i]] for i in tr], np.int32)
    text_main = utext[ci]; text_syn = stext_u[ci]; has_syn = has_syn_u[ci]
    subj_id = np.array([sidx[subj[i]] for i in tr], np.int32)
    noflip = np.array([not (allow_flip(clean[i]) and allow_flip(subj[i])) for i in tr])

    def evtext(rows):
        return utext[np.array([cidx[clean[i]] for i in rows], np.int32)]

    # prompts for sampling (raw strings; tokenized here in both formats)
    rare = ["anchor", "iron anchor", "candelabra", "unlit candelabra", "umbrella", "open umbrella",
            "folded umbrella", "closed umbrella", "trumpet", "golden trumpet", "potted cactus",
            "cactus in a pot", "wizard hat", "witch hat", "sunglasses", "red potion"]
    panel10 = ["red potion", "blue potion", "knight shield", "silver magic staff", "gold coin", "blue sword",
               "gold key", "green gem", "treasure chest", "red mushroom"]
    prompts = {}
    for name, lst in (("rare", rare), ("panel10", panel10), ("eval50", eval50)):
        tt, _ = enc(lst)
        prompts[name] = {p: tt[k].tolist() for k, p in enumerate(lst)}

    np.savez_compressed(OUT / "v3_train.npz", rows=tr.astype(np.int32), text_main=text_main, text_syn=text_syn,
                        has_syn=has_syn, subj_id=subj_id, subj_text=subj_text, noflip=noflip)
    probe_subj_id = np.array([probe.index(subj[i]) for i in probe_rows], np.int16)
    np.savez_compressed(OUT / "v3_eval.npz",
                        probe_rows=probe_rows.astype(np.int32), probe_text=evtext(probe_rows), probe_subj=probe_subj_id,
                        probe_cbucket=np.array([list(quota).index(cbucket(subj[i])) for i in probe_rows], np.int8),
                        iid_rows=iid_rows.astype(np.int32), iid_text=evtext(iid_rows),
                        iid_cbucket=np.array([list(quota).index(b) for b in iid_cb], np.int8),
                        iid_sbucket=np.array([["lt_50", "50_300", "gt_300"].index(sbucket(subj[i])) for i in iid_rows], np.int8),
                        iid_seen_by_v2=np.array([subj[i] in v2_seen_subj and not drop_blind[i] for i in iid_rows]))
    (OUT / "prompts_v3.json").write_text(json.dumps(prompts))
    man = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "text_len": L,
        "text_format": "uint16 CLIP BPE ids: BOS 49406, <=24 ids, EOS 49407, padded with EOS to 26 (openai CLIP tokenizer, lowercase)",
        "v2_format": "ids between BOS and first EOS, +1, zero-padded to 24",
        "row_ids_index": "external token cache (sha256 8de0b0aa5938d1d3...)",
        "rows_total": N, "rows_train": int(len(tr)), "rows_probe_eval": int(len(probe_rows)),
        "rows_iid_eval": int(len(iid_rows)),
        "dropped": {"blind50_phrase_caption_or_subject": int(drop_blind.sum()), "eval50_exact_caption": int(drop_eval.sum()),
                    "probe_subject_rows": int((is_probe_subj & base_ok).sum()), "sha_dupe_of_eval": n_sha,
                    "near_dupe_of_eval_hamming_le4_specific_refs": nd, "near_specific_refs": n_specific, "near_max_match": NEAR_MAX_MATCH, "intra_train_exact_dupes": nx},
        "restored_heldout_subjects": len(held) - len(probe),
        "train_subjects": len(usub), "unique_captions": len(ucap),
        "synonym_rows": int(has_syn.sum()), "noflip_rows": int(noflip.sum()),
        "variants": {"p_subject_caption": P_SUBJ, "p_synonym_if_available": P_SYN, "synonyms": SYN},
        "probe_subjects": [{"subject": s, "rows_total": sfreq[s], "held_in": held[s], "head_word_df": hdf[s],
                            "cbucket": cbucket(s)} for s in probe],
        "buckets": {"concept (head-word doc freq)": list(quota), "subject_freq": ["lt_50", "50_300", "gt_300"],
                    "subject_freq_hist": dict(Counter(sbucket(s) for s in subjects)),
                    "note": "every subject has 96-358 rows, so the requested <50 subject-frequency bucket is empty; "
                            "the rare/mid/common concept buckets use the doc frequency of the subject's head word instead"},
        "iid_counts": {"cbucket": dict(Counter(iid_cb)), "seen_by_v2_subject": int(sum(subj[i] in v2_seen_subj for i in iid_rows))},
        "restored_concepts_check": {s: {"train_rows": int(sum(1 for i in tr if subj[i] == s))} for s in
                                    ["anchor", "candelabra", "wizard hat", "witch hat", "sunglasses", "umbrella", "trumpet"]},
    }
    for f in ("v3_train.npz", "v3_eval.npz", "prompts_v3.json"):
        man["sha256_" + f] = hashlib.sha256((OUT / f).read_bytes()).hexdigest()
    (OUT / "manifest_v3.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "probe_subjects"}, indent=1), t(), flush=True)


if __name__ == "__main__":
    main()
