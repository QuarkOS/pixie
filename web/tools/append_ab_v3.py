#!/usr/bin/env python3
"""Append same-model v3 (condboost checkpoint) A/B pairs for a prompt list.

Both sides of each new pair are the live chat checkpoint (worker/config_v3.json
`ckpt`, model key `condboost` in api/_lib/abdata.js). Each pair gets two different
seeds so the sprites are not the same sample. Existing pairs are never rewritten.
The previous pair count is recorded in AB.versions so older done-bitsets stay valid.

Sprites are sampled on the GPU host. This machine does not need the checkpoint
for a dry run.

Dry run (no writes, no torch):

    python3 tools/append_ab_v3.py --prompts tools/ab_prompt_swap.json --dry-run

Generate and append (run on the 3080 Ti, from a checkout that can read the
checkpoint path in worker/config_v3.json):

    python3 tools/append_ab_v3.py --prompts tools/ab_prompt_swap.json

That writes new public/ab/<id>.png sheets (32x16, sprite A then sprite B) and
appends the pairs to api/_lib/abdata.js. Copy those two outputs onto the app
host and restart the app. Do not point this at a live data directory.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "export const AB = "
MATCHUP = "condboost_vs_condboost"
MODEL = "condboost"
SEED_MOD = 2**31 - 1


def load_ab(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(PREFIX):
        raise SystemExit(f"{path} does not start with {PREFIX!r}")
    body = text[len(PREFIX):].strip()
    if body.endswith(";"):
        body = body[:-1]
    data = json.loads(body)
    if not isinstance(data.get("pairs"), list) or not isinstance(data.get("versions"), list):
        raise SystemExit("abdata.js is missing pairs or versions")
    return data


def save_ab(path: Path, data: dict) -> None:
    raw = PREFIX + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(raw, encoding="utf-8")
    os.replace(tmp, path)


def load_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        doc = json.loads(text)
        if isinstance(doc, list):
            prompts = doc
        elif isinstance(doc.get("swap"), list):
            prompts = [row["new"] for row in doc["swap"]]
        elif isinstance(doc.get("replacements"), list):
            prompts = doc["replacements"]
        else:
            raise SystemExit(f"{path} has no swap/replacements list")
    else:
        prompts = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                prompts.append(line)
    out = []
    seen = set()
    for p in prompts:
        if not isinstance(p, str):
            raise SystemExit("prompt list must be strings")
        p = " ".join(p.split())
        if len(p) < 2 or len(p) > 80:
            raise SystemExit(f"prompt length out of range: {p!r}")
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    if not out:
        raise SystemExit("prompt list is empty")
    return out


def retired_prompts(path: Path) -> set[str]:
    if path.suffix != ".json":
        return set()
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc.get("swap"), list):
        return {row["old"] for row in doc["swap"]}
    return set()


def two_seeds() -> list[int]:
    a = secrets.randbelow(SEED_MOD - 1) + 1
    b = secrets.randbelow(SEED_MOD - 1) + 1
    while b == a:
        b = secrets.randbelow(SEED_MOD - 1) + 1
    return [a, b]


def new_id(used: set[str]) -> str:
    for _ in range(100):
        pid = secrets.token_hex(6)
        if pid not in used:
            used.add(pid)
            return pid
    raise SystemExit("could not mint a unique pair id")


def plan(data: dict, prompts: list[str], per: int, matchup: str) -> tuple[list[dict], list[str]]:
    meta = next((m for m in data.get("matchups") or [] if m.get("key") == matchup), None)
    if not meta:
        raise SystemExit(f"matchup {matchup} is not in abdata.js")
    if meta.get("a") != meta.get("b"):
        raise SystemExit(f"matchup {matchup} is not same-model ({meta.get('a')} vs {meta.get('b')})")
    pool = meta.get("pool") or matchup
    mode = meta.get("mode") or "cfg2"
    model = meta["a"]
    have: dict[str, int] = {}
    used = set()
    for pair in data["pairs"]:
        used.add(pair["id"])
        if pair.get("m") == matchup:
            have[pair.get("prompt")] = have.get(pair.get("prompt"), 0) + 1
    fresh: list[dict] = []
    skipped: list[str] = []
    for prompt in prompts:
        need = per - have.get(prompt, 0)
        if need <= 0:
            skipped.append(prompt)
            continue
        for _ in range(need):
            seeds = two_seeds()
            if seeds[0] == seeds[1]:
                raise SystemExit("seed sampler produced identical sides")
            fresh.append({
                "id": new_id(used),
                "prompt": prompt,
                "kind": "ab",
                "m": matchup,
                "pool": pool,
                "ck": [model, model],
                "seeds": seeds,
                "mode": mode,
            })
    return fresh, skipped


def sample_two(gen, prompt: str, seeds: list[int]):
    """One sprite per seed, in seed order. Not the reward-v7 best-of-N picker."""
    import numpy as np
    import torch
    import worker_v3 as W

    n = len(seeds)
    row = W.trim(W.t26(gen.tok, prompt), gen.TL)
    blank = [W.M.CLIP_BOS] + [W.M.CLIP_EOS] * (gen.TL - 1)
    h = gen._encode([row] * n)
    hu = gen._encode([blank] * n)
    seed_t = torch.tensor(seeds, dtype=torch.int64, device=gen.device)
    ct = gen.cond_tensor(None, n)
    with torch.no_grad():
        img = gen.sampler.sample(h, hu, seeds=seed_t, cond=ct)
    if gen.device.type == "cuda":
        torch.cuda.synchronize()
    ind = img.cpu().numpy()
    return [W.rgba_from_indices(np.asarray(r, dtype=np.int64).reshape(-1), gen.palette) for r in ind]


def sprites_differ(a, b) -> bool:
    import numpy as np
    return not np.array_equal(a, b)


def write_sheet(path: Path, left, right) -> None:
    from PIL import Image
    import numpy as np
    sheet = Image.fromarray(np.concatenate([np.asarray(left), np.asarray(right)], axis=1), "RGBA")
    if sheet.size != (32, 16):
        raise SystemExit(f"expected a 32x16 sheet, got {sheet.size}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "PNG")


def generate(pairs: list[dict], png_dir: Path) -> None:
    os.environ.setdefault("CHAT_WORKER_KEY", "ab-append-local")
    sys.path.insert(0, str(ROOT / "worker"))
    import worker_v3 as W
    # Single samples, not the chat best-of-64 scorer. Skip the reward head.
    W.SCORER_CFG = "feat_linear"
    W.USE_GRAPH = False
    gen = W.Gen()
    try:
        for pair in pairs:
            seeds = list(pair["seeds"])
            sprites = None
            for _ in range(5):
                sprites = sample_two(gen, pair["prompt"], seeds)
                if sprites_differ(sprites[0], sprites[1]):
                    break
                seeds[1] = two_seeds()[1]
                while seeds[1] == seeds[0]:
                    seeds[1] = two_seeds()[1]
            else:
                raise SystemExit(f"identical sprites for {pair['prompt']} after retries")
            pair["seeds"] = seeds
            write_sheet(png_dir / f"{pair['id']}.png", sprites[0], sprites[1])
            print(f"wrote {pair['id']} {pair['prompt']} seeds {seeds[0]} {seeds[1]}", flush=True)
    finally:
        del gen


def main() -> int:
    ap = argparse.ArgumentParser(description="Append condboost-vs-condboost A/B pairs")
    ap.add_argument("--prompts", default=str(ROOT / "tools" / "ab_prompt_swap.json"), help="JSON swap file or a text prompt list")
    ap.add_argument("--abdata", default=str(ROOT / "api" / "_lib" / "abdata.js"))
    ap.add_argument("--png-dir", default=str(ROOT / "public" / "ab"))
    ap.add_argument("--matchup", default=MATCHUP)
    ap.add_argument("--per", type=int, default=18, help="pairs per prompt (the live same-model matchup already uses 18)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    args = ap.parse_args()
    if args.per < 1 or args.per > 64:
        raise SystemExit("--per must be 1..64")
    prompt_path = Path(args.prompts)
    prompts = load_prompts(prompt_path)
    blocked = retired_prompts(prompt_path) & set(prompts)
    if blocked:
        raise SystemExit("refusing to add retired prompts: " + ", ".join(sorted(blocked)))
    ab_path = Path(args.abdata)
    before = ab_path.read_bytes()
    data = load_ab(ab_path)
    model = (data.get("models") or {}).get(MODEL) or {}
    fresh, skipped = plan(data, prompts, args.per, args.matchup)
    summary = {
        "dry_run": bool(args.dry_run),
        "matchup": args.matchup,
        "model": MODEL,
        "sha": model.get("sha"),
        "ckpt": model.get("path"),
        "prompts": prompts,
        "per": args.per,
        "existing_pairs": len(data["pairs"]),
        "new_pairs": len(fresh),
        "skipped_full": skipped,
        "versions_prefix": len(data["pairs"]) if fresh else None,
        "example": fresh[0] if fresh else None,
        "png_dir": str(Path(args.png_dir)),
        "abdata": str(ab_path),
    }
    print("PLAN " + json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        if ab_path.read_bytes() != before:
            raise SystemExit("dry-run mutated abdata.js")
        print(f"dry-run: would append {len(fresh)} pairs "
              f"({len(prompts) - len(skipped)} prompts x up to {args.per}) "
              f"onto {args.matchup}; versions would record {len(data['pairs'])}", flush=True)
        return 0
    if not fresh:
        print("nothing to append", flush=True)
        return 0
    generate(fresh, Path(args.png_dir))
    # Record the pre-append length so tokens minted against today's pool stay valid.
    old_len = len(data["pairs"])
    if old_len not in data["versions"]:
        data["versions"].append(old_len)
    data["pairs"].extend(fresh)
    save_ab(ab_path, data)
    print(f"appended {len(fresh)} pairs; abdata.js now {len(data['pairs'])} pairs; versions {data['versions']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
