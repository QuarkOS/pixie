"""Verified native-32 caches and frozen holdout policy."""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The Blind-50 / Eval-50 holdout files are not part of the public tree. Put your own
# 50-phrase JSON lists in <repo>/holdout/ or point PIXIE_HOLDOUT_DIR at them.
HOLDOUT_DIR = Path(os.environ.get("PIXIE_HOLDOUT_DIR") or Path(__file__).resolve().parents[2] / "holdout")
DEFAULT_BLIND50 = HOLDOUT_DIR / "blind-50.json"
DEFAULT_EVAL50 = HOLDOUT_DIR / "eval-50.json"
HOLDOUTS_PRESENT = DEFAULT_BLIND50.is_file() and DEFAULT_EVAL50.is_file()
HOLDOUTS_MISSING_REASON = "Blind-50/Eval-50 holdout files are not shipped; set PIXIE_HOLDOUT_DIR"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def normalize_holdout(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join("".join(" " if unicodedata.category(char).startswith("P") else char
                            for char in text).split())


class Holdouts:
    def __init__(self, blind50=DEFAULT_BLIND50, eval50=DEFAULT_EVAL50):
        self.hashes = {}
        phrases = set()
        for name, file in (("blind50", blind50), ("eval50", eval50)):
            path = Path(file)
            if not path.is_file():
                raise ValueError(f"Required {name} file is missing")
            try:
                values = json.loads(path.read_text(encoding="utf-8-sig"))
            except (ValueError, UnicodeError) as error:
                raise ValueError(f"Required {name} must be a JSON array of 50 lowercase phrases") from error
            if not isinstance(values, list) or len(values) != 50:
                raise ValueError(f"Required {name} must contain exactly 50 phrases")
            if any(not isinstance(item, str) or not normalize(item) or item != item.lower() for item in values):
                raise ValueError(f"Required {name} must contain nonempty lowercase phrases")
            cleaned = [normalize_holdout(item) for item in values]
            if not all(cleaned) or len(set(cleaned)) != 50:
                raise ValueError(f"Required {name} must contain 50 unique phrases")
            phrases.update(cleaned)
            self.hashes[name] = sha256(path)
        self.pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + r")(?!\w)")

    def matches(self, *texts):
        return any(self.pattern.search(normalize_holdout(text)) is not None for text in texts)


@dataclass
class Dataset:
    arrays: dict
    ids: list
    manifest: dict
    fingerprint: str
    palette: np.ndarray
    records: list
    holdouts: Holdouts


def verify_holdout_bytes(manifest, policy):
    if not isinstance(manifest, dict) or not isinstance(manifest.get("holdouts"), dict):
        raise ValueError("Dataset manifest must include frozen holdout byte hashes")
    expected = manifest.get("holdouts", {})
    changed = [name for name, digest in policy.hashes.items() if expected.get(name) != digest]
    if set(expected) != set(policy.hashes) or changed:
        raise ValueError("Frozen holdout byte hashes differ for " + ", ".join(changed or ["manifest keys"])
                         + "; supply the original Blind-50 and Eval-50 files used to build this cache. "
                         "Do not reformat holdouts or edit the dataset manifest to bypass this check")


def load_dataset(dataset_dir, blind50=DEFAULT_BLIND50, eval50=DEFAULT_EVAL50):
    root = Path(dataset_dir)
    policy = Holdouts(blind50, eval50)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "pixie32-v1":
        raise ValueError("Unsupported dataset schema")
    verify_holdout_bytes(manifest, policy)
    files = manifest["files"]
    required = {"ids.txt", "records.json", "palette.npy", "tokens.npy", "text.npy", "subj_text.npy", "subject_id.npy", "weight.npy", "noflip.npy", "split.npy", "cond.npy"}
    if not required <= set(files):
        raise ValueError("Dataset manifest is missing required files")
    for name, digest in files.items():
        if Path(name).name != name or sha256(root / name) != digest:
            raise ValueError(f"Cache integrity mismatch: {name}")
    ids = (root / "ids.txt").read_text(encoding="utf-8").splitlines()
    if len(ids) != manifest["rows"] or len(set(ids)) != len(ids):
        raise ValueError("Invalid ordered dataset ids")
    records = json.loads((root / "records.json").read_text(encoding="utf-8"))
    if [r["id"] for r in records] != ids:
        raise ValueError("Image/caption cache ids do not match ordered records")
    arrays = {}
    for name in manifest["arrays"]:
        if Path(name).name != name or name not in files or not name.endswith(".npy"):
            raise ValueError("Invalid array manifest entry")
        if manifest["array_ids_sha256"].get(name) != files["ids.txt"]:
            raise ValueError(f"Cache row identity mismatch: {name}")
        value = np.load(root / name, allow_pickle=False, mmap_mode="r")
        if len(value) != len(ids):
            raise ValueError(f"Cache row count mismatch: {name}")
        arrays[name[:-4]] = value
    shapes = {"tokens": (1024,), "text": (18,), "subj_text": (18,), "subject_id": (), "weight": (), "noflip": (), "split": (), "cond": (6,)}
    for name, tail in shapes.items():
        if arrays[name].shape != (len(ids), *tail):
            raise ValueError(f"Invalid cache shape: {name}")
    for name in ("tokens", "text", "subj_text"):
        if arrays[name].dtype != np.uint16:
            raise ValueError(f"Invalid cache dtype: {name}")
    for name in ("text", "subj_text"):
        value = arrays[name]
        if np.any(value[:, 0] != 49406) or np.any(value[:, -1] != 49407) or np.any((value[:, 1:] >= 49406) & (value[:, 1:] != 49407)):
            raise ValueError(f"Invalid CLIP cache ids: {name}")
        ended = np.maximum.accumulate(value[:, 1:] == 49407, axis=1)
        if np.any(ended & (value[:, 1:] != 49407)):
            raise ValueError(f"Invalid CLIP EOS padding: {name}")
    if np.any(arrays["tokens"] > 4096) or np.any(~np.isin(arrays["split"], (0, 1, 2))):
        raise ValueError("Invalid pixel or split ids")
    if not np.isfinite(arrays["weight"]).all() or np.any(arrays["weight"] <= 0):
        raise ValueError("Invalid sampling weights")
    if arrays["noflip"].dtype != np.bool_ or not np.issubdtype(arrays["subject_id"].dtype, np.integer) or np.any(arrays["subject_id"] < 0):
        raise ValueError("Invalid subject or flip cache")
    cond = arrays["cond"]
    if not np.issubdtype(cond.dtype, np.integer) or np.any((cond[:, :3] < 1) | (cond[:, :3] > 4097)) or np.any((cond[:, 3:] < 1) | (cond[:, 3:] > np.array([3, 2, 3]))):
        raise ValueError("Invalid conditioning cache")
    for record in records:
        if policy.matches(record["caption"], record["subject"], record["trimmed_caption"], record["trimmed_subject"]):
            raise ValueError("Frozen holdout appears in dataset cache")
    train = arrays["split"] == 0
    if not train.any():
        raise ValueError("Dataset has no training rows")
    for left, right in ((0, 1), (1, 2)):
        a = {r["sheet"] for r, split in zip(records, arrays["split"]) if split == left}
        b = {r["sheet"] for r, split in zip(records, arrays["split"]) if split == right}
        if a & b:
            raise ValueError("Sheet leakage across splits")
    train_caps = {normalize(r["caption"]) for r, split in zip(records, arrays["split"]) if split == 0}
    eval_caps = {normalize(r["caption"]) for r, split in zip(records, arrays["split"]) if split != 0}
    train_subs = {normalize(r["subject"]) for r, split in zip(records, arrays["split"]) if split == 0}
    probe_subs = {normalize(r["subject"]) for r, split in zip(records, arrays["split"]) if split == 2}
    if train_caps & eval_caps or train_subs & probe_subs:
        raise ValueError("Caption or unseen-subject split leakage")
    palette = np.load(root / "palette.npy", allow_pickle=False)
    if palette.shape != (4097, 3) or palette.dtype != np.uint8:
        raise ValueError("Palette must be uint8 [4097,3]")
    return Dataset(arrays, ids, manifest, sha256(manifest_path), palette, records, policy)
