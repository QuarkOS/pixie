"""Read-only byte identity check before loading arrays, weights or a GPU."""
import argparse
import json
from pathlib import Path

from train.pixie32.data import Holdouts, verify_holdout_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--blind50", type=Path, required=True)
    parser.add_argument("--eval50", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema") != "pixie32-v1":
            raise ValueError("Unsupported dataset schema")
        policy = Holdouts(args.blind50, args.eval50)
        verify_holdout_bytes(manifest, policy)
    except (OSError, ValueError):
        print(json.dumps({"byte_identity": False, "action": "supply original frozen holdout files; do not edit or reserialize them or the manifest"}))
        return 1
    print(json.dumps({"byte_identity": True, "holdout_hashes": policy.hashes, "scope": "holdout schema and raw file bytes only; arrays and weights not loaded"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
