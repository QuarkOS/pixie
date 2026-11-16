"""Compatibility entry point; see train/sample_pixie.py (handles P32 and v3)."""
from train.sample_pixie import empty_text, load_clip, load_model, load_state, main, to_rgba  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
