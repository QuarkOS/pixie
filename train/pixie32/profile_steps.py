"""Profile bounded training updates from an explicit full P32 checkpoint."""
import argparse
from pathlib import Path

from train.pixie32 import train_p32
from train.pixie32.clip_cache import idle_device


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    supplied = args.trainer_args[1:] if args.trainer_args[:1] == ["--"] else args.trainer_args
    owned = {"--out", "--smoke", "--smoke-steps", "--profile-steps", "--profile-warmup"}
    if any(token.split("=", 1)[0] in owned for token in supplied):
        parser.error("profiler owns output, smoke and profile settings")
    if args.out.exists() or not 1 <= args.steps <= 200 or not 0 <= args.warmup <= 100:
        parser.error("use a new output folder, 1..200 steps and 0..100 warmup updates")
    config = train_p32.parser().parse_args(supplied + [
        "--out", str(args.out), "--smoke", "--smoke-steps", str(args.warmup + args.steps),
        "--profile-steps", str(args.steps), "--profile-warmup", str(args.warmup)])
    if config.resume is None and config.branch_from is None:
        parser.error("supply --resume or --branch-from with a full P32 checkpoint")
    idle_device(config.device)
    train_p32.run(config)


if __name__ == "__main__":
    main()
