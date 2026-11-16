"""Generate a held-out Krea 2 animation-sheet baseline.

The default mode writes a deterministic request plan and never contacts a
server.  ``--execute`` sends one 2x2 request at a time to an already running
sd.cpp server and keeps every successful sheet.  Generated sheets are review
artifacts, not normalized training examples.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .common import DEFAULT_BLIND50, DEFAULT_EVAL50, holdout_policy, new_output, sha256
from train.pixie32.data import normalize_holdout


SCHEMA = "pixie-animation-baseline-v1"
PROMPT_SCHEMA = "pixie-animation-baseline-prompts-v1"
GRID = 2
FRAME_COUNT = 4
MAGENTA = [255, 0, 255]
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class PilotInputError(ValueError):
    """Raised when a frozen prompt or provenance input is unsafe or incomplete."""


def _safe_url(value: str, *, name: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise PilotInputError(f"{name} must be an HTTP(S) URL without credentials")
    if parsed.query or parsed.fragment:
        raise PilotInputError(f"{name} must not contain a query or fragment")
    return value


def _server_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise PilotInputError("--server must be an HTTP(S) URL without credentials")
    if parsed.query or parsed.fragment:
        raise PilotInputError("--server must not contain a query or fragment")
    return value.rstrip("/")


def _validate_digest(value: str, name: str) -> str:
    if not _SHA256.fullmatch(value or ""):
        raise PilotInputError(f"{name} must be a 64-character SHA-256 digest")
    return value.lower()


def _load_json(path: Path, label: str) -> tuple[Any, bytes, str]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotInputError(f"{label} is not valid UTF-8 JSON") from exc
    return value, raw, hashlib.sha256(raw).hexdigest()


def _subjects(value: Any, label: str) -> list[str]:
    if isinstance(value, list):
        result = value
    elif isinstance(value, dict) and isinstance(value.get("subjects"), list):
        result = value["subjects"]
    else:
        raise PilotInputError(f"{label} must be a JSON array or an object with a subjects array")
    if not result or any(not isinstance(item, str) or not normalize_holdout(item) for item in result):
        raise PilotInputError(f"{label} must contain nonempty subject strings")
    normalized = [normalize_holdout(item) for item in result]
    if len(set(normalized)) != len(normalized):
        raise PilotInputError(f"{label} contains duplicate subjects")
    return result


def _prompt_metadata(value: Any) -> tuple[list[dict[str, Any]], str, list[str]]:
    if not isinstance(value, dict) or value.get("schema", PROMPT_SCHEMA) != PROMPT_SCHEMA:
        raise PilotInputError(f"prompts must have schema {PROMPT_SCHEMA}")
    heldout = value.get("heldout_subjects")
    audit = value.get("audit_set")
    if heldout is not None:
        heldout_subjects = _subjects(heldout, "prompts.heldout_subjects")
        audit_id = str(value.get("audit_set_id") or (audit.get("id") if isinstance(audit, dict) else audit or ""))
    else:
        heldout_subjects = []
        audit_id = str(value.get("audit_set_id") or (audit.get("id") if isinstance(audit, dict) else audit or ""))
    if not audit_id:
        raise PilotInputError("prompts must declare heldout_subjects or an audit_set id")
    if isinstance(audit, dict) and audit.get("frozen") is False:
        raise PilotInputError("prompts audit_set must be frozen")
    entries = value.get("prompts")
    if not isinstance(entries, list) or not entries:
        raise PilotInputError("prompts must contain a nonempty prompts array")
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise PilotInputError(f"prompts[{index}] must be an object")
        subject = item.get("subject")
        action = item.get("action")
        if not isinstance(subject, str) or not normalize_holdout(subject):
            raise PilotInputError(f"prompts[{index}].subject must be a nonempty string")
        if not isinstance(action, str) or not normalize_holdout(action):
            raise PilotInputError(f"prompts[{index}].action must be a nonempty string")
        direction = item.get("direction", "unknown")
        if not isinstance(direction, str) or not direction.strip():
            raise PilotInputError(f"prompts[{index}].direction must be a string")
        loop = item.get("loop", False)
        if not isinstance(loop, bool):
            raise PilotInputError(f"prompts[{index}].loop must be boolean")
        parsed.append({"subject": subject.strip(), "action": action.strip(), "direction": direction.strip(), "loop": loop})
    if heldout_subjects:
        allowed = {normalize_holdout(item) for item in heldout_subjects}
        missing = sorted({normalize_holdout(item["subject"]) for item in parsed} - allowed)
        if missing:
            raise PilotInputError("prompt subjects are outside the declared heldout subject set")
    return parsed, audit_id, heldout_subjects


def _caption(item: dict[str, Any]) -> str:
    direction = item["direction"]
    direction_part = "" if direction.casefold() in {"", "unknown", "none"} else f", facing {direction}"
    loop_part = ", looping" if item["loop"] else ", non-looping"
    return (
        f"pixie_anim, pixel art sprite sheet, 2x2 grid, 4 frames, {item['subject']}, "
        f"{item['action']}{direction_part}{loop_part}, fixed camera and scale, "
        "frames in chronological order from left to right then top to bottom, magenta background"
    )


def _safe_error(error: BaseException, server: str | None = None) -> dict[str, str]:
    text = str(error)
    if server:
        text = text.replace(server, "<server>")
    text = re.sub(r"https?://[^\s]+", "<url>", text)
    text = re.sub(r"[A-Za-z]:[\\/][^\s]+", "<path>", text)
    text = re.sub(r"(?<![A-Za-z0-9_])/[^\s]+", "<path>", text)
    return {"error_type": type(error).__name__, "error": text[:240]}


def _request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return response.status, json.loads(body or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read() or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = {}
        return exc.code, body


Transport = Callable[[str, dict[str, Any] | None], tuple[int, dict[str, Any]]]


def _generate_one(
    server: str,
    body: dict[str, Any],
    *,
    transport: Transport | None = None,
    timeout: float = 900.0,
    poll_interval: float = 0.05,
) -> tuple[bytes, int | None, str]:
    request = transport or (lambda url, payload: _request_json(url, payload, min(timeout, 60.0)))
    code, submitted = request(server + "/sdcpp/v1/img_gen", body)
    if code != 202 or not isinstance(submitted, dict):
        raise RuntimeError(f"submit HTTP {code}")
    poll = submitted.get("poll_url") or (f"/sdcpp/v1/jobs/{submitted['id']}" if submitted.get("id") else None)
    if not isinstance(poll, str):
        raise RuntimeError("submit response omitted a poll URL")
    poll_url = urllib.parse.urljoin(server + "/", poll)
    base = urllib.parse.urlparse(server)
    parsed_poll = urllib.parse.urlparse(poll_url)
    base_path = base.path.rstrip("/")
    if (parsed_poll.scheme, parsed_poll.netloc) != (base.scheme, base.netloc) or not (
        parsed_poll.path == base_path or parsed_poll.path.startswith(base_path + "/")
    ):
        raise RuntimeError("poll URL is outside the configured server origin")
    started = time.monotonic()
    while time.monotonic() - started <= timeout:
        code, job = request(poll_url, None)
        if code != 200 or not isinstance(job, dict):
            raise RuntimeError(f"poll HTTP {code}")
        status = job.get("status")
        if status == "completed":
            try:
                encoded = job["result"]["images"][0]["b64_json"]
                png = base64.b64decode(encoded, validate=True)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise RuntimeError("completed response omitted a valid image") from exc
            actual = job.get("seed")
            result = job.get("result")
            image = result.get("images", [{}])[0] if isinstance(result, dict) and isinstance(result.get("images"), list) and result.get("images") else {}
            if actual is None and isinstance(result, dict):
                actual = result.get("seed")
            if actual is None and isinstance(image, dict):
                actual = image.get("seed")
                if actual is None and isinstance(image.get("metadata"), dict):
                    actual = image["metadata"].get("seed")
            if isinstance(actual, bool) or not isinstance(actual, int):
                actual = None
            return png, actual, "server_returned" if actual is not None else "requested_only"
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"job {status}")
        time.sleep(poll_interval)
    raise TimeoutError("job timeout")


def _verify_png(png: bytes, size: int) -> None:
    from PIL import Image

    try:
        with Image.open(__import__("io").BytesIO(png)) as image:
            if image.format != "PNG" or image.size != (size, size):
                raise PilotInputError(f"server image must be a {size}x{size} PNG")
    except PilotInputError:
        raise
    except Exception as exc:
        raise PilotInputError("server image is not a readable PNG") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(
    *,
    prompts_path: Path,
    training_subjects_path: Path,
    output_path: Path,
    count: int,
    seed: int,
    model_sha256: str,
    lora_sha256: str,
    server: str | None = None,
    model: str | None = None,
    lora: str | None = None,
    model_source_url: str | None = None,
    lora_source_url: str | None = None,
    blind50: Path | None = None,
    eval50: Path | None = None,
    lora_multiplier: float = 1.0,
    steps: int = 5,
    size: int = 512,
    execute: bool = False,
    transport: Transport | None = None,
    timeout: float = 900.0,
) -> dict[str, Any]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or count > 10000:
        raise PilotInputError("count must be between 1 and 10000")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PilotInputError("seed must be an integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps != 5:
        raise PilotInputError("the baseline pilot uses exactly 5 Turbo steps")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % GRID:
        raise PilotInputError("size must be a positive multiple of 2")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise PilotInputError("timeout must be finite and positive")
    model_sha256 = _validate_digest(model_sha256, "--model-sha256")
    lora_sha256 = _validate_digest(lora_sha256, "--lora-sha256")
    if model_source_url:
        model_source_url = _safe_url(model_source_url, name="--model-source-url")
    if lora_source_url:
        lora_source_url = _safe_url(lora_source_url, name="--lora-source-url")
    if execute:
        if not server or not model or not lora:
            raise PilotInputError("--execute requires --server, --model and --lora")
        server = _server_url(server)
    elif server:
        server = _server_url(server)
    prompts, prompts_raw, prompts_digest = _load_json(Path(prompts_path), "prompts")
    entries, audit_id, heldout_subjects = _prompt_metadata(prompts)
    if count > len(entries):
        raise PilotInputError("--n cannot exceed the number of frozen prompt entries")
    training, training_raw, training_digest = _load_json(Path(training_subjects_path), "training-subjects")
    training_subjects = _subjects(training, "training-subjects")
    train_norm = {normalize_holdout(item) for item in training_subjects}
    prompt_norm = {normalize_holdout(item["subject"]) for item in entries}
    heldout_norm = {normalize_holdout(item) for item in heldout_subjects}
    if train_norm & (prompt_norm | heldout_norm):
        raise PilotInputError("held-out prompt subjects overlap frozen training subjects")
    holdouts = holdout_policy(blind50 or DEFAULT_BLIND50, eval50 or DEFAULT_EVAL50)
    captions = [_caption(entry) for entry in entries]
    for index, entry in enumerate(entries):
        fields = [entry["subject"], entry["action"], entry["direction"], captions[index], *heldout_subjects]
        if holdouts.matches(*fields):
            raise PilotInputError("prompt matches Blind-50 or Eval-50")
    if isinstance(lora_multiplier, bool) or not isinstance(lora_multiplier, (int, float)) or not math.isfinite(lora_multiplier) or lora_multiplier <= 0:
        raise PilotInputError("lora multiplier must be positive")
    input_paths = [Path(prompts_path).resolve(), Path(training_subjects_path).resolve()]
    output_candidate = Path(output_path).resolve()
    for source in input_paths:
        source_parent = source.parent
        if output_candidate == source_parent or source_parent in output_candidate.parents:
            raise PilotInputError("output must be outside the frozen input directories")
    output = new_output(Path(output_path))
    (output / "sheets").mkdir()
    (output / "prompts.json").write_bytes(prompts_raw)
    (output / "training_subjects.json").write_bytes(training_raw)
    common = {
        "schema": SCHEMA,
        "purpose": "animation_baseline_eval",
        "split": "test",
        "review_status": "unreviewed",
        "auto_accepted": False,
        "grid": [GRID, GRID],
        "frame_count": FRAME_COUNT,
        "frame_order": "row_major",
        "cell_size": [size // GRID, size // GRID],
        "sheet_size": [size, size],
        "background": {"type": "flat", "rgb": MAGENTA},
        "model_sha256": model_sha256,
        "lora_sha256": lora_sha256,
        "weight_identity_verification": "operator_supplied_hashes; loaded_server_weights_not_verified",
        "holdouts": holdouts.hashes,
        "prompts_sha256": prompts_digest,
        "training_subjects_sha256": training_digest,
        "audit_set_id": audit_id,
        "heldout_subjects": heldout_subjects,
        "requested_count": count,
        "requested_seed": seed,
        "sampler": {
            "sample_method": "euler",
            "sample_steps": steps,
            "custom_sigmas": [1.0, 0.92664734, 0.82570166, 0.67798672, 0.44119975, 0.0],
            "guidance": {"txt_cfg": 1.0},
        },
        "vae_tiling_params": {"enabled": False},
        "steps": steps,
        "lora_multiplier": lora_multiplier,
        "seed_semantics": "server_returned when supplied, otherwise requested_only",
    }
    if model_source_url:
        common["model_source_url"] = model_source_url
    if lora_source_url:
        common["lora_source_url"] = lora_source_url
    _write_json(output / "plan.json", {
        **common,
        "execute_requested": execute,
        "requests": [{"index": index, "prompt": _caption(entries[index]), "requested_seed": seed + index,
                      "status": "planned"} for index in range(count)],
    })
    requests_path = output / "requests.jsonl"
    events_path = output / "events.jsonl"
    requests_path.write_text("", encoding="utf-8")
    events_path.write_text("", encoding="utf-8")
    for index in range(count):
        item = entries[index]
        requested_seed = seed + index
        prompt = _caption(item)
        body: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": "text, labels, border, frame numbers",
            "width": size,
            "height": size,
            "seed": requested_seed,
            "batch_count": 1,
            "sample_params": {
                "sample_method": "euler",
                "sample_steps": steps,
                "custom_sigmas": common["sampler"]["custom_sigmas"],
                "guidance": {"txt_cfg": 1.0},
            },
            "vae_tiling_params": {"enabled": False},
            "preview": "none",
            "output_format": "png",
            "embed_image_metadata": False,
        }
        if lora:
            body["lora"] = [{"path": lora, "multiplier": lora_multiplier}]
        if model:
            body["model"] = model
        record: dict[str, Any] = {
            "index": index,
            "subject": item["subject"],
            "action": item["action"],
            "direction": item["direction"],
            "loop": item["loop"],
            "caption": prompt,
            "requested_seed": requested_seed,
            "status": "planned" if not execute else "pending",
            "generation_s": None,
            "request_config": {
                "negative_prompt": "text, labels, border, frame numbers",
                "width": size,
                "height": size,
                "batch_count": 1,
                "sampler": common["sampler"],
                "vae_tiling_params": common["vae_tiling_params"],
                "model_sha256": model_sha256,
                "lora_sha256": lora_sha256,
                "lora_multiplier": lora_multiplier,
            },
        }
        try:
            if execute:
                assert server is not None
                started = time.monotonic()
                png, actual_seed, seed_semantics = _generate_one(
                    server, body, transport=transport, timeout=timeout
                )
                record["generation_s"] = round(time.monotonic() - started, 3)
                _verify_png(png, size)
                relative_name = f"sheets/sheet-{index:06d}.png"
                destination = output / relative_name
                if destination.exists():
                    raise PilotInputError("sheet output already exists")
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".sheet-", suffix=".tmp", delete=False) as temp:
                    temp.write(png)
                    temporary = Path(temp.name)
                os.replace(temporary, destination)
                record.update(
                    {
                        "status": "completed",
                        "sheet": relative_name,
                        "file_sha256": hashlib.sha256(png).hexdigest(),
                        "actual_seed": actual_seed,
                        "seed_semantics": seed_semantics,
                    }
                )
            with requests_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            event = {"index": index, "status": record["status"], "requested_seed": requested_seed}
            if "file_sha256" in record:
                event["file_sha256"] = record["file_sha256"]
        except Exception as error:
            if execute and record["generation_s"] is None:
                record["generation_s"] = round(time.monotonic() - started, 3)
            record.update({"status": "failed", **_safe_error(error, server)})
            with requests_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            event = {"index": index, "status": "failed", **_safe_error(error, server)}
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    manifest = {
        **common,
        "executed": execute,
        "completed_count": 0,
        "failed_count": 0,
        "files": {
            "plan.json": sha256(output / "plan.json"),
            "prompts.json": sha256(output / "prompts.json"),
            "training_subjects.json": sha256(output / "training_subjects.json"),
            "requests.jsonl": sha256(requests_path),
            "events.jsonl": sha256(events_path),
        },
    }
    request_records = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines()]
    manifest["completed_count"] = sum(record.get("status") == "completed" for record in request_records)
    manifest["failed_count"] = sum(record.get("status") == "failed" for record in request_records)
    manifest["holdouts"] = holdouts.hashes
    for sheet in sorted((output / "sheets").glob("*.png")):
        manifest["files"][f"sheets/{sheet.name}"] = sha256(sheet)
    _write_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="Frozen animation audit prompt JSON")
    parser.add_argument("--training-subjects", type=Path, required=True, help="Frozen training-subject JSON")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; it must not exist")
    parser.add_argument("--n", type=int, required=True, dest="count", help="Number of prompt entries to run")
    parser.add_argument("--seed", type=int, required=True, help="First deterministic request seed")
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--lora-sha256", required=True)
    parser.add_argument("--model", help="Server-side model identifier or path, not written to provenance")
    parser.add_argument("--lora", help="Server-side LoRA identifier or path, not written to provenance")
    parser.add_argument("--model-source-url")
    parser.add_argument("--lora-source-url")
    parser.add_argument("--blind50", type=Path, default=DEFAULT_BLIND50)
    parser.add_argument("--eval50", type=Path, default=DEFAULT_EVAL50)
    parser.add_argument("--lora-mult", type=float, default=1.0)
    parser.add_argument("--steps", type=int, choices=[5], default=5)
    parser.add_argument("--size", type=int, default=512, help="Square sheet size in pixels")
    parser.add_argument("--server", help="Already running sd.cpp server URL")
    parser.add_argument("--execute", action="store_true", help="Send requests; default writes a dry-run plan")
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = run(
            prompts_path=args.prompts,
            training_subjects_path=args.training_subjects,
            output_path=args.output,
            count=args.count,
            seed=args.seed,
            model_sha256=args.model_sha256,
            lora_sha256=args.lora_sha256,
            server=args.server,
            model=args.model,
            lora=args.lora,
            model_source_url=args.model_source_url,
            lora_source_url=args.lora_source_url,
            blind50=args.blind50,
            eval50=args.eval50,
            lora_multiplier=args.lora_mult,
            steps=args.steps,
            size=args.size,
            execute=args.execute,
            timeout=args.timeout,
        )
    except (PilotInputError, ValueError, OSError) as error:
        print(f"pilot failed: {type(error).__name__}: {error}")
        return 2
    print(json.dumps({"output": str(args.output.name), "executed": manifest["executed"], "count": manifest["requested_count"], "failed": manifest["failed_count"]}))
    return 3 if manifest["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
