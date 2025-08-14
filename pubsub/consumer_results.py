"""
Results consumer: listens for completion events from workers (separator/diarizer/etc.)
and prints them immediately (optionally filters by job_id and/or saves to disk).

Message schema (example):
{
  "job_id": "uuid",
  "stage": "diarization" | "separation" | "...",
  "ok": true,
  "meta": {
    "artifacts": [...],
    "artifact_urls": [...],
    "presigned_urls": [...],
    "stats": {...},
    "model": "...",
    "device": "cuda"
  }
}
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from kafka import KafkaConsumer  # type: ignore

# Structured logging (falls back gracefully if not installed)
try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None


# ---------- Config ----------
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
RESULTS_TOPIC = os.getenv("SOUNDMIND_RESULTS_TOPIC", "soundmind_results")
GROUP_ID = os.getenv("KAFKA_GROUP_ID_RESULTS", "results_sink")


def get_logger():
    """Return a structured logger; fallback to basic print-like logger."""
    if structlog:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
        )
        return structlog.get_logger("results_consumer")
    # Fallback logger with minimal API
    class FallbackLogger:
        def bind(self, **kwargs): return self
        def info(self, msg, **kw): print("INFO ", msg, kw)
        def warning(self, msg, **kw): print("WARN ", msg, kw, file=sys.stderr)
        def error(self, msg, **kw): print("ERROR", msg, kw, file=sys.stderr)
    return FallbackLogger()


log = get_logger()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Listen for SoundMind results (no polling needed).")
    ap.add_argument("--job", help="Filter to a specific job_id", default=None)
    ap.add_argument("--save", help="Path to save JSONL of results (e.g., outputs/results.jsonl)", default=None)
    ap.add_argument(
        "--format",
        choices=["pretty", "raw", "compact"],
        default="pretty",
        help="Output format: pretty (human), raw (one JSON per line), compact (single-line summary)",
    )
    ap.add_argument(
        "--from-beginning",
        action="store_true",
        help="Read the topic from the beginning instead of the latest offsets.",
    )
    return ap.parse_args()


def format_pretty(msg: Dict[str, Any]) -> str:
    job_id = msg.get("job_id")
    stage = msg.get("stage")
    ok = msg.get("ok")
    meta = msg.get("meta", {})
    stats = meta.get("stats", {})
    model = meta.get("model")
    device = meta.get("device")
    art = meta.get("artifacts", [])
    s3 = meta.get("artifact_urls", [])
    presigned = meta.get("presigned_urls", [])
    when = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    lines = [
        f"🟢 {when}  job={job_id} stage={stage} ok={ok}",
        f"    model={model} device={device} stats={stats}",
        f"    artifacts ({len(art)}):",
    ]
    for a in art:
        lines.append(f"      - {a}")
    if s3:
        lines.append(f"    s3 urls ({len(s3)}):")
        for u in s3:
            lines.append(f"      - {u}")
    if presigned:
        lines.append(f"    presigned ({len(presigned)}):")
        for p in presigned:
            lines.append(f"      - {p}")
    return "\n".join(lines)


def format_compact(msg: Dict[str, Any]) -> str:
    job_id = msg.get("job_id")
    stage = msg.get("stage")
    ok = msg.get("ok")
    meta = msg.get("meta", {})
    stats = meta.get("stats", {})
    model = meta.get("model")
    device = meta.get("device")
    n_art = len(meta.get("artifacts", []) or [])
    n_s3 = len(meta.get("artifact_urls", []) or [])
    return f"{job_id} {stage} ok={ok} model={model} device={device} artifacts={n_art} s3={n_s3} stats={stats}"


def maybe_save(path: Optional[str], msg: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg) + "\n")


def main() -> None:
    args = parse_args()
    log.info("🔎 Listening for results", topic=RESULTS_TOPIC, bootstrap=BOOTSTRAP, group_id=GROUP_ID)

    consumer = KafkaConsumer(
        RESULTS_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        enable_auto_commit=True,
        auto_offset_reset="earliest" if args.from_beginning else "latest",
    )

    try:
        for msg in consumer:
            payload: Dict[str, Any] = msg.value

            # Optional filtering by job_id
            if args.job and payload.get("job_id") != args.job:
                continue

            # Print
            if args.format == "pretty":
                print(format_pretty(payload))
            elif args.format == "compact":
                print(format_compact(payload))
            else:  # raw
                print(json.dumps(payload))

            # Save (JSONL)
            maybe_save(args.save, payload)

    except KeyboardInterrupt:
        log.info("👋 Shutting down results consumer.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()