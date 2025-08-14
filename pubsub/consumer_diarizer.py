"""
Kafka consumer for the *diarization* stage.

Pattern:
- Listen on TOPIC for SoundMindJob messages (JSON).
- For each job:
    - Log context (job_id, audio_path, task)
    - Call the diarizer worker (pyannote-based for now)
    - Update per-job status and print structured logs
- On failures:
    - Log details and (optionally) push to a DLQ topic
"""

from __future__ import annotations
import json
import os
import sys
import traceback
from typing import Any, Dict

# Kafka client
from kafka import KafkaConsumer, KafkaProducer  # type: ignore

# Structured logging (falls back gracefully if not installed)
try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None

# ---- Project imports (you already have these modules) ----
# paths.py centralizes output directories
from utils.paths import diar_dir
# status.py reads/writes outputs/<job_id>/status.json
from utils.status import mark_stage, record_metrics

# The actual diarization logic lives in the worker module:
# modal_jobs/diarizer_worker/diarizer.py
from modal_jobs.diarizer_worker.diarizer import run_diarization  # you'll implement next

# --------------- Config ---------------
TOPIC = os.getenv("SOUNDMIND_TOPIC", "soundmind_jobs")
DLQ_TOPIC = os.getenv("SOUNDMIND_DLQ_TOPIC", "soundmind_jobs_dlq")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
GROUP_ID = os.getenv("KAFKA_GROUP_ID_DIARIZER", "diarizer_worker")

# You can pin an implementation label for output folder structure
IMPL_NAME = "pyannote"


def get_logger():
    """Return a structured logger; fallback to basic print-like logger."""
    if structlog:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
        )
        return structlog.get_logger("diarizer_consumer")
    # Fallback logger with minimal API
    class FallbackLogger:
        def bind(self, **kwargs): return self
        def info(self, msg, **kw): print("INFO ", msg, kw)
        def warning(self, msg, **kw): print("WARN ", msg, kw, file=sys.stderr)
        def error(self, msg, **kw): print("ERROR", msg, kw, file=sys.stderr)
    return FallbackLogger()


log = get_logger()


def send_to_dlq(producer: KafkaProducer, job: Dict[str, Any], reason: str) -> None:
    """Optional: push the failed job to a DLQ with the error reason attached."""
    payload = {**job, "_error_reason": reason}
    producer.send(DLQ_TOPIC, value=json.dumps(payload).encode("utf-8"))
    producer.flush()


def process_job(job: Dict[str, Any], producer: KafkaProducer | None = None) -> None:
    """
    Process a single job dict produced by your SoundMind producer.

    Expected job fields (based on your producer):
      - job_id: str (uuid)
      - audio_path: str (path on disk)
      - task: Optional[str] (e.g., "diarize" or extra hints)
    """
    job_id = job["job_id"]
    audio_path = job["audio_path"]
    task = job.get("task", "")

    logger = log.bind(job_id=job_id, stage="diarization")
    logger.info("Received job", audio_path=audio_path, task=task)

    try:
        # Ensure output dir exists for this implementation
        out_dir = diar_dir(job_id, impl=IMPL_NAME)

        # ---- Call the worker (pyannote) ----
        # You will implement run_diarization() in modal_jobs/diarizer_worker/diarizer.py
        # It should return a dict of metadata, e.g.:
        # {
        #   "segments_rttm": ".../diarization/pyannote/segments.rttm",
        #   "timeline_json": ".../diarization/pyannote/timeline.json",
        #   "speakers": 2,
        #   "model": "pyannote/speaker-diarization-3.1",
        #   "duration_ms": 1234,
        # }
        meta = run_diarization(
            job_id=job_id,
            audio_path=audio_path,
            impl=IMPL_NAME,
        )

        # ---- Update status.json with stage completion + metadata ----
        mark_stage(
            job_id=job_id,
            stage="diarization",
            meta={
                "impl": IMPL_NAME,
                "speakers": meta.get("speakers"),
                "segments_rttm": meta.get("segments_rttm"),
                "timeline_json": meta.get("timeline_json"),
                "model": meta.get("model"),
            },
        )

        # Optional: record timing metrics for observability
        if "duration_ms" in meta:
            record_metrics(job_id, {"diarization_ms": meta["duration_ms"]})

        logger.info(
            "Diarization complete",
            speakers=meta.get("speakers"),
            segments_rttm=meta.get("segments_rttm"),
        )

    except Exception as e:
        err = "".join(traceback.format_exception(e))
        log.error("Diarization failed", job_id=job_id, error=str(e))
        if producer is not None:
            send_to_dlq(producer, job, reason=str(e))
        # Re-raise if you want the process to crash; otherwise swallow to keep consuming.
        # raise


def main() -> None:
    log.info("🔌 [DIARIZER] Listening", topic=TOPIC, bootstrap=BOOTSTRAP, group_id=GROUP_ID)

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        enable_auto_commit=True,
        auto_offset_reset="earliest",
    )

    # DLQ producer is optional; if Kafka is single-node you can omit this
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP)

    try:
        for msg in consumer:
            job = msg.value  # already JSON-deserialized
            process_job(job, producer=producer)
    except KeyboardInterrupt:
        log.info("👋 Shutting down diarizer consumer.")
    finally:
        try:
            producer.flush()
            producer.close()
        except Exception:
            pass
        consumer.close()


if __name__ == "__main__":
    main()
