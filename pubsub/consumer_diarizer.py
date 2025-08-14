"""
Kafka consumer for the *diarization* stage.

Pattern:
- Listen on TOPIC for SoundMindJob messages (JSON).
- For each job:
    - Log context (job_id, audio_path, task/params)
    - Execute diarization (Modal GPU/CPU, or Local) via unified interface
    - Update per-job status and print structured logs
- On failures:
    - Log details and (optionally) push to a DLQ topic
"""

from __future__ import annotations
import json
import os
import sys
import traceback
from typing import Any, Dict, Optional

# Kafka client
from kafka import KafkaConsumer, KafkaProducer  # type: ignore

# Structured logging (falls back gracefully if not installed)
try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None

# ---- Project utilities ----
from utils.paths import diar_dir
from utils.status import mark_stage, record_metrics

# Local fallback (CPU/GPU depending on env DEVICE)
from modal_jobs.diarizer_worker.modal_entry import run_on_accelerator as run_local


# --------------- Config ---------------
TOPIC = os.getenv("SOUNDMIND_DIAR_TOPIC", "soundmind_diarization_jobs")
RESULTS_TOPIC = os.getenv("SOUNDMIND_RESULTS_TOPIC", "soundmind_results")  # optional
DLQ_TOPIC = os.getenv("SOUNDMIND_DLQ_TOPIC", "soundmind_jobs_dlq")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
GROUP_ID = os.getenv("KAFKA_GROUP_ID_DIARIZER", "diarizer_worker")

# Execution mode: "modal" | "local" (can be overridden per-job too)
RUN_MODE_DEFAULT = os.getenv("RUN_MODE", "modal").lower()

# Modal resolution (matches your modal_app.py)
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "soundmind-diarizer")
MODAL_FN_GPU = os.getenv("MODAL_FN_DIAR_GPU", "run_diar_gpu")
MODAL_FN_CPU = os.getenv("MODAL_FN_DIAR_CPU", "run_diar_cpu")

# Implementation label (for local out_dir pathing)
IMPL_NAME = os.getenv("DIAR_IMPL", "pyannote")


# --------------- Logging ---------------
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


# --------------- Helpers ---------------
def send_to_dlq(producer: KafkaProducer, job: Dict[str, Any], reason: str) -> None:
    """Optional: push the failed job to a DLQ with the error reason attached."""
    payload = {**job, "_error_reason": reason}
    producer.send(DLQ_TOPIC, value=json.dumps(payload).encode("utf-8"))
    producer.flush()


def maybe_publish_result(producer: Optional[KafkaProducer], result: Dict[str, Any]) -> None:
    """Optionally publish result metadata to a results topic for API/UI ingestion."""
    if not producer or not RESULTS_TOPIC:
        return
    try:
        producer.send(RESULTS_TOPIC, value=json.dumps(result).encode("utf-8"))
        producer.flush()
    except Exception:
        # Non-fatal
        pass


def _run_modal(job_id: str, rel_audio_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call Modal function (GPU preferred). Falls back to CPU function if GPU is unavailable
    or if caller set params['device']="cpu".
    """
    from modal import Function

    device_pref = str(params.get("device", "cuda")).lower()
    fn_name = MODAL_FN_GPU if device_pref == "cuda" else MODAL_FN_CPU
    try:
        fn = Function.from_name(MODAL_APP_NAME, fn_name)
    except Exception:
        # Fallback to CPU function lookup
        fn = Function.from_name(MODAL_APP_NAME, MODAL_FN_CPU)

    # Ensure out_dir for consistency of return shape (worker may ignore in Modal)
    params = dict(params or {})
    params.setdefault("out_dir", f"/tmp/outputs/{job_id}/diarization")

    # Execute remote (Modal functions expect params_json: str)
    meta = fn.remote(
        job_id=job_id,
        rel_audio_path=rel_audio_path,
        params_json=json.dumps(params),
    )
    return dict(meta or {})


def _run_local(job_id: str, rel_or_abs_audio_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Local execution path. Respects DEVICE env for GPU if available.
    """
    from pathlib import Path as _P
    p = dict(params or {})
    p.setdefault("out_dir", str(diar_dir(job_id, impl=IMPL_NAME)))
    audio_abs = str(_P(rel_or_abs_audio_path).resolve())
    return run_local(audio_abs, job_id, p)


# --------------- Core ---------------
def process_job(job: Dict[str, Any], producer: Optional[KafkaProducer] = None) -> None:
    """
    Process a single job dict produced by your SoundMind producer.

    Expected job fields (compatible with separation):
      - job_id: str (uuid)
      - audio_path: str (path; local or repo-relative if Modal image adds /pkg)
      - params: Optional[dict] (e.g., {"min_speakers": 1, "max_speakers": 4, "device": "cuda"})
      - run_mode: Optional["modal"|"local"] (overrides RUN_MODE_DEFAULT)
      - task: Optional[str] (freeform text)
    """
    job_id = job["job_id"]
    rel_audio_path = job["audio_path"]
    params = dict(job.get("params", {}) or {})
    task = job.get("task", "")
    run_mode = str(job.get("run_mode", RUN_MODE_DEFAULT)).lower()

    logger = log.bind(job_id=job_id, stage="diarization", run_mode=run_mode)
    logger.info("Received job", audio_path=rel_audio_path, task=task, params=params)

    try:
        # Ensure local output dir creates predictable paths (even for Modal we keep parity in returned meta)
        out_dir = diar_dir(job_id, impl=IMPL_NAME)
        os.makedirs(out_dir, exist_ok=True)

        # --- Execute ---
        if run_mode == "modal":
            meta = _run_modal(job_id=job_id, rel_audio_path=rel_audio_path, params=params)
        else:
            meta = _run_local(job_id=job_id, rel_or_abs_audio_path=rel_audio_path, params=params)

        # Normalize & extract fields
        artifacts = list(meta.get("artifacts", []))
        presigned = list(meta.get("presigned_urls", []))
        artifact_urls = list(meta.get("artifact_urls", []))
        stats = dict(meta.get("stats", {}))
        model = meta.get("model") or meta.get("model_id") or "pyannote-default"
        device = meta.get("device") or os.environ.get("DEVICE", "cpu")

        # ---- Update status.json with stage completion + metadata ----
        mark_stage(
            job_id=job_id,
            stage="diarization",
            meta={
                "impl": IMPL_NAME,
                "model": model,
                "device": device,
                "artifacts": artifacts,
                "artifact_urls": artifact_urls,
                "presigned_urls": presigned,
                "stats": stats,
                # Backward-compat keys if your UI expects them:
                "segments_rttm": next((a for a in artifacts if a.endswith(".rttm")), None),
                "timeline_json": next((a for a in artifacts if a.endswith(".json")), None),
            },
        )

        # Optional: record timing metrics for observability
        duration_ms = meta.get("metrics", {}).get("duration_ms") or stats.get("elapsed_ms") or None
        if duration_ms is not None:
            record_metrics(job_id, {"diarization_ms": int(duration_ms)})

        logger.info(
            "Diarization complete",
            num_artifacts=len(artifacts),
            s3=len(artifact_urls),
            presigned=len(presigned),
            device=device,
            model=model,
        )

        logger.info(
            "Artifacts uploaded",
            examples_presigned=presigned[:2],  # show first couple links
        )

        # Optionally publish the whole result to a results topic (for API/UI)
        result_envelope = {
            "job_id": job_id,
            "stage": "diarization",
            "ok": True,
            "meta": meta,
        }
        maybe_publish_result(producer, result_envelope)

    except Exception as e:
        err_txt = "".join(traceback.format_exception(e))
        log.error("Diarization failed", job_id=job_id, error=str(e))
        if producer is not None:
            send_to_dlq(producer, job, reason=str(e))
        # Also publish a failure envelope if using results topic
        fail_env = {"job_id": job_id, "stage": "diarization", "ok": False, "error": str(e)}
        maybe_publish_result(producer, fail_env)
        # Do not re-raise to keep consuming; change to `raise` if you prefer crash-on-fail.


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

    # DLQ/Results producer is optional; if Kafka is single-node you can still use it
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
