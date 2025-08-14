# modal_jobs/diarizer_worker/diarizer.py
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, cast

import torch
import soundfile as sf  # type: ignore
from pyannote.core import Annotation, Segment
from pyannote.audio import Pipeline

from utils.paths import diar_dir

try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None

DEFAULT_MODEL_ID = os.getenv("DIAR_MODEL_ID", "pyannote/speaker-diarization-3.1")
# Build a real torch.device from env; default to CUDA if available
DEFAULT_DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def get_logger():
    if structlog:
        structlog.configure(
            processors=[structlog.processors.TimeStamper(fmt="iso"),
                        structlog.processors.JSONRenderer()],
        )
        return structlog.get_logger("diarizer_worker")
    class FallbackLogger:
        def bind(self, **kwargs): return self
        def info(self, msg, **kw): print("INFO ", msg, kw)
        def warning(self, msg, **kw): print("WARN ", msg, kw)
        def error(self, msg, **kw): print("ERROR", msg, kw)
    return FallbackLogger()


log = get_logger()


# ---------- I/O helpers ----------

def _audio_duration_s(audio_path: str) -> float:
    info = sf.info(audio_path)
    return float(info.frames) / float(info.samplerate)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _write_rttm(annotation: Annotation, file_id: str, out_path: Path) -> None:
    lines = []
    tracks = cast(Iterable[Tuple[Segment, object, object]], annotation.itertracks(yield_label=True))
    for segment, _track, speaker in tracks:
        start = segment.start
        dur = segment.duration
        spk = str(speaker)
        lines.append(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA>")
    _atomic_write_text(out_path, "\n".join(lines) + ("\n" if lines else ""))

def _write_timeline_json(annotation: Annotation, out_path: Path) -> None:
    segments = []
    tracks = cast(Iterable[Tuple[Segment, object, object]], annotation.itertracks(yield_label=True))
    for segment, _track, speaker in tracks:
        segments.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "speaker": str(speaker),
        })
    _atomic_write_text(out_path, json.dumps({"segments": segments}, indent=2))


# ---------- Core ----------

def _load_pipeline(model_id: str, device_str: str) -> Pipeline:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Missing HF_TOKEN env var for pyannote models. "
            "Create a token at https://huggingface.co/settings/tokens and export HF_TOKEN=..."
        )

    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)

    # Convert string like "cuda", "cuda:0", "cpu" to torch.device for type safety
    try:
        torch_device = torch.device(device_str)
        try:
            pipeline.to(torch_device)  # type: ignore[arg-type]
        except Exception:
            # Some pipeline parts may not fully move; proceed anyway.
            pass
    except Exception:
        # If device parsing fails, just proceed on default device.
        pass

    return pipeline


def _run_pyannote_diarization(audio_path: str, model_id: str, device_str: str) -> Annotation:
    pipeline = _load_pipeline(model_id, device_str)
    # Runs VAD → embeddings → clustering; returns Annotation
    diarization: Annotation = pipeline(audio_path)
    return diarization


def run_diarization(job_id: str, audio_path: str, impl: str = "pyannote") -> Dict[str, Any]:
    """
    Run diarization on `audio_path`, write RTTM & timeline JSON under job outputs,
    and return metadata consumed by the Kafka consumer.
    """
    logger = log.bind(job_id=job_id, stage="diarization", impl=impl)
    audio_path = str(audio_path)
    file_id = Path(audio_path).stem

    out_base = diar_dir(job_id, impl=impl)
    rttm_path = out_base / "segments.rttm"
    timeline_path = out_base / "timeline.json"

    model_id = DEFAULT_MODEL_ID
    device_str = DEFAULT_DEVICE

    logger.info("Starting diarization", audio=audio_path, model=model_id, device=device_str)
    t0 = time.time()

    if impl != "pyannote":
        raise ValueError(f"Unsupported impl='{impl}' (expected 'pyannote')")

    annotation = _run_pyannote_diarization(audio_path, model_id, device_str)

    _write_rttm(annotation, file_id=file_id, out_path=rttm_path)
    _write_timeline_json(annotation, out_path=timeline_path)

    speaker_labels = list(annotation.labels())
    speakers = len(speaker_labels)
    duration_s = _audio_duration_s(audio_path)
    duration_ms = int((time.time() - t0) * 1000)

    logger.info(
        "Diarization done",
        speakers=speakers,
        duration_s=round(duration_s, 3),
        runtime_ms=duration_ms,
        rttm=str(rttm_path),
    )

    return {
        "segments_rttm": str(rttm_path),
        "timeline_json": str(timeline_path),
        "speakers": speakers,
        "speaker_labels": speaker_labels,
        "model": model_id,
        "device": device_str,
        "audio_duration_s": duration_s,
        "duration_ms": duration_ms,
    }
