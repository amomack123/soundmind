# modal_jobs/diarizer_worker/modal_entry.py
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Iterable, Tuple

# Your existing diarizer impl (no params kwarg)
from .diarizer import run_diarization as _run_diarization

import torch
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.core import Annotation


# --- helpers -----------------------------------------------------------------

def _ensure_wav(audio_path: str) -> str:
    """
    Keep things predictable in slim images by standardizing to mono/16k WAV.
    Only used by the fallback path below.
    """
    import subprocess
    import tempfile

    if audio_path.lower().endswith(".wav"):
        return audio_path

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", tmp.name]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp.name


def _iter_with_label(ann: Annotation) -> Iterable[Tuple[Any, Any, Any]]:
    """
    Wrapper to avoid Pyright/Mypy complaining about itertracks()' union type.
    At runtime, with yield_label=True, it's (segment, track, label).
    """
    for seg, trk, lab in ann.itertracks(yield_label=True):  # type: ignore[misc]
        yield seg, trk, lab


def _annotation_to_json(ann: Annotation) -> Dict[str, Any]:
    """
    Convert PyAnnote Annotation to a light JSON structure:
      {"segments": [{"speaker": "SPEAKER_00", "start": 0.32, "end": 1.58, "duration": 1.26}, ...]}
    """
    segments: List[Dict[str, Any]] = []
    for segment, _track, label in _iter_with_label(ann):
        start, end = float(segment.start), float(segment.end)
        segments.append(
            {"speaker": str(label), "start": start, "end": end, "duration": end - start}
        )
    segments.sort(key=lambda x: x["start"])
    return {"segments": segments}


def _write_rttm(annotation: Annotation, rttm_path: str, uri: str = "audio") -> None:
    """
    Minimal SPEAKER RTTM writer to avoid extra deps.
    """
    lines = []
    for segment, _track, label in _iter_with_label(annotation):
        onset = segment.start
        dur = segment.duration
        spk = str(label)
        # SPEAKER <uri> 1 <onset> <duration> <ortho> <stype> <name> <conf> <slat>
        lines.append(f"SPEAKER {uri} 1 {onset:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>")
    Path(rttm_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- main entrypoints ---------------------------------------------------------

def run_on_accelerator(
    audio_path: str,
    job_id: str,
    params: Dict[str, Any] | None = None,
) -> dict:
    """
    Unified accelerator entry.
    Returns (keeps your original public shape):
      {
        "stage": "diarization",
        "artifacts": [segments_rttm, timeline_json],
        "metrics": {"duration_ms": int},
        "model": "<model_id>",
        "job_id": "<job_id>",
        ... (full diarizer meta + stats/status)
      }
    """
    t0 = time.time()
    params = dict(params or {})
    out_dir = Path(params.get("out_dir", f"/tmp/outputs/{job_id}/diarization"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- call your existing implementation first ---
    base_meta = _run_diarization(job_id=job_id, audio_path=audio_path)

    # Expect these keys from your impl (may be None/absent)
    segments_rttm = base_meta.get("segments_rttm")
    timeline_json = base_meta.get("timeline_json")

    # If your implementation didn't produce files, generate them here via pyannote (fallback).
    if not segments_rttm or not timeline_json:
        device = os.environ.get("DEVICE", "cpu")
        # Try multiple environment variable names for the HF token
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        print(f"Using device: {device}")
        # Debug token access - don't print full token, just check if it exists
        if hf_token:
            print(f"HF token found: {hf_token[:5]}...")
        else:
            print("WARNING: No Hugging Face token found in environment variables!")

        try:
            if not hf_token:
                raise ValueError("No Hugging Face token found! Cannot load PyAnnote models.")
                
            print(f"Loading PyAnnote pipeline with token {hf_token[:5]}...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            if device == "cuda":
                pipeline.to(torch.device("cuda"))  # <- fix typing & runtime
        except Exception as e:
            import traceback
            error_msg = f"Failed to load PyAnnote pipeline: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            raise RuntimeError(error_msg) from e

        wav_path = _ensure_wav(audio_path)
        diar_kwargs: Dict[str, Any] = {}
        # Optional knobs if provided by caller; safe to ignore otherwise
        if "min_speakers" in params:
            diar_kwargs["min_speakers"] = int(params["min_speakers"])
        if "max_speakers" in params:
            diar_kwargs["max_speakers"] = int(params["max_speakers"])

        ann: Annotation = pipeline(wav_path, **diar_kwargs)

        # Persist artifacts
        segments_rttm = segments_rttm or str(out_dir / "diarization.rttm")
        timeline_json = timeline_json or str(out_dir / "diarization.json")
        _write_rttm(ann, segments_rttm, uri=Path(wav_path).stem)
        json.dump(_annotation_to_json(ann), open(timeline_json, "w"), ensure_ascii=False, indent=2)

        # Surface minimal stats alongside your meta
        base_meta.setdefault("stats", {})
        base_meta["stats"].update(
            {
                "device": device,
                "num_speakers": len(set(ann.labels())),
                "audio_duration_sec": float(sf.info(wav_path).duration),
            }
        )

        # Note in meta that we used the fallback
        base_meta.setdefault("notes", [])
        base_meta["notes"].append("pyannote-fallback-generated-artifacts")

    # Build public shape (preserve your keys, add ours)
    artifacts = [p for p in (segments_rttm, timeline_json) if p]
    model_id = base_meta.get("model", "pyannote-default")

    result = {
        "stage": "diarization",
        "artifacts": artifacts,
        "metrics": {"duration_ms": int((time.time() - t0) * 1000)},
        "model": model_id,
        "job_id": job_id,
        **base_meta,
    }

    # Always drop a status.json for quick probing
    try:
        (out_dir / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception:
        pass

    return result


# Optional shim to keep backwards compatibility with any older imports
def diarize(audio_path: str, job_id: str, params: Dict[str, Any] | None = None) -> dict:
    return run_on_accelerator(audio_path=audio_path, job_id=job_id, params=params)
