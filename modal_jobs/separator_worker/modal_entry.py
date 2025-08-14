# modal_jobs/separator_worker/modal_entry.py
from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
import time
import os

# Your existing local separator (we'll update nemo_separator below)
from .nemo_separator import separate as _separate_local


def _default_out_dir(job_id: str) -> Path:
    return Path("outputs") / job_id / "separation"


def _finalize_stem(path: str, ref_audio: str, min_ms: int = 500) -> bool:
    """
    Ensure a stem file exists and is not empty.
    If missing or <= ~header size, write `min_ms` of silence using ref_audio's sr/ch.
    Returns True if we wrote silence (i.e., file was empty/missing).
    """
    try:
        if os.path.exists(path) and os.path.getsize(path) > 100:  # > header-only WAV
            return False
    except Exception:
        # proceed to write a valid file
        pass

    import soundfile as sf
    import numpy as np

    # Read ref to get sr/ch; always_2d=True to infer channels robustly
    y, sr = sf.read(ref_audio, always_2d=True)
    ch = y.shape[1] if y.ndim == 2 else 1
    frames = max(1, int(sr * (min_ms / 1000.0)))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((frames, ch), dtype="float32"), sr)
    return True


def _stem_metrics(path: str) -> dict | None:
    """Return {'duration_sec': float, 'rms': float} for a WAV (or None if unreadable)."""
    try:
        import soundfile as sf
        import numpy as np
        y, sr = sf.read(path, always_2d=True)
        if sr <= 0 or y.size == 0:
            return {"duration_sec": 0.0, "rms": 0.0}
        duration_sec = y.shape[0] / float(sr)
        # RMS over all channels
        rms = float(np.sqrt((y.astype("float64") ** 2).mean()))
        return {"duration_sec": round(duration_sec, 6), "rms": round(rms, 8)}
    except Exception:
        return None


def run_on_accelerator(
    audio_path: str,
    job_id: str,
    params: Dict[str, Any] | None = None,
) -> dict:
    """
    Unified entry for accelerator work. Returns a metadata dict:
      {
        "stage": "separation",
        "artifacts": [paths...],
        "artifacts_dir": "<dir>",
        "metrics": {"duration_ms": int},
        "model": "<model_id>",
        "job_id": "<job_id>",
        "warnings": [ ... ],
        "stem_metrics": { "<file>": {"duration_sec":..., "rms":...}, ... }
      }
    """
    t0 = time.time()
    params = params or {}

    # Resolve output dir
    out_dir = params.get("out_dir")
    out_path = Path(out_dir) if out_dir else _default_out_dir(job_id)
    out_path.mkdir(parents=True, exist_ok=True)

    # Call your local function (NeMo or fallback; implemented in nemo_separator.py below)
    stems_dir = _separate_local(
        audio_path=audio_path,
        job_id=job_id,
        params=params,
        out_dir=str(out_path),
    )

    # Collect artifacts
    sd = Path(stems_dir)
    artifacts: List[str] = []
    if sd.exists():
        for ext in ("*.wav", "*.flac", "*.ogg", "*.mp3"):
            artifacts.extend([str(p) for p in sd.glob(ext)])

    # Ensure WAVs are non-empty; write short silence if needed (500 ms)
    wrote_silence_for: List[str] = []
    for p in list(artifacts):
        if os.path.splitext(p)[1].lower() == ".wav":
            if _finalize_stem(p, ref_audio=audio_path, min_ms=500):
                wrote_silence_for.append(os.path.basename(p))

    # Per-stem metrics
    stem_metrics: Dict[str, dict] = {}
    for p in artifacts:
        if os.path.splitext(p)[1].lower() == ".wav":
            m = _stem_metrics(p)
            if m is not None:
                stem_metrics[os.path.basename(p)] = m

    meta = {
        "stage": "separation",
        "artifacts": artifacts or [str(sd)],  # at least include the dir
        "artifacts_dir": str(sd),
        "metrics": {"duration_ms": int((time.time() - t0) * 1000)},
        "model": params.get("model", "nemo"),
        "job_id": job_id,
        "stem_metrics": stem_metrics,
    }

    if wrote_silence_for:
        meta["warnings"] = [
            f"{name} was empty; wrote 500ms of silence" for name in wrote_silence_for
        ]

    return meta


# --- Backward compatibility (keep old signature/behavior) ---
def separate(audio_path: str, job_id: str, params: dict, out_dir: str) -> str:
    """
    Legacy thin wrapper. Returns the stems directory path string (old behavior).
    Prefer run_on_accelerator(...) for new code.
    """
    p = dict(params or {})
    p["out_dir"] = out_dir
    meta = run_on_accelerator(audio_path=audio_path, job_id=job_id, params=p)
    return meta["artifacts_dir"]
