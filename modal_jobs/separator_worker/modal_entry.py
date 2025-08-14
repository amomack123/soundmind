# modal_jobs/separator_worker/modal_entry.py
from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
import time

# Your existing local separator
from .nemo_separator import separate as _separate_local


def _default_out_dir(job_id: str) -> Path:
    return Path("outputs") / job_id / "separation"


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
    }
    """
    t0 = time.time()
    params = params or {}

    # Resolve output dir
    out_dir = params.get("out_dir")
    out_path = Path(out_dir) if out_dir else _default_out_dir(job_id)
    out_path.mkdir(parents=True, exist_ok=True)

    # Call the existing local function (same signature you already had)
    stems_dir = _separate_local(
        audio_path=audio_path,
        job_id=job_id,
        params=params,
        out_dir=str(out_path),
    )

    # Collect artifacts (wav/ogg/flac) if present
    sd = Path(stems_dir)
    artifacts: List[str] = []
    if sd.exists():
        for ext in ("*.wav", "*.flac", "*.ogg", "*.mp3"):
            artifacts.extend([str(p) for p in sd.glob(ext)])

    meta = {
        "stage": "separation",
        "artifacts": artifacts or [str(sd)],  # at least include the dir
        "artifacts_dir": str(sd),
        "metrics": {"duration_ms": int((time.time() - t0) * 1000)},
        "model": params.get("model", "nemo-default"),
        "job_id": job_id,
    }
    return meta


# --- Backward compatibility (keep old signature/behavior) ---
def separate(audio_path: str, job_id: str, params: dict, out_dir: str) -> str:
    """
    Legacy thin wrapper. Returns the stems directory path string (old behavior).
    Prefer run_on_accelerator(...) for new code.
    """
    # Ensure legacy out_dir is honored
    p = dict(params or {})
    p["out_dir"] = out_dir
    meta = run_on_accelerator(audio_path=audio_path, job_id=job_id, params=p)
    return meta["artifacts_dir"]
