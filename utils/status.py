# utils/status.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timezone
from utils.paths import status_path


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def update_status(path: Path, **kwargs: Any) -> Dict[str, Any]:
    data = _read_json(path)
    data.update(kwargs)
    data.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(data, indent=2))
    return data


def init_status(path: Path, job_id: str) -> Dict[str, Any]:
    base = {
        "job_id": job_id,
        "status": "started",
        "stems": [],
        "diarization": {},
        "events": [],
    }
    return update_status(path, **base)

# ✅ New: mark completion/progress for a named stage
def mark_stage(job_id: str, stage: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update per-stage metadata in status.json.
    - If stage == 'diarization', writes to the dedicated top-level key for convenience.
    - Otherwise, stores under 'stages[stage]'.
    """
    path = status_path(job_id)
    if stage == "diarization":
        return update_status(path, diarization=meta)
    # generic bucket for other stages (e.g., 'separation', 'sed', 'enhancer')
    data = _read_json(path)
    stages = data.get("stages", {})
    stages[stage] = meta
    return update_status(path, stages=stages)


# ✅ New: accumulate numeric/runtime metrics
def record_metrics(job_id: str, metrics: Dict[str, float | int]) -> Dict[str, Any]:
    """
    Merge metrics into status.json['metrics'] (non-destructive for other fields).
    Example: record_metrics(job_id, {'diarization_ms': 1234, 'vrAM_MB': 2048})
    """
    path = status_path(job_id)
    data = _read_json(path)
    current = data.get("metrics", {})
    # shallow merge is fine for flat metrics
    current.update(metrics)
    return update_status(path, metrics=current)


# (optional but handy) append stems without dupes
def append_stems(job_id: str, stems: List[str]) -> Dict[str, Any]:
    path = status_path(job_id)
    data = _read_json(path)
    existing = set(data.get("stems", []))
    existing.update(stems)
    return update_status(path, stems=sorted(existing))