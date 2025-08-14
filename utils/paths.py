# utils/paths.py
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def job_dir(job_id: str) -> Path:
    d = OUTPUTS / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def stems_dir(job_id: str) -> Path:
    d = job_dir(job_id) / "stems"
    d.mkdir(parents=True, exist_ok=True)
    return d


def diarization_dir(job_id: str, impl: str | None = None) -> Path:
    base = job_dir(job_id) / "diarization"
    base.mkdir(parents=True, exist_ok=True)
    if impl:
        sub = base / impl
        sub.mkdir(parents=True, exist_ok=True)
        return sub
    return base

def diar_dir(job_id: str, impl: str | None = None) -> Path:
    return diarization_dir(job_id, impl)

def status_path(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"