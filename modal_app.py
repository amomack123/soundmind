# soundmind/modal_app.py
import os
from pathlib import Path
import modal

app = modal.App("soundmind-diarizer")
REPO_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim()
    # 1) Install CUDA-enabled PyTorch from the official PyTorch index
    .run_commands(
        "python -m pip install --upgrade pip",
        # CUDA 12.1 wheels for PyTorch 2.3.1 (+ matching torchvision/torchaudio)
        "python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 "
        "torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1"
    )
    # 2) Install the rest from Modal’s mirror
    .pip_install(
        "numpy>=1.26",
        "pyannote.audio>=3.1",
        "pyannote.core>=5.0",
        "pyannote.metrics>=3.2",
        "soundfile>=0.12",
        "structlog>=24.1",
    )
    # 3) Add your repo LAST
    .add_local_dir(str(REPO_ROOT), remote_path="/pkg")
)

@app.function(image=image, timeout=600, gpu="T4",
              secrets=[modal.Secret.from_name("hf-token")])
def run_sep_gpu(job_id: str, rel_audio_path: str, params: dict | None = None) -> dict:
    os.environ["DEVICE"] = "cuda"
    from modal_jobs.separator_worker.modal_entry import run_on_accelerator
    out_dir = f"/tmp/outputs/{job_id}/separation"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = dict(params or {}); p.setdefault("out_dir", out_dir)
    audio_abs = str(Path("/pkg") / rel_audio_path)
    return run_on_accelerator(audio_path=audio_abs, job_id=job_id, params=p)

@app.function(image=image, timeout=600)
def run_sep_cpu(job_id: str, rel_audio_path: str, params: dict | None = None) -> dict:
    os.environ["DEVICE"] = "cpu"
    from modal_jobs.separator_worker.modal_entry import run_on_accelerator
    out_dir = f"/tmp/outputs/{job_id}/separation"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = dict(params or {}); p.setdefault("out_dir", out_dir)
    audio_abs = str(Path("/pkg") / rel_audio_path)
    return run_on_accelerator(audio_path=audio_abs, job_id=job_id, params=p)
