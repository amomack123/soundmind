from pathlib import Path
import shutil

def separate(audio_path: str, job_id: str, params: dict, out_dir: str) -> str:
    """
    Replace this stub with actual NeMo speech enhancement / separation.
    For now, copy input into out_dir and create fake stems.
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    # TODO: integrate real NeMo pipeline (DPRNN/Conv‑TasNet or NeMo ASR+denoise chain)
    shutil.copy2(audio_path, p / "original.wav")
    # Fake outputs to keep the pipeline moving
    (p / "speech.wav").write_bytes(b"")
    (p / "child_laughter.wav").write_bytes(b"")
    return str(p)