# modal_jobs/separator_worker/nemo_separator.py
from pathlib import Path
import shutil

def separate(audio_path: str, job_id: str, params: dict, out_dir: str) -> str:
    """
    Minimal, dependency-light separation:
      - Copies input to original.wav
      - Simple energy-envelope gate -> speech.wav
      - Residual -> child_laughter.wav (placeholder "noise/residual" stem)

    This is a pragmatic stand-in until a full NeMo pipeline is wired.
    It guarantees non-empty outputs for typical speech clips.
    """
    import soundfile as sf
    import numpy as np

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Read input (2D array for consistent channel handling)
    y, sr = sf.read(audio_path, always_2d=True)
    if y.ndim == 1:
        y = y[:, None]  # (frames, 1)

    # Save original.wav (copy if possible; otherwise write PCM wav)
    orig_path = out / "original.wav"
    try:
        shutil.copyfile(audio_path, orig_path)
    except Exception:
        sf.write(str(orig_path), y, sr)

    # Mono for envelope analysis
    mono = y.mean(axis=1)

    # Energy envelope via moving average (~50 ms)
    win = max(1, int(sr * 0.05))
    env = np.convolve(np.abs(mono), np.ones(win, dtype=np.float64) / win, mode="same")

    # Threshold using median-based rule (robust for quiet clips)
    eps = 1e-8
    thr = float(np.median(env) * 1.5 + eps)

    # Binary mask → lightly smoothed (~30 ms) to avoid choppiness
    speech_mask = (env > thr).astype(np.float32)
    smooth_win = max(1, int(sr * 0.03))
    speech_mask = np.convolve(
        speech_mask, np.ones(smooth_win, dtype=np.float32) / smooth_win, mode="same"
    )
    speech_mask = np.clip(speech_mask, 0.0, 1.0)

    # Apply mask to all channels
    speech = (y * speech_mask[:, None]).astype("float32")
    residual = (y - speech).astype("float32")

    # Write stems
    sf.write(str(out / "speech.wav"), speech, sr)
    sf.write(str(out / "child_laughter.wav"), residual, sr)

    return str(out)
