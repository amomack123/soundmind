# tools/eval_diarization.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, cast

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


def load_rttm_as_annotation(rttm_path: Path) -> Annotation:
    """Parse RTTM (SPEAKER lines) → pyannote.core.Annotation.

    RTTM SPEAKER columns we care about (1-based):
    1: type, 2: file, 3: channel, 4: start, 5: duration, 8: speaker
    """
    ann = Annotation()
    with open(rttm_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0].upper() != "SPEAKER":
                continue
            # indices per NIST RTTM (0-based here)
            # 0 type, 1 file, 2 ch, 3 start, 4 dur, 5 ortho, 6 stype, 7 speaker, ...
            try:
                start = float(parts[3])
                dur = float(parts[4])
                speaker = parts[7]
                seg = Segment(start, start + dur)
                ann[seg] = speaker
            except Exception:
                # skip malformed line
                continue
    return ann


def compute_der(ref_rttm: Path, hyp_rttm: Path, collar: float = 0.25, skip_overlap: bool = False) -> float:
    ref = load_rttm_as_annotation(ref_rttm)
    hyp = load_rttm_as_annotation(hyp_rttm)
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    score = metric(ref, hyp)          # runtime: float; static: float | Details
    der = cast(float, score)          # make the type checker happy
    return der


def main():
    p = argparse.ArgumentParser(description="Evaluate diarization with DER")
    p.add_argument("--ref", type=Path, required=True, help="Reference RTTM")
    p.add_argument("--hyp_py", type=Path, help="Hypothesis RTTM (pyannote)")
    p.add_argument("--hyp_nemo", type=Path, help="Hypothesis RTTM (NeMo)")
    p.add_argument("--collar", type=float, default=0.25)
    p.add_argument("--skip_overlap", action="store_true")
    args = p.parse_args()

    print("\n📏 DER evaluation")
    print(f"Ref:  {args.ref}")
    results: Dict[str, float] = {}

    if args.hyp_py:
        der_py = compute_der(args.ref, args.hyp_py, args.collar, args.skip_overlap)
        results["pyannote"] = der_py
        print(f"pyannote DER: {der_py*100:.2f}%")

    if args.hyp_nemo:
        der_nemo = compute_der(args.ref, args.hyp_nemo, args.collar, args.skip_overlap)
        results["nemo"] = der_nemo
        print(f"NeMo     DER: {der_nemo*100:.2f}%")

    if results:
        best_key = min(results.keys(), key=lambda k: results[k])
        print(f"\n🏆 Best so far: {best_key} ({results[best_key]*100:.2f}% DER)")
    else:
        print("No hypotheses provided.")


if __name__ == "__main__":
    main()
