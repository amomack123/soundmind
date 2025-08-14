# soundmind/modal_app.py
import os
import json
from pathlib import Path
import modal
import json as _json
from botocore.exceptions import ClientError  # for safe S3 tagging

app = modal.App("soundmind-diarizer")
REPO_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libsndfile1")  # needed for audio I/O
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 "
        "torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1"
    )
    .pip_install(
        "numpy>=1.26",
        "huggingface_hub>=0.22.2",
        "pyannote.audio>=3.1",
        "pyannote.core>=5.0",
        "pyannote.metrics>=3.2",
        "soundfile>=0.12",
        "structlog>=24.1",
        "boto3>=1.34",
    )
    .add_local_dir(str(REPO_ROOT), remote_path="/pkg")
)

# Secrets:
# - hf-hub must contain key: HUGGING_FACE_HUB_TOKEN=hf_********
# - aws-creds should contain your AWS access keys
hf_secret = modal.Secret.from_name("hf-hub")
aws_secret = modal.Secret.from_name("aws-creds")


def _ensure_hf_token_env() -> bool:
    """Map HUGGING_FACE_HUB_TOKEN -> HF_TOKEN for worker code expecting HF_TOKEN."""
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token  # ensure legacy name exists
    return bool(token)


def _try_tag_object(s3, bucket: str, key: str, job_id: str, kind: str) -> None:
    """Best-effort S3 object tagging: do not fail the job if tagging is not permitted."""
    try:
        s3.put_object_tagging(
            Bucket=bucket,
            Key=key,
            Tagging={
                "TagSet": [
                    {"Key": "job_id", "Value": job_id},
                    {"Key": "kind", "Value": kind},
                ]
            },
        )
    except ClientError as e:
        # Non-fatal: keep going even if AccessDenied or other tagging issues
        print(f"[warn] could not tag s3://{bucket}/{key}: {e}")


def _upload_and_sign(s3, bucket: str, region: str, job_id: str, kind: str, paths: list[str]) -> dict:
    from pathlib import Path as _P
    import os

    prefix = f"jobs/{job_id}/{kind}/"
    artifact_urls, presigned_urls = [], []
    for fp in paths:
        p = _P(fp)
        if not p.is_file():
            continue
        try:
            if os.path.getsize(fp) <= 100:  # skip header-only/empty wavs
                print(f"[warn] skipping empty artifact: {fp}")
                continue
        except Exception:
            print(f"[warn] could not stat {fp}, skipping")
            continue

        key = prefix + p.name
        s3.upload_file(str(p), bucket, key)
        try:
            _try_tag_object(s3, bucket, key, job_id, kind)  # best-effort tagging
        except Exception:
            pass
        artifact_urls.append(f"s3://{bucket}/{key}")
        presigned_urls.append(
            s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=86400,
            )
        )
    return {"artifact_urls": artifact_urls, "presigned_urls": presigned_urls}



def _write_status(path: str, status: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        _json.dump(status, f, indent=2)


@app.function(
    image=image,
    timeout=600,
    gpu="T4",
    secrets=[hf_secret, aws_secret],
)
def run_sep_gpu(job_id: str, rel_audio_path: str, params_json: str = "{}") -> dict:
    import boto3
    from modal_jobs.separator_worker.modal_entry import run_on_accelerator
    from config import S3_BUCKET, AWS_REGION

    os.environ["DEVICE"] = "cuda"
    _ensure_hf_token_env()

    params = json.loads(params_json) if params_json else {}
    out_dir = f"/tmp/outputs/{job_id}/separation"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = dict(params)
    p.setdefault("out_dir", out_dir)

    audio_abs = str(Path("/pkg") / rel_audio_path)
    meta = run_on_accelerator(audio_path=audio_abs, job_id=job_id, params=p)

    # Upload artifacts to S3 (+best-effort tags)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    uploads = _upload_and_sign(
        s3=s3,
        bucket=S3_BUCKET,
        region=AWS_REGION,
        job_id=job_id,
        kind="separation",
        paths=meta.get("artifacts", []),
    )

    # Write & upload status.json (+best-effort tags)
    status_path = f"/tmp/outputs/{job_id}/separation/status.json"
    status = {
        "job_id": job_id,
        "kind": "separation",
        "model": meta.get("model"),
        "device": "cuda",
        "artifact_count": len(meta.get("artifacts", [])),
        "s3_prefix": f"jobs/{job_id}/separation/",
        "warnings": meta.get("warnings", []),
        "stem_metrics": meta.get("stem_metrics", {}),
    }
    _write_status(status_path, status)
    status_key = f"jobs/{job_id}/separation/status.json"
    s3.upload_file(status_path, S3_BUCKET, status_key)
    _try_tag_object(s3, S3_BUCKET, status_key, job_id, "separation")

    meta["status_url"] = f"s3://{S3_BUCKET}/{status_key}"
    meta["status_presigned"] = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": status_key},
        ExpiresIn=86400,
    )

    meta.update(uploads)
    return meta


@app.function(image=image, timeout=600, secrets=[hf_secret])
def run_sep_cpu(job_id: str, rel_audio_path: str, params_json: str = "{}") -> dict:
    from modal_jobs.separator_worker.modal_entry import run_on_accelerator

    os.environ["DEVICE"] = "cpu"
    _ensure_hf_token_env()

    params = json.loads(params_json) if params_json else {}
    out_dir = f"/tmp/outputs/{job_id}/separation"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = dict(params)
    p.setdefault("out_dir", out_dir)

    audio_abs = str(Path("/pkg") / rel_audio_path)
    return run_on_accelerator(audio_path=audio_abs, job_id=job_id, params=p)


@app.function(
    image=image,
    timeout=600,
    gpu="T4",
    secrets=[hf_secret, aws_secret],
)
def run_diar_gpu(job_id: str, rel_audio_path: str, params_json: str = "{}") -> dict:
    """
    Diarize on GPU. Returns metadata with RTTM + JSON artifacts, and S3 URLs + presigned URLs.
    """
    import boto3
    from config import S3_BUCKET, AWS_REGION
    from modal_jobs.diarizer_worker.modal_entry import run_on_accelerator

    os.environ["DEVICE"] = "cuda"
    _ensure_hf_token_env()

    params = json.loads(params_json) if params_json else {}
    out_dir = f"/tmp/outputs/{job_id}/diarization"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = dict(params)
    p.setdefault("out_dir", out_dir)

    audio_abs = str(Path("/pkg") / rel_audio_path)
    meta = run_on_accelerator(audio_path=audio_abs, job_id=job_id, params=p)

    # Upload artifacts to S3 (+best-effort tags)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    uploads = _upload_and_sign(
        s3=s3,
        bucket=S3_BUCKET,
        region=AWS_REGION,
        job_id=job_id,
        kind="diarization",
        paths=meta.get("artifacts", []),
    )

    # Write & upload status.json (+best-effort tags)
    status_path = f"/tmp/outputs/{job_id}/diarization/status.json"
    status = {
        "job_id": job_id,
        "kind": "diarization",
        "model": meta.get("model"),
        "device": "cuda",
        "artifact_count": len(meta.get("artifacts", [])),
        "s3_prefix": f"jobs/{job_id}/diarization/",
    }
    _write_status(status_path, status)
    status_key = f"jobs/{job_id}/diarization/status.json"
    s3.upload_file(status_path, S3_BUCKET, status_key)
    _try_tag_object(s3, S3_BUCKET, status_key, job_id, "diarization")

    meta["status_url"] = f"s3://{S3_BUCKET}/{status_key}"
    meta["status_presigned"] = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": status_key},
        ExpiresIn=86400,
    )

    meta.update(uploads)
    return meta


@app.function(image=image, timeout=600, secrets=[hf_secret])
def run_diar_cpu(job_id: str, rel_audio_path: str, params_json: str = "{}") -> dict:
    """
    CPU fallback for diarization (slower). Same return shape as GPU.
    """
    from modal_jobs.diarizer_worker.modal_entry import run_on_accelerator

    os.environ["DEVICE"] = "cpu"
    _ensure_hf_token_env()

    params = json.loads(params_json) if params_json else {}
    out_dir = f"/tmp/outputs/{job_id}/diarization"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = dict(params)
    p.setdefault("out_dir", out_dir)

    audio_abs = str(Path("/pkg") / rel_audio_path)
    return run_on_accelerator(audio_path=audio_abs, job_id=job_id, params=p)


# --- simple smoke test to verify HF token is visible in the worker ---
@app.function(image=image, timeout=120, secrets=[hf_secret])
def debug_env() -> str:
    t = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or ""
    return f"Token present: {bool(t)}, prefix: {t[:6]}***" if t else "Token missing"
