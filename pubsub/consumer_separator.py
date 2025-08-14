# pubsub/consumer_separator.py
import os, traceback, time
from kafka import KafkaConsumer, KafkaProducer
from config import BOOTSTRAP, TOPIC_JOBS, TOPIC_DLQ, ARTIFACTS_DIR, RUN_MODE, S3_BUCKET, AWS_REGION
from shared.job_models import SoundMindJob
from shared.s3_io import upload_file, presign
import json as _json

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MAX_RETRIES = 3

# If you run the consumer outside repo root, set REPO_ROOT env to that root path.
REPO_ROOT = os.getenv("REPO_ROOT", os.getcwd())

def _repo_relpath(path: str) -> str:
    abs_audio = os.path.abspath(path)
    try:
        return os.path.relpath(abs_audio, start=REPO_ROOT)
    except Exception:
        # Fallback to cwd if REPO_ROOT is mis-set
        return os.path.relpath(abs_audio, start=os.getcwd())

# -------------------------------
# chooser (robust): try Modal, else fall back to local
# -------------------------------
USE_MODAL = (RUN_MODE == "modal")

if USE_MODAL:
    try:
        from modal import Function  # type: ignore
        _remote_sep = Function.from_name("soundmind-diarizer", "run_sep_gpu")  # Modal 1.x

        def run_sep(audio_path: str, job_id: str, params: dict | None = None) -> dict:
            # The Modal container sees the repo at /pkg; send a repo-relative path
            rel_audio = _repo_relpath(audio_path)
            return _remote_sep.remote(
                job_id=job_id,
                rel_audio_path=rel_audio,
                params_json=_json.dumps(params or {})
            )
        print("⚙️  Using Modal (GPU) for separation")
    except Exception as e:
        print(f"⚠️ Modal not configured (RUN_MODE='modal'): {e}\n   → Falling back to local execution.")
        USE_MODAL = False

if not USE_MODAL:
    # Local path: use the unified entrypoint
    from modal_jobs.separator_worker.modal_entry import run_on_accelerator as _run_sep_local

    def run_sep(audio_path: str, job_id: str, params: dict | None = None) -> dict:
        p = dict(params or {})
        p.setdefault("out_dir", os.path.join(ARTIFACTS_DIR, job_id))  # write artifacts locally
        meta = _run_sep_local(audio_path=audio_path, job_id=job_id, params=p)

        # Upload local artifacts to S3
        urls, presigned = [], []
        prefix = f"jobs/{job_id}/separation/"
        for fp in meta.get("artifacts", []):
            if os.path.isfile(fp):
                key = prefix + os.path.basename(fp)
                s3_url = upload_file(fp, S3_BUCKET, key)
                urls.append(s3_url)
                presigned.append(presign(S3_BUCKET, key))
        meta["artifact_urls"] = urls
        meta["presigned_urls"] = presigned
        return meta
    print("⚙️  Using LOCAL execution for separation")

# -------------------------------
# processing + main loop
# -------------------------------
def process(job: SoundMindJob) -> dict:
    return run_sep(job.audio_path, job.job_id, job.params)

def main():
    consumer = KafkaConsumer(
        TOPIC_JOBS,
        bootstrap_servers=BOOTSTRAP,
        group_id="separator_worker",
        value_deserializer=lambda m: _json.loads(m.decode("utf-8")),
        enable_auto_commit=True,
        auto_offset_reset="earliest",
    )
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: _json.dumps(v).encode("utf-8"),
    )
    print(f"👂 Listening on {TOPIC_JOBS} (mode={RUN_MODE})")

    for msg in consumer:
        raw = msg.value
        job = SoundMindJob(**raw)
        model = (job.params or {}).get("model")
        print(f"\n🎯 Job {job.job_id} | task='{job.task}' | model={model}")
        try:
            meta = process(job)
            print(f"✅ [{meta.get('stage','separation').upper()}] Artifacts:")
            for pth in meta.get("artifacts", []):
                print(f"   • {pth}")
            if meta.get("artifacts_dir"):
                print(f"📁 Dir: {meta['artifacts_dir']}")
            dur = (meta.get("metrics") or {}).get("duration_ms", 0)
            print(f"⏱️ Took: {dur} ms  •  Model: {meta.get('model')}")
            if meta.get("presigned_urls"):
                print("🔗 Presigned URLs:")
                for url in meta["presigned_urls"]:
                    print(f"   • {url}")
        except Exception as e:
            print("💥 Error:", e)
            traceback.print_exc()
            job.retry_count += 1
            if job.retry_count <= MAX_RETRIES:
                print(f"🔁 Retry {job.retry_count}/{MAX_RETRIES}")
                time.sleep(1.0 * job.retry_count)
                producer.send(TOPIC_JOBS, job.model_dump())
            else:
                print("🛑 DLQ")
                producer.send(TOPIC_DLQ, job.model_dump())
            producer.flush()

if __name__ == "__main__":
    main()
