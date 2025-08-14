import json, os, traceback, time
from kafka import KafkaConsumer, KafkaProducer
from config import BOOTSTRAP, TOPIC_JOBS, TOPIC_DLQ, ARTIFACTS_DIR, RUN_MODE
from shared.job_models import SoundMindJob

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
            return _remote_sep.remote(job_id, rel_audio, params or {})
        print("⚙️  Using Modal (GPU) for separation")
    except Exception as e:
        print(f"⚠️ Modal not configured (RUN_MODE='modal'): {e}\n   → Falling back to local execution.")
        USE_MODAL = False

if not USE_MODAL:
    from modal_jobs.separator_worker.modal_entry import run_on_accelerator as _run_sep_local

    def run_sep(audio_path: str, job_id: str, params: dict | None = None) -> dict:
        p = dict(params or {})
        p.setdefault("out_dir", os.path.join(ARTIFACTS_DIR, job_id))  # write artifacts locally
        return _run_sep_local(audio_path=audio_path, job_id=job_id, params=p)
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
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        enable_auto_commit=True,
        auto_offset_reset="earliest",
    )
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
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
            for p in meta.get("artifacts", []):
                print(f"   • {p}")
            if meta.get("artifacts_dir"):
                print(f"📁 Dir: {meta['artifacts_dir']}")
            dur = (meta.get("metrics") or {}).get("duration_ms", 0)
            print(f"⏱️ Took: {dur} ms  •  Model: {meta.get('model')}")
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
