import json, os, traceback, time
from kafka import KafkaConsumer, KafkaProducer
from config import BOOTSTRAP, TOPIC_JOBS, TOPIC_DLQ, ARTIFACTS_DIR, RUN_MODE
from shared.job_models import SoundMindJob
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# chooser
if RUN_MODE == "modal":
    from modal_jobs.separator_worker.modal_entry import separate  # remote
else:
    from modal_jobs.separator_worker.nemo_separator import separate  # local

MAX_RETRIES = 3

def process(job: SoundMindJob):
    return separate(
        audio_path=job.audio_path,
        job_id=job.job_id,
        params=job.params,
        out_dir=os.path.join(ARTIFACTS_DIR, job.job_id),
    )

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
        print(f"\n🎯 Job {job.job_id} | task='{job.task}' | model={job.params.get('model')}")
        try:
            stems_dir = process(job)
            print(f"✅ Done: {stems_dir}")
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
