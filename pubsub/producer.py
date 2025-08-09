import json, argparse, os
from kafka import KafkaProducer
from shared.job_models import SoundMindJob
from config import BOOTSTRAP, TOPIC_JOBS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path or URL to audio")
    ap.add_argument("--task", default="", help="Free text instruction")
    ap.add_argument("--model", default="nemo", choices=["nemo","htdemucs"])
    args = ap.parse_args()

    job = SoundMindJob(audio_path=args.file, task=args.task)
    job.params["model"] = args.model

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    producer.send(TOPIC_JOBS, job.model_dump())
    producer.flush()
    print(f"📤 Enqueued job {job.job_id} → {TOPIC_JOBS}")

if __name__ == "__main__":
    main()
