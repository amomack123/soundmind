import json, argparse, os
from kafka import KafkaProducer  # type: ignore
from shared.job_models import SoundMindJob
from config import BOOTSTRAP, TOPIC_JOBS  # TOPIC_JOBS = separation/default jobs topic

# Optional env override for diarization topic
DIAR_TOPIC = os.getenv("SOUNDMIND_DIAR_TOPIC", "soundmind_diarization_jobs")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path or URL to audio (repo-relative for Modal, e.g. samples/foo.wav)")
    ap.add_argument("--task", default="", help="Free text instruction (ignored for diarization)")
    ap.add_argument("--model", default="nemo", choices=["nemo", "htdemucs", "pyannote"],
                    help="nemo/htdemucs => separation; pyannote => diarization")
    ap.add_argument("--params", default="{}", help='Extra params JSON, e.g. \'{"device":"cuda"}\'')
    ap.add_argument("--run-mode", default=os.getenv("RUN_MODE", "modal"),
                    choices=["modal", "local"], help="Execution mode hint for consumers")
    args = ap.parse_args()

    # Parse params
    try:
        extra_params = json.loads(args.params) if args.params else {}
        if not isinstance(extra_params, dict):
            raise ValueError("params must be a JSON object")
    except Exception as e:
        raise SystemExit(f"--params must be JSON object. Got {args.params!r}. Error: {e}")

    # Build job using shared model (lets it mint job_id, etc.)
    job = SoundMindJob(audio_path=args.file, task=args.task or "")
    job.params["model"] = args.model
    job.params.update(extra_params)
    job.params["run_mode"] = args.run_mode

    # Route based on model: diarization vs separation
    if args.model == "pyannote" or (args.task and args.task.lower() == "diarize"):
        # Force diarization semantics
        job.task = "diarize"
        topic = DIAR_TOPIC
    else:
        topic = TOPIC_JOBS

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    payload = job.model_dump() if hasattr(job, "model_dump") else job.__dict__
    producer.send(topic, payload)
    producer.flush()
    print(f"📤 Enqueued job {payload.get('job_id')} → topic '{topic}' (model={args.model})")

if __name__ == "__main__":
    main()
