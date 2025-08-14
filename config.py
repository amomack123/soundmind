import os

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_JOBS = os.getenv("TOPIC_JOBS", "soundmind_jobs")
TOPIC_DLQ  = os.getenv("TOPIC_DLQ", "soundmind_jobs_dlq")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
RUN_MODE = os.getenv("RUN_MODE", "local")   # local | modal

S3_BUCKET = os.getenv("S3_BUCKET", "soundmind-jobs-amogh-us-east-2-a1b2")   # pick your bucket
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")