import os
import boto3
from urllib.parse import urlparse

_REGION = os.getenv("AWS_REGION", "us-west-2")
_S3 = boto3.client("s3", region_name=_REGION)

def upload_file(local_path: str, bucket: str, key: str) -> str:
    _S3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

def presign(bucket: str, key: str, expires: int = 86400) -> str:
    return _S3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )

def download_to_tmp(s3_url: str) -> str:
    p = urlparse(s3_url)
    local = f"/tmp/{os.path.basename(p.path)}"
    _S3.download_file(p.netloc, p.path.lstrip("/"), local)
    return local
