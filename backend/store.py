# backend/store.py
"""S3 JSON cache, the daily brief counter, and the API key lookup."""
import json
import os
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

S3_CACHE_BUCKET = os.getenv("S3_CACHE_BUCKET", "")
API_KEY_PARAM = os.getenv("ANTHROPIC_KEY_PARAM", "/marketpulse/anthropic-api-key")
KEY_REFRESH_SECONDS = 600

_s3 = boto3.client("s3")
_ssm = boto3.client("ssm")
_key_cache = {"value": None, "fetched_at": 0.0}


def get_json(key: str, ttl_seconds: int):
    """Return the cached payload if it is younger than ttl_seconds, else None."""
    if not S3_CACHE_BUCKET:
        return None
    try:
        obj = _s3.get_object(Bucket=S3_CACHE_BUCKET, Key=key)
        doc = json.loads(obj["Body"].read())
    except (ClientError, ValueError):
        return None
    if time.time() - float(doc.get("_cached_at", 0)) > ttl_seconds:
        return None
    return doc.get("payload")


def put_json(key: str, payload) -> None:
    if not S3_CACHE_BUCKET:
        return
    doc = {"_cached_at": time.time(), "payload": payload}
    try:
        _s3.put_object(
            Bucket=S3_CACHE_BUCKET,
            Key=key,
            Body=json.dumps(doc, separators=(",", ":")).encode("utf-8"),
            ContentType="application/json",
        )
    except ClientError as e:
        print(f"cache write failed for {key}: {e.response['Error']['Code']}")


def reserve_brief_slot(daily_limit: int) -> bool:
    """Count one model call against today's limit. False once the limit is reached.

    The read-then-write is not atomic, so a burst can overshoot by a call or two.
    That is acceptable here: the limit exists to bound spend, not to meter exactly.
    """
    if not S3_CACHE_BUCKET:
        return False
    key = f"usage/{datetime.now(timezone.utc):%Y-%m-%d}.json"
    used = 0
    try:
        obj = _s3.get_object(Bucket=S3_CACHE_BUCKET, Key=key)
        used = int(json.loads(obj["Body"].read()).get("briefs", 0))
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("NoSuchKey", "404"):
            # If the counter cannot be read, fail closed and spend nothing.
            print(f"usage read failed: {e.response['Error']['Code']}")
            return False
    except ValueError:
        return False
    if used >= daily_limit:
        return False
    try:
        _s3.put_object(
            Bucket=S3_CACHE_BUCKET,
            Key=key,
            Body=json.dumps({"briefs": used + 1}).encode("utf-8"),
            ContentType="application/json",
        )
    except ClientError as e:
        print(f"usage write failed: {e.response['Error']['Code']}")
        return False
    return True


def get_api_key():
    """Read the Claude API key from SSM Parameter Store (SecureString), cached in memory."""
    now = time.time()
    if _key_cache["value"] and now - _key_cache["fetched_at"] < KEY_REFRESH_SECONDS:
        return _key_cache["value"]
    try:
        resp = _ssm.get_parameter(Name=API_KEY_PARAM, WithDecryption=True)
        value = resp["Parameter"]["Value"].strip()
    except ClientError as e:
        print(f"api key lookup failed: {e.response['Error']['Code']}")
        return None
    # A placeholder value means the key has not been set yet.
    if not value.startswith("sk-ant-"):
        return None
    _key_cache.update(value=value, fetched_at=now)
    return value
