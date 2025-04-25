# server/aws_client.py
import os
import boto3
import logging
from datetime import datetime, timezone  # Add timezone
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_s3():
    """Return a singleton boto3 client (connection-pooled)."""
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

S3_BUCKET = os.getenv("S3_BUCKET", "dfjsx-Uploads")

def upload_bytes(data: bytes, filename: str) -> str:
    """
    Push raw bytes to S3 and return the key that was written.
    Key pattern: uploads/<UTC-timestamp>_<original filename>
    """
    s3 = get_s3()
    key = f"uploads/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{filename}"
    
    logger.debug(f"Uploading to bucket: {S3_BUCKET}, key: {key}")
    try:
        response = s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data)
        logger.debug(f"S3 response: {response}")
        # Verify the upload
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        logger.debug(f"File verified in S3: {key}")
        return key
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        raise Exception(f"S3 upload failed: {e}")


