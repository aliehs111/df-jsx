# server/routers/datarows.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from server.database import get_db
from server.models import Dataset
import pandas as pd
import boto3
import os
import io

router = APIRouter()
s3_client = boto3.client("s3")
S3_BUCKET = os.getenv("S3_BUCKET")

@router.get("/datasets/{dataset_id}/rows")
async def get_dataset_rows(
    dataset_id: int,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
    dataset = result.scalars().first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.s3_key_cleaned:
        raise HTTPException(status_code=400, detail="No cleaned dataset available")

    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=dataset.s3_key_cleaned)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        rows = df.head(limit).to_dict(orient="records")
        return {"rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 fetch failed: {str(e)}")

