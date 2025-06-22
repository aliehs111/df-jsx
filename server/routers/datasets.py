# server/routers/datasets.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import io
import boto3
import logging

from server.database import get_async_db
from server.models import Dataset as DatasetModel
from server.schemas import DatasetSummary
from server.auth.userroutes import current_user

router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = logging.getLogger("server.main")
s3 = boto3.client("s3")
S3_BUCKET = "dfjsx-uploads"


@router.get(
    "/cleaned",
    response_model=list[DatasetSummary],
    dependencies=[Depends(current_user)],
)
async def list_cleaned_datasets(db: AsyncSession = Depends(get_async_db)):
    try:
        stmt = select(DatasetModel).filter(
            DatasetModel.has_cleaned_data == True, DatasetModel.s3_key_cleaned != None
        )
        result = await db.execute(stmt)
        datasets = result.scalars().all()
        logger.info(f"Fetched {len(datasets)} cleaned datasets")
        return datasets
    except Exception as e:
        logger.error(f"Failed to fetch cleaned datasets: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch cleaned datasets: {str(e)}"
        )


@router.get("/{dataset_id}/columns", dependencies=[Depends(current_user)])
async def get_dataset_columns(
    dataset_id: int, db: AsyncSession = Depends(get_async_db)
):
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.has_cleaned_data or not dataset.s3_key_cleaned:
        raise HTTPException(
            status_code=400,
            detail="No cleaned data available. Please clean the dataset in Data Cleaning.",
        )
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=dataset.s3_key_cleaned)
        content = response["Body"].read()
        encodings = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    io.StringIO(content.decode(encoding, errors="replace"))
                )
                logger.info(
                    f"Loaded CSV {dataset.s3_key_cleaned} with encoding {encoding}"
                )
                break
            except Exception as e:
                logger.warning(
                    f"Failed with encoding {encoding} for {dataset.s3_key_cleaned}: {str(e)}"
                )
        if df is None:
            raise Exception("Failed to parse CSV with any encoding")
        return {"columns": df.columns.tolist()}
    except Exception as e:
        logger.error(f"Failed to fetch columns for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load cleaned data: {str(e)}. Please ensure the dataset is cleaned and saved in Data Cleaning.",
        )


@router.get(
    "/{dataset_id}/column/{column_name}/unique", dependencies=[Depends(current_user)]
)
async def get_column_unique_values(
    dataset_id: int, column_name: str, db: AsyncSession = Depends(get_async_db)
):
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.has_cleaned_data or not dataset.s3_key_cleaned:
        raise HTTPException(
            status_code=400,
            detail="No cleaned data available. Please clean the dataset in Data Cleaning.",
        )
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=dataset.s3_key_cleaned)
        content = response["Body"].read()
        encodings = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    io.StringIO(content.decode(encoding, errors="replace"))
                )
                logger.info(
                    f"Loaded CSV {dataset.s3_key_cleaned} with encoding {encoding}"
                )
                break
            except Exception as e:
                logger.warning(
                    f"Failed with encoding {encoding} for {dataset.s3_key_cleaned}: {str(e)}"
                )
        if df is None:
            raise Exception("Failed to parse CSV with any encoding")
        if column_name not in df.columns:
            raise HTTPException(
                status_code=400, detail=f"Column '{column_name}' not found"
            )
        unique_count = df[column_name].dropna().nunique()
        return {"unique_count": unique_count}
    except Exception as e:
        logger.error(
            f"Failed to fetch unique values for dataset {dataset_id}, column {column_name}: {str(e)}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load column data: {str(e)}. Please ensure the dataset is cleaned in Data Cleaning.",
        )
