from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from server.database import get_async_db
from server.models import Dataset as DatasetModel
from server.schemas import DatasetSummary
from server.auth.userroutes import current_user

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get(
    "/cleaned",
    response_model=List[DatasetSummary],
    dependencies=[Depends(current_user)],
)
async def list_cleaned_datasets(db: AsyncSession = Depends(get_async_db)):
    """
    Return only those datasets whose cleaned_data is not null.
    We do NOT include ORDER BY here to avoid MySQL “Out of sort memory” issues.
    """
    stmt = select(
        DatasetModel.id,
        DatasetModel.title,
        DatasetModel.description,
        DatasetModel.filename,
        DatasetModel.s3_key,
        DatasetModel.uploaded_at,
    ).where(DatasetModel.cleaned_data.isnot(None))

    result = await db.execute(stmt)
    rows = result.all()

    return [
        DatasetSummary(
            id=row.id,
            title=row.title,
            description=row.description,
            filename=row.filename,
            s3_key=row.s3_key,
            uploaded_at=row.uploaded_at,
        )
        for row in rows
    ]


@router.get("/{dataset_id}/columns", dependencies=[Depends(current_user)])
async def get_dataset_columns(
    dataset_id: int, db: AsyncSession = Depends(get_async_db)
):
    """
    Return the column names of a dataset's cleaned_data.
    """
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.cleaned_data:
        raise HTTPException(status_code=400, detail="No cleaned data available")

    try:
        df = pd.DataFrame(dataset.cleaned_data)
        return {"columns": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid cleaned_data format: {str(e)}"
        )


@router.get(
    "/{dataset_id}/column/{column_name}/unique", dependencies=[Depends(current_user)]
)
async def get_column_unique_values(
    dataset_id: int, column_name: str, db: AsyncSession = Depends(get_async_db)
):
    """
    Return the number of unique values in a specified column of a dataset's cleaned_data.
    """
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.cleaned_data:
        raise HTTPException(status_code=400, detail="No cleaned data available")

    try:
        df = pd.DataFrame(dataset.cleaned_data)
        if column_name not in df.columns:
            raise HTTPException(
                status_code=400, detail=f"Column '{column_name}' not found"
            )
        unique_count = df[column_name].dropna().nunique()
        return {"unique_count": unique_count}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing column: {str(e)}"
        )
