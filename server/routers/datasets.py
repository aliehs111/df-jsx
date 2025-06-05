# server/routers/datasets.py

from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
