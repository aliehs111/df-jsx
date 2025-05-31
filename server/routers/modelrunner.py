# server/routers/modelrunner.py

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from server.database import get_async_db as get_db


router = APIRouter(prefix="/models", tags=["models"])

@router.post("/run")
async def run_model(payload: dict, db: AsyncSession = Depends(get_db)):
    dataset_id = payload.get("dataset_id")
    model_name = payload.get("model_name")

    if not dataset_id or not model_name:
        raise HTTPException(status_code=400, detail="Missing dataset_id or model_name")

    # Later: insert real model logic, API call, or background task here
    return {
        "dataset_id": dataset_id,
        "model": model_name,
        "status": "success",
        "message": f"Model '{model_name}' executed on dataset {dataset_id}"
    }
