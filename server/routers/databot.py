# server/routers/databot.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from openai import OpenAI
import os

from server.database import get_async_db
from server.models import Dataset as DatasetModel
from server import schemas

router = APIRouter(prefix="/databot", tags=["Databot"])
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.post("/query")
async def databot_query(
    request: schemas.DatabotQuery,
    db: AsyncSession = Depends(get_async_db),
):
    dataset_id = request.dataset_id
    question = request.question

    # Fetch dataset
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Build Context
    context = f"""
Dataset Title: {dataset.title}
Description: {dataset.description or "No description"}
Rows: {dataset.n_rows or "Unknown"}
Columns: {dataset.n_columns or "Unknown"}
Target Column: {dataset.target_column or "None"}
Missing Values: {dataset.has_missing_values if dataset.has_missing_values is not None else "Unknown"}
Current Stage: {dataset.current_stage or "N/A"}
"""

    # Column Metadata
    if dataset.column_metadata:
        context += "\nColumn Metadata:\n"
        for col, meta in dataset.column_metadata.items():
            context += (
                f"- {col}: dtype={meta.get('dtype')}, "
                f"unique={meta.get('n_unique')}, "
                f"nulls={meta.get('null_count')}\n"
            )

    # Processing Log
    if dataset.processing_log:
        context += "\nProcessing Log (cleaning steps applied):\n"
        if isinstance(dataset.processing_log, str):
            context += f"- {dataset.processing_log}\n"
        elif isinstance(dataset.processing_log, list):
            for step in dataset.processing_log:
                context += f"- {step}\n"

    # Normalization Params
    if dataset.normalization_params:
        context += "\nNormalization Parameters:\n"
        for col, params in dataset.normalization_params.items():
            context += f"- {col}: {params}\n"

    # Categorical Mappings
    if dataset.categorical_mappings:
        context += "\nCategorical Mappings:\n"
        for col, mapping in dataset.categorical_mappings.items():
            context += f"- {col}: {mapping}\n"

    # OpenAI Query
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful tutor for data science. "
                        "Use the provided dataset metadata — including cleaning history "
                        "and preprocessing details — to answer questions clearly and concisely."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")

    return {"answer": answer}

@router.get("/suggestions/{dataset_id}")
async def get_databot_suggestions(dataset_id: int, db: AsyncSession = Depends(get_async_db)):
    ds = await db.get(DatasetModel, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    suggestions = []
    for col, meta in (ds.column_metadata or {}).items():
        if meta["dtype"] == "object" and any(kw in col.lower() for kw in ["price", "amount", "value", "market", "volume"]):
            suggestions.append(f"Convert '{col}' to numeric (currently object).")
        if meta["null_count"] > 0:
            suggestions.append(f"Handle missing values in '{col}' ({meta['null_count']} nulls).")
        if meta["dtype"].startswith("int") or meta["dtype"].startswith("float"):
            if meta.get("std", 0) > meta.get("mean", 0) * 2:
                suggestions.append(f"Consider binning '{col}' due to high variance (std={meta.get('std', 0):.2f}).")
    return {"suggestions": suggestions}








