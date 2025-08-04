# server/routers/databot.py

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import os
from openai import OpenAI
from server.database import get_async_db
from server.models import Dataset as DatasetModel

router = APIRouter(prefix="/databot", tags=["Databot"])
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Request schema
class DatabotQuery(BaseModel):
    dataset_id: int
    question: str

@router.post("/query")
async def databot_query(
    request: DatabotQuery,
    db: AsyncSession = Depends(get_async_db)
):
    dataset_id = request.dataset_id
    question = request.question

    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Build context
    context = f"""
Dataset Title: {dataset.title}
Description: {dataset.description or "No description"}
Rows: {dataset.n_rows or "Unknown"}
Columns: {dataset.n_columns or "Unknown"}
Target Column: {dataset.target_column or "None"}
Missing Values: {dataset.has_missing_values if dataset.has_missing_values is not None else "Unknown"}
Current Stage: {dataset.current_stage or "N/A"}
"""

    if dataset.column_metadata:
        context += "\nColumn Metadata:\n"
        for col, meta in dataset.column_metadata.items():
            context += f"- {col}: dtype={meta.get('dtype')}, unique={meta.get('n_unique')}, nulls={meta.get('null_count')}\n"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful tutor for data science. Use the provided dataset metadata to answer questions clearly and concisely."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")

    return {"answer": answer}









