# server/routes/databot.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import os
import openai

from server.database import get_async_db
from server.models import Dataset


router = APIRouter(prefix="/databot", tags=["Databot"])

@router.post("/query")
async def databot_query(
    dataset_id: int,
    question: str,
    db: AsyncSession = Depends(get_async_db)
):
    # ✅ fetch dataset metadata
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # ✅ build context
    context = f"""
    Dataset: {dataset.title}
    Rows: {dataset.n_rows}
    Columns: {dataset.n_columns}
    Target column: {dataset.target_column or "None"}
    Missing values: {dataset.has_missing_values}
    Current stage: {dataset.current_stage or "N/A"}
    """

    # ✅ query OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="Missing OpenAI API key")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a teaching assistant for data science."},
                {"role": "user", "content": f"Context:\n{context}\n\nUser Question: {question}"}
            ],
        )
        answer = response.choices[0].message["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")

    return {"answer": answer}
