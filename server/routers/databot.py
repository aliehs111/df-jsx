# server/routers/databot.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from openai import OpenAI
import os
from typing import Optional, Dict, Any

from server.database import get_async_db
from server.models import Dataset as DatasetModel
from server import schemas
from pydantic import BaseModel
router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DatasetState(BaseModel):
    dataset_id: int
    options: dict

dataset_states = {}

class Action(BaseModel):
    action: str

actions_store = {}  # Temporary in-memory store; replace with DB later

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
async def get_suggestions(dataset_id: int, page: str = Query(None), db: AsyncSession = Depends(get_async_db)):
    suggestions = []
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    state = dataset_states.get(dataset_id, {})
    if page == "data-cleaning":
        if dataset.column_metadata:
            for col, meta in dataset.column_metadata.items():
                if col in state.get("selected_columns", {}).get("fillna", []):
                    suggestions.append(f"Impute '{col}' with {state.get('fillna_strategy', 'unknown')} (has {meta['null_count']} nulls).")
                elif meta["null_count"] > 0:
                    suggestions.append(f"Handle missing values in '{col}' ({meta['null_count']} nulls).")
                if meta["dtype"] == "object" and meta["n_unique"] < 10:
                    suggestions.append(f"Encode '{col}' as categorical (low unique values: {meta['n_unique']}).")
                if meta["dtype"] in ["float64", "int64"]:
                    suggestions.append(f"Scale '{col}' to normalize numeric data.")
        for action in actions_store.get(dataset_id, [])[-5:]:
            if "Selected column" in action:
                col = action.split("'")[1]
                suggestions.append(f"Since you selected '{col}', consider specific cleaning options.")
        if dataset.processing_log:
            suggestions.append(f"Previous cleaning steps applied: {dataset.processing_log}")
    else:
        suggestions.append("General dataset review: Check for missing values and dtypes.")
    return {"suggestions": suggestions}

class ModelBotQuery(BaseModel):
    question: str
    model_context: Optional[Dict[str, Any]] = None

@router.post("/query-model")
async def databot_query_model(request: ModelBotQuery):
    ctx = request.model_context or {}
    prob = ctx.get("result", {}).get("prob")
    bucket = ctx.get("result", {}).get("bucket")
    confusion = ctx.get("result", {}).get("confusion_sources", [])
    rewrite = ctx.get("result", {}).get("rewrite")

    pct = f"{round(prob * 100)}%" if isinstance(prob, (int, float)) else "—"

    # Build a compact context summary for the model explanation
    summary_lines = [
        "You are a helpful tutor for model explanations. Explain clearly and concisely.",
        "Context: Accessibility Misinterpretation Risk prediction.",
        f"Audience: {ctx.get('inputs', {}).get('audience', '?')}, "
        f"Medium: {ctx.get('inputs', {}).get('medium', '?')}, "
        f"Intent: {ctx.get('inputs', {}).get('intent', '—')}",
        f"Score: {bucket} ({pct})." if bucket else None,
    ]
    if confusion:
        summary_lines.append(
            "Top confusion sources: " + " | ".join(
                f"{s.get('type', '?')}: {', '.join(s.get('evidence', []) or [])}"
                for s in confusion
            )
        )
    if rewrite:
        summary_lines.append(f"≤15-word rewrite: {rewrite}")

    system_prompt = (
        "You are ModelBot. Explain why the model predicted this score, point out the most "
        "influential confusion sources, and suggest concrete improvements. Keep it practical."
    )
    user_prompt = "\n".join([ln for ln in summary_lines if ln]) + \
                  f"\n\nUser question: {request.question.strip()}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt },
            ],
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")


@router.post("/track/{dataset_id}")
async def track_action(dataset_id: int, action: Action):
    if dataset_id not in actions_store:
        actions_store[dataset_id] = []
    actions_store[dataset_id].append(action.action)
    return {"status": "ok"}

@router.post("/state/{dataset_id}")
async def save_state(dataset_id: int, state: DatasetState):
    dataset_states[dataset_id] = state.options
    return {"status": "ok"}


