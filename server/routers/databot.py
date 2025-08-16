from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from openai import OpenAI
import os
from typing import Optional, Dict, Any, Tuple, Literal, List

from server.database import get_async_db
from server.models import Dataset as DatasetModel
from server import schemas
from pydantic import BaseModel

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Dataset Advisor (badges, scoring) ----------


# ---------- Dataset Advisor (badges, scoring, candidates) ----------


BADGE_LIST = {
    "has_target",
    "binary",
    "multiclass",
    "high_dimensional",
    "categorical_features",
    "text_features",
    "mostly_numeric",
}

def _norm_dtype(s: str | None) -> str:
    if not s:
        return ""
    s = str(s).lower()
    if s in {"int", "int32", "int64", "integer"}:
        return "int64"
    if s in {"float", "float32", "float64", "double"}:
        return "float64"
    if s in {"str", "string", "object", "category", "categorical"}:
        return "object"
    if s in {"bool", "boolean"}:
        return "bool"
    if s in {"datetime", "date", "timestamp"}:
        return "datetime"
    return s

def _safe_get_meta(cm: dict | None, col: str) -> dict:
    if not isinstance(cm, dict) or not col:
        return {}
    meta = cm.get(col)
    return meta if isinstance(meta, dict) else {}

def _count_types(column_metadata: dict | None) -> Tuple[int, int, int, int]:
    if not isinstance(column_metadata, dict):
        return 0, 0, 0, 0
    numeric_count = categorical_count = text_count = 0
    for _, meta in column_metadata.items():
        if not isinstance(meta, dict):
            continue
        dt = _norm_dtype(meta.get("dtype"))
        if dt in {"int64", "float64"}:
            numeric_count += 1
        elif dt in {"bool", "object"}:
            n_unique = meta.get("n_unique")
            if isinstance(n_unique, int) and n_unique > 50:
                text_count += 1
            else:
                categorical_count += 1
    total = len(column_metadata)
    return numeric_count, categorical_count, text_count, total

def _infer_candidate_targets(cm: dict | None) -> Dict[str, List[str]]:
    """
    Returns:
      {
        "binary": [colA, colB, ...],        # 2–3 uniques
        "multiclass": [colX, colY, ...],    # 3–20 uniques
      }
    Heuristics are conservative and ignore high-cardinality.
    """
    out = {"binary": [], "multiclass": []}
    if not isinstance(cm, dict):
        return out
    for col, meta in cm.items():
        if not isinstance(meta, dict):
            continue
        dt = _norm_dtype(meta.get("dtype"))
        # candidate targets are typically categorical/bool-ish
        if dt not in {"object", "bool", "int64"}:
            continue
        n_unique = meta.get("n_unique")
        if not isinstance(n_unique, int):
            continue
        # treat tiny-cardinality ints/objects as categorical labels
        if 2 <= n_unique <= 3:
            out["binary"].append(col)
        if 3 <= n_unique <= 20:
            out["multiclass"].append(col)
    # de-dup overlap (e.g., n_unique == 3 goes to both; keep both)
    return out

def _compute_badges_and_suggestions(dataset) -> Tuple[List[str], Dict[str, int | float | bool], List[str], List[str], Dict[str, List[str]]]:
    """
    Returns: badges, signals, suggested_models, why, candidates
    """
    badges: List[str] = []
    suggested: List[str] = []
    why: List[str] = []

    cm = dataset.column_metadata if hasattr(dataset, "column_metadata") else None
    n_rows = getattr(dataset, "n_rows", None) or 0
    n_columns = getattr(dataset, "n_columns", None) or (len(cm) if isinstance(cm, dict) else 0)
    target_col = getattr(dataset, "target_column", None)

    numeric_count, categorical_count, text_count, total_cols = _count_types(cm)
    total_cols = n_columns or total_cols

    # Candidates (even if no target is set)
    candidates = _infer_candidate_targets(cm)

    # has_target
    has_target = bool(target_col) and isinstance(cm, dict) and target_col in cm
    if has_target:
        badges.append("has_target")

    # class_count (if target set)
    class_count = None
    if has_target:
        tmeta = _safe_get_meta(cm, target_col)
        cu = tmeta.get("n_unique")
        if isinstance(cu, int):
            class_count = cu

    # binary / multiclass (from target) — candidates handled later
    if has_target and isinstance(class_count, int):
        if class_count == 2:
            badges.append("binary")
        elif class_count > 2:
            badges.append("multiclass")

    # high_dimensional
    if isinstance(total_cols, int) and total_cols >= 25:
        badges.append("high_dimensional")

    # categorical_features
    if categorical_count >= 2:
        badges.append("categorical_features")

    # text_features
    if text_count >= 1:
        badges.append("text_features")

    # mostly_numeric
    mostly_numeric = (numeric_count / max(total_cols or 1, 1)) >= 0.7
    if mostly_numeric:
        badges.append("mostly_numeric")

    # Suggested models
    if "binary" in badges:
        suggested.extend(["LogisticRegression", "RandomForest"])
    elif "multiclass" in badges:
        suggested.append("RandomForest")
    if ("high_dimensional" in badges and "mostly_numeric" in badges) or not has_target:
        if "PCA_KMeans" not in suggested:
            suggested.append("PCA_KMeans")

    # WHY bullets (at most 2)
    if "binary" in badges:
        why.append("Binary target → logistic fits")
    if "multiclass" in badges:
        why.append("Multiclass target (>2 classes)")
    if not has_target and candidates["binary"]:
        why.append(f"Has binary candidate columns: {', '.join(candidates['binary'][:2])}")
    if not has_target and not candidates["binary"] and candidates["multiclass"]:
        why.append(f"Has multiclass candidate columns: {', '.join(candidates['multiclass'][:2])}")
    if "mostly_numeric" in badges and len(why) < 2:
        why.append("Mostly numeric features")
    if "high_dimensional" in badges and len(why) < 2:
        why.append("High dimensional feature space")

    signals = {
        "numeric_count": numeric_count,
        "categorical_count": categorical_count,
        "text_count": text_count,
        "n_columns": total_cols,
        "has_target": has_target,
        "class_count": class_count if class_count is not None else None,
        "high_dimensional": "high_dimensional" in badges,
        "mostly_numeric": mostly_numeric,
        "n_rows": n_rows,
    }

    def _dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    badges = _dedupe([b for b in badges if b in BADGE_LIST])
    suggested = _dedupe(suggested)
    why = _dedupe(why)[:2]

    return badges, signals, suggested, why, candidates

def _score_for_task(task: Literal["logistic", "multiclass", "cluster"], badges: List[str], candidates: Dict[str, List[str]]) -> int:
    s = 0
    if task == "logistic":
        # Hard preference if real target is binary OR candidate exists
        if ("has_target" in badges and "binary" in badges) or (candidates.get("binary")):
            s += 10
        if "mostly_numeric" in badges:
            s += 3
        if "categorical_features" in badges:
            s += 2
        if "high_dimensional" not in badges:
            s += 1
    elif task == "multiclass":
        if ("has_target" in badges and "multiclass" in badges) or (candidates.get("multiclass")):
            s += 10
        if "categorical_features" in badges:
            s += 3
        if "mostly_numeric" in badges:
            s += 2
        if "high_dimensional" not in badges:
            s += 1
    elif task == "cluster":
        if "high_dimensional" in badges:
            s += 3
        if "mostly_numeric" in badges:
            s += 3
        if "text_features" not in badges:
            s += 1
    return s


class DatasetState(BaseModel):
    dataset_id: int
    options: dict

dataset_states = {}

class Action(BaseModel):
    action: str

actions_store = {}

class DatabotQueryFlexible(BaseModel):
    question: str
    dataset_id: Optional[int] = None
    bot_type: Optional[str] = None
    model_context: Optional[Dict[str, Any]] = None
    app_info: Optional[str] = None

class WelcomeQuery(BaseModel):
    question: str
    app_info: str

@router.post("/query_welcome")
async def databot_query_welcome(request: WelcomeQuery):
    question = request.question or ""
    app_info = request.app_info or ""
    system_prompt = (
        "You’re an expert on the df.jsx app, bursting with enthusiasm! Use the provided app info "
        "to answer questions in a fun, engaging, conversational way, summarizing details naturally. "
        "Help users pick the best dataset for models (e.g., Random Forest, Logistic Regression) and "
        "interpret model outputs (e.g., feature importances, classification metrics). For unrelated "
        "questions, say: 'I’m here to help with df.jsx. Ask about features, workflow, or model guidance!'"
    )
    user_prompt = f"App Info:\n{app_info}\n\nQuestion: {question}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")

@router.post("/query")
async def databot_query(
    request: DatabotQueryFlexible,
    db: AsyncSession = Depends(get_async_db),
):
    question = request.question or ""
    if request.dataset_id is not None:
        dataset_id = request.dataset_id
        result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
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
                context += (
                    f"- {col}: dtype={meta.get('dtype')}, "
                    f"unique={meta.get('n_unique')}, "
                    f"nulls={meta.get('null_count')}\n"
                )
        if dataset.processing_log:
            context += "\nProcessing Log (cleaning steps applied):\n"
            if isinstance(dataset.processing_log, str):
                context += f"- {dataset.processing_log}\n"
            elif isinstance(dataset.processing_log, list):
                for step in dataset.processing_log:
                    context += f"- {step}\n"
        if dataset.normalization_params:
            context += "\nNormalization Parameters:\n"
            for col, params in dataset.normalization_params.items():
                context += f"- {col}: {params}\n"
        if dataset.categorical_mappings:
            context += "\nCategorical Mappings:\n"
            for col, mapping in dataset.categorical_mappings.items():
                context += f"- {col}: {mapping}\n"
        if request.app_info:  # Check if app_info exists
            context += "\nApp Guide:\n" + request.app_info       
        system_prompt = (
            "You are a helpful tutor for df.jsx. For questions about using the app or cleaning data, provide step-by-step instructions from the App Guide, "
            "tailored to the dataset’s metadata (e.g., column names, missing values). For dataset-specific questions, focus on metadata details. "
            "Answer clearly and concisely, combining both contexts when relevant."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    elif request.model_context:
        ctx = request.model_context or {}
        feat = (ctx.get("feature") or "").lower()
        inputs = ctx.get("inputs") or {}
        result = ctx.get("result") or {}
        lines = []
        if feat.startswith("college_earnings"):
            pct = result.get("prob")
            pct = f"{round(pct*100)}%" if isinstance(pct, (int, float)) else "—"
            lines += [
                "Context: College Earnings — 5y ≥ $75k.",
                f"Inputs: CIP4={inputs.get('cip4') or '?'}, Degree={inputs.get('degree_level') or '?'}, "
                f"State={inputs.get('state') or '?'}"
                + (f", Type={inputs.get('public_private')}" if inputs.get('public_private') else ""),
                f"Score: {result.get('bucket') or '—'} ({pct}).",
            ]
            drivers = result.get("drivers") or []
            if drivers:
                lines.append("Drivers: " + ", ".join(
                    (f"{d.get('direction','')}{d.get('factor','')}" for d in drivers)
                ))
        else:
            pct = result.get("prob")
            pct = f"{round(pct*100)}%" if isinstance(pct, (int, float)) else "—"
            lines += [
                "Context: Accessibility Misinterpretation Risk.",
                f"Audience: {inputs.get('audience') or '?'}, Medium: {inputs.get('medium') or '?'}, "
                f"Intent: {inputs.get('intent') if inputs.get('intent') is not None else '—'}",
                f"Score: {result.get('bucket') or '—'} ({pct}).",
            ]
            cs = result.get("confusion_sources") or []
            if cs:
                lines.append("Confusion: " + " | ".join(
                    f"{c.get('type')}: {', '.join(c.get('evidence') or [])}" for c in cs
                ))
            if result.get("rewrite"):
                lines.append(f"Rewrite ≤15: {result['rewrite']}")
        context = "\n".join([ln for ln in lines if ln])
        system_prompt = (
            "You are ModelBot, an expert assistant for model predictions. Answer questions naturally and concisely, "
            "using the provided context to explain results or related details. Avoid repeating structured responses like "
            "'Influential Factors' or 'Actionable Improvements' unless explicitly asked. Focus on conversational, "
            "context-specific answers, maintaining the existing context for follow-up questions."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        raise HTTPException(status_code=422, detail="Provide dataset_id or model_context")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")

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

@router.get("/cleaned_datasets/recommendations")
async def cleaned_datasets_recommendations(
    task: Literal["logistic", "multiclass", "cluster"] = Query(..., description="Which modeling task to recommend for"),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Ranked list of cleaned datasets for the given task.
    Falls back to candidate target inference when target_column isn't set.
    """
    # Adjust the "cleaned" condition to whatever you actually use
    q = await db.execute(
        select(DatasetModel).where(
            getattr(DatasetModel, "has_cleaned_data", True) == True  # noqa: E712
        )
    )
    rows = q.scalars().all()

    items = []
    for ds in rows:
        badges, signals, suggested_models, why, candidates = _compute_badges_and_suggestions(ds)

        score = _score_for_task(task, badges, candidates)

        # For supervised tasks, allow through if either a proper target matches OR we have candidates
        if task == "logistic":
            if not (("has_target" in badges and "binary" in badges) or candidates.get("binary")):
                continue
        elif task == "multiclass":
            if not (("has_target" in badges and "multiclass" in badges) or candidates.get("multiclass")):
                continue
        # cluster: keep everything, rely on score

        updated_at = getattr(ds, "updated_at", None) or getattr(ds, "uploaded_at", None)
        updated_iso = updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at) if updated_at else None

        items.append({
            "dataset_id": ds.id,
            "title": getattr(ds, "title", f"Dataset {ds.id}"),
            "n_rows": getattr(ds, "n_rows", None),
            "n_columns": getattr(ds, "n_columns", None),
            "target_column": getattr(ds, "target_column", None),
            "badges": badges,
            "suggested_models": suggested_models,
            "why": why,
            "signals": signals,
            "candidates": candidates,   # <-- helpful to show suggested target columns
            "updated_at": updated_iso,
            "_score": score,
        })

    # Sort: score desc → updated_at desc → id asc
    def _sort_key(it: dict):
        return (
            it.get("_score", 0),
            it.get("updated_at") or "",
            -int(it.get("dataset_id") or 0),
        )

    items.sort(key=_sort_key, reverse=True)
    for it in items:
        it.pop("_score", None)

    return {"task": task, "items": items}
