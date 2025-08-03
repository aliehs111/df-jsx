# server/routers/modelrunner.py

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import numpy as np
import io
import boto3
import base64
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from pydantic import BaseModel
import logging
import os
from server.database import get_async_db
from server.models import Dataset as DatasetModel
from server.auth.userroutes import current_user

# -----------------------
# Router + Logging
# -----------------------
router = APIRouter(prefix="/models", tags=["models"])
logger = logging.getLogger("server.modelrunner")

# -----------------------
# Config
# -----------------------
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "dfjsx-uploads")
NORTHFLANK_GPU_URL = os.getenv("NORTHFLANK_GPU_URL")
NORTHFLANK_API_KEY = os.getenv("NORTHFLANK_API_KEY")

s3 = boto3.client("s3", region_name=AWS_REGION)

# -----------------------
# Request Schema
# -----------------------
class ModelRunRequest(BaseModel):
    dataset_id: int
    model_name: str
    target_column: str | None = None
    n_estimators: int | None = 100
    max_depth: int | None = None
    C: float | None = 1.0
    n_clusters: int | None = 3

@router.get("/available")
async def get_available_models():
    return {
        "models": [
            {"name": "RandomForest", "description": "Random forest classification"},
            {"name": "PCA_KMeans", "description": "Clustering using PCA + KMeans"},
            {"name": "LogisticRegression", "description": "Binary/multi-class logistic regression"},
            {"name": "Sentiment", "description": "Text sentiment analysis (Northflank GPU)"},
            {"name": "AnomalyDetection", "description": "GPU anomaly detection (Northflank)"},
            {"name": "TimeSeriesForecasting", "description": "GPU time series forecasting (Northflank)"},
        ]
    }

# -----------------------
# CPU Models
# -----------------------
def run_model_pca_kmeans(df: pd.DataFrame, n_clusters=3):
    df_numeric = df.select_dtypes(include=["int64", "float64"]).dropna()
    if df_numeric.shape[1] < 2:
        raise ValueError("PCA requires at least 2 numeric columns.")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df_numeric)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(reduced)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap="viridis", s=50)
    ax.set_title("PCA + KMeans Clustering")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {
        "input_shape": list(df.shape),
        "n_clusters": int(n_clusters),
        "cluster_counts": pd.Series(clusters).value_counts().to_dict(),
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "image_base64": img_base64,
    }

def run_model_random_forest(df: pd.DataFrame, target_column, n_estimators=100, max_depth=None):
    if target_column not in df.columns:
        raise ValueError(f"Missing '{target_column}' column.")
    y = df[target_column].dropna()
    X = df.select_dtypes(include=["int64", "float64"]).drop(columns=[target_column], errors="ignore").dropna()
    y = y.loc[X.index]
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)
    report = classification_report(y, preds, output_dict=True)
    conf_mat = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(X.columns, rf.feature_importances_)
    ax.set_title("Random Forest Feature Importances")
    ax.set_xticklabels(X.columns, rotation=45, ha="right")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {
        "input_shape": list(df.shape),
        "class_counts": y.value_counts().to_dict(),
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "feature_importances": rf.feature_importances_.tolist(),
        "image_base64": img_base64,
    }

def run_model_logistic_regression(df: pd.DataFrame, target_column, C=1.0):
    if target_column not in df.columns:
        raise ValueError(f"Missing '{target_column}' column.")
    y = df[target_column].dropna()
    X = df.select_dtypes(include=["int64", "float64"]).drop(columns=[target_column], errors="ignore").dropna()
    y = y.loc[X.index]
    lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
    lr.fit(X, y)
    preds = lr.predict(X)
    probs = lr.predict_proba(X)
    report = classification_report(y, preds, output_dict=True)
    conf_mat = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(6, 4))
    if probs.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
        ax.legend(loc="lower right")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {
        "input_shape": list(df.shape),
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "coefficients": lr.coef_.tolist(),
        "image_base64": img_base64,
    }

# -----------------------
# GPU (Northflank)
# -----------------------
def run_model_northflank(dataset_id: int, model_name: str, params: dict | None = None):
    if not NORTHFLANK_GPU_URL or not NORTHFLANK_API_KEY:
        raise ValueError("Northflank config vars missing")
    payload = {"dataset_id": str(dataset_id), "model": model_name, "params": params or {}}
    headers = {"Authorization": f"Bearer {NORTHFLANK_API_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.post(NORTHFLANK_GPU_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Northflank request failed: {str(e)}")

# -----------------------
# Run endpoint
# -----------------------
@router.post("/run")
async def run_model(payload: ModelRunRequest, db: AsyncSession = Depends(get_async_db), user=Depends(current_user)):
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == payload.dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset or not dataset.has_cleaned_data or not dataset.s3_key_cleaned:
        raise HTTPException(status_code=400, detail="Dataset not ready. Clean your data first.")
    response = s3.get_object(Bucket=S3_BUCKET, Key=dataset.s3_key_cleaned)
    content = response["Body"].read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8", errors="replace")))

    if payload.model_name == "PCA_KMeans":
        output = run_model_pca_kmeans(df, n_clusters=payload.n_clusters)
    elif payload.model_name == "RandomForest":
        output = run_model_random_forest(df, payload.target_column, payload.n_estimators, payload.max_depth)
    elif payload.model_name == "LogisticRegression":
        output = run_model_logistic_regression(df, payload.target_column, C=payload.C)
    elif payload.model_name in ["Sentiment", "AnomalyDetection", "TimeSeriesForecasting"]:
        if not payload.target_column:
            raise HTTPException(status_code=400, detail="Target column is required for this model")
        texts = df[payload.target_column].dropna().astype(str).tolist()[:100]
        output = run_model_northflank(payload.dataset_id, payload.model_name, params={"texts": texts})
    else:
        raise HTTPException(status_code=400, detail=f"Model {payload.model_name} not supported")

    return {"dataset_id": int(payload.dataset_id), "model": payload.model_name, "status": "success", **output}







      



