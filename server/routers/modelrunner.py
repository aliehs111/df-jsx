# server/routers/modelrunner.py

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import numpy as np
import io, os, json, base64, logging
import boto3
from botocore.config import Config
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from server.database import get_async_db
from server.models import Dataset as DatasetModel
from server.auth.userroutes import current_user

# -----------------------
# Router + Logging
# -----------------------
router = APIRouter(prefix="/models", tags=["models"])
logger = logging.getLogger("server.modelrunner")

# -----------------------
# AWS Clients
# -----------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
runtime = boto3.client(
    "sagemaker-runtime",
    region_name=AWS_REGION,
    config=Config(connect_timeout=60, read_timeout=300, retries={"max_attempts": 3}),
)
s3 = boto3.client("s3", region_name=AWS_REGION)

S3_BUCKET = os.getenv("S3_BUCKET", "dfjsx-uploads")
ENDPOINT_NAME = os.getenv("SAGEMAKER_MME_ENDPOINT", "huggingface-mme-gpu")
PREFIX = "mme_models"

MODEL_ENDPOINTS = {
    "Sentiment": ENDPOINT_NAME,
    "AnomalyDetection": ENDPOINT_NAME,
    "TimeSeriesForecasting": ENDPOINT_NAME,
}

# -----------------------
# Schemas
# -----------------------
class ModelRunRequest(BaseModel):
    dataset_id: int
    model_name: str
    target_column: str | None = None
    n_estimators: int | None = 100
    max_depth: int | None = None
    C: float | None = 1.0
    n_clusters: int | None = 3

# -----------------------
# Helpers
# -----------------------
def json_error(status: int, message: str, suggestion: str = None):
    payload = {"error": message}
    if suggestion:
        payload["suggestion"] = suggestion
    return JSONResponse(status_code=status, content=payload)

# -----------------------
# Endpoints
# -----------------------
@router.get("/available")
async def get_available_models():
    return {
        "models": [
            {"name": "RandomForest", "description": "Random forest classification"},
            {"name": "PCA_KMeans", "description": "Clustering using PCA + KMeans"},
            {"name": "LogisticRegression", "description": "Binary/multi-class logistic regression"},
            {"name": "Sentiment", "description": "Text sentiment analysis via SageMaker (GPU)"},
            {"name": "AnomalyDetection", "description": "Detect unusual patterns in numeric data"},
            {"name": "TimeSeriesForecasting", "description": "Predict future values from time series"},
            # {"name": "FeatureImportance", "description": "Rank features using XGBoost"}, # optional
        ]
    }

@router.get("/predict")
def predict(model: str = Query(...), text: str = Query(...)):
    """Quick prediction for a single text using MME."""
    if model not in MODEL_ENDPOINTS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not supported for SageMaker prediction."
        )
    payload = {"inputs": [text]}
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            TargetModel=f"{PREFIX}/{model}.tar.gz",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode())
        return {"model": model, "input": text, "prediction": result}
    except Exception as e:
        logger.error(f"SageMaker prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"SageMaker prediction failed: {str(e)}")

# -----------------------
# Model Runners
# -----------------------
def run_model_pca_kmeans(df: pd.DataFrame, n_clusters=3):
    df_numeric = df.select_dtypes(include=["int64", "float64"])
    valid_cols = [c for c in df_numeric.columns if df_numeric[c].notna().any()]
    df_numeric = df_numeric[valid_cols].dropna()
    if len(valid_cols) < 2 or df_numeric.empty:
        raise ValueError("PCA requires at least 2 numeric columns with non-NaN values.")
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
        "numeric_shape": list(df_numeric.shape),
        "n_clusters": int(n_clusters),
        "cluster_counts": {str(k): int(v) for k, v in pd.Series(clusters).value_counts().items()},
        "pca_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "image_base64": img_base64,
    }

def run_model_random_forest(df: pd.DataFrame, target_column, n_estimators=100, max_depth=None):
    if target_column not in df.columns:
        raise ValueError(f"Missing '{target_column}' column.")
    y = df[target_column].dropna()
    if y.nunique() < 2:
        raise ValueError("Classification requires at least 2 classes.")
    feature_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c != target_column]
    X = df[feature_cols].dropna()
    y = y.loc[X.index]
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)
    report = classification_report(y, preds, output_dict=True)
    conf_mat = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(feature_cols, rf.feature_importances_)
    ax.set_title("Random Forest Feature Importances")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {
        "input_shape": list(df.shape),
        "numeric_shape": list(X.shape),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "feature_importances": rf.feature_importances_.tolist(),
        "image_base64": img_base64,
    }

def run_model_logistic_regression(df: pd.DataFrame, target_column, C=1.0, max_iter=1000):
    if target_column not in df.columns:
        raise ValueError(f"Missing '{target_column}' column.")
    y = df[target_column].dropna()
    if y.nunique() < 2:
        raise ValueError("Classification requires at least 2 classes.")
    feature_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c != target_column]
    X = df[feature_cols].dropna()
    y = y.loc[X.index]
    lr = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    lr.fit(X, y)
    preds = lr.predict(X)
    probs = lr.predict_proba(X) if hasattr(lr, "predict_proba") else None
    report = classification_report(y, preds, output_dict=True)
    conf_mat = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(6, 4))
    if probs is not None and probs.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "ROC not available", ha="center")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {
        "input_shape": list(df.shape),
        "numeric_shape": list(X.shape),
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "coefficients": lr.coef_.tolist(),
        "intercept": lr.intercept_.tolist(),
        "image_base64": img_base64,
    }

def run_model_sagemaker(df: pd.DataFrame, text_column: str, model_name: str):
    if text_column not in df.columns:
        raise ValueError(f"Missing '{text_column}' column.")
    texts = df[text_column].dropna().astype(str).tolist()
    endpoint_name = MODEL_ENDPOINTS.get(model_name)
    if not endpoint_name:
        raise ValueError(f"No SageMaker endpoint configured for model {model_name}")
    labels, scores = [], []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        payload = json.dumps({"inputs": texts[i:i+batch_size]})
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload,
            Accept="application/json",
            TargetModel=f"{model_name.lower()}.tar.gz",
        )
        batch_result = json.loads(response["Body"].read().decode("utf-8"))
        if model_name == "Sentiment":
            label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
            labels.extend([label_map.get(r["label"], r["label"]) for r in batch_result])
            scores.extend([r["score"] for r in batch_result])
    if model_name == "Sentiment":
        sentiment_counts = pd.Series(labels).value_counts().to_dict()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(sentiment_counts.keys(), sentiment_counts.values())
        ax.set_title("Sentiment Distribution")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return {
            "input_shape": list(df.shape),
            "num_texts": len(texts),
            "sentiment_counts": {k: int(v) for k, v in sentiment_counts.items()},
            "sample_results": list(zip(texts[:10], labels[:10], scores[:10])),
            "image_base64": img_base64,
        }

# -----------------------
# Run Endpoint
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
        output = run_model_sagemaker(df, text_column=payload.target_column, model_name=payload.model_name)
    else:
        raise HTTPException(status_code=400, detail=f"Model {payload.model_name} not supported")
    return {"dataset_id": int(payload.dataset_id), "model": payload.model_name, "status": "success", **output}





      



