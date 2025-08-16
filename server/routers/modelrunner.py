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
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
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

def _to_native(obj):
    """Recursively convert NumPy types/arrays to plain Python so FastAPI can JSON-encode."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def run_model_logistic_regression(df: pd.DataFrame, target_column, C=1.0):
    if target_column not in df.columns:
        raise ValueError(f"Missing '{target_column}' column.")

    # -----------------------------
    # 1) Target-only NA filter
    # -----------------------------
    y = df[target_column]
    mask = y.notna()
    y = y[mask]

    # Numeric features (incl. ints/floats); drop target if numeric
    num_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)
    if target_column in num_cols:
        num_cols.remove(target_column)

    # A few low-cardinality categoricals (safe one-hot)
    cat_candidates = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if target_column in cat_candidates:
        cat_candidates.remove(target_column)
    cat_cols = [c for c in cat_candidates if 2 <= df[c].nunique(dropna=True) <= 20][:5]

    if not num_cols and not cat_cols:
        raise ValueError("No usable features after filtering numeric/categorical columns.")

    X = df.loc[mask, num_cols + cat_cols].copy()

    # Must still have >= 2 classes after filtering
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(f"Target '{target_column}' has <2 classes after target filtering.")

    # -----------------------------
    # 2) Stratified split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -----------------------------
    # 3) Pipeline: impute + scale/encode + LR
    # -----------------------------
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), num_cols if num_cols else []),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
            ]), cat_cols if cat_cols else []),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    lr = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="lbfgs",
        multi_class="auto",
        class_weight="balanced",
        random_state=42,
    )

    pipe = Pipeline([("prep", pre), ("clf", lr)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

    # Metrics
    report = classification_report(y_test, preds, output_dict=True)
    conf_mat = confusion_matrix(y_test, preds, labels=np.unique(y_test))

    # -----------------------------
    # 4) Plot: ROC if binary else Confusion Matrix
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    uniq = np.unique(y_test)

    if probs is not None and len(uniq) == 2 and probs.shape[1] >= 2:
        # Align y_test to 0/1 in the same order as predict_proba columns (pipe.classes_)
        le = LabelEncoder().fit(pipe.classes_)
        y_bin = le.transform(y_test)  # 0/1
        # positive class is index 1
        pos_idx = 1
        fpr, tpr, _ = roc_curve(y_bin, probs[:, pos_idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], lw=1, linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
    else:
        cm = np.nan_to_num(conf_mat.astype(float))
        im = ax.imshow(cm, aspect="auto", vmin=0)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(uniq)))
        ax.set_yticks(range(len(uniq)))
        ax.set_xticklabels(uniq, rotation=45, ha="right")
        ax.set_yticklabels(uniq)
        ax.set_title("Confusion Matrix")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # Coefficients (multiclass: n_classes x n_features_after_transform)
    coef = pipe.named_steps["clf"].coef_

    payload = {
        "model": "LogisticRegression",
        "input_shape": [int(v) for v in list(df.shape)],
        "class_counts": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()},
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "coefficients": coef.tolist(),
        "image_base64": image_base64,
    }
    return _to_native(payload)



# -----------------------
# GPU (Northflank)
# -----------------------
def run_model_northflank(dataset_id: int, model_name: str, params: dict | None = None):
    if not NORTHFLANK_GPU_URL or not NORTHFLANK_API_KEY:
        raise ValueError("Northflank config vars missing")
    payload = {"dataset_id": str(dataset_id), "model": model_name, "params": params or {}}
    headers = {
        "Authorization": f"Bearer {NORTHFLANK_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            NORTHFLANK_GPU_URL,
            json=payload,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Northflank request failed: {str(e)}"
        )

# -----------------------
# Run endpoint
# -----------------------
@router.post("/run")
async def run_model(
    payload: ModelRunRequest,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(current_user)
):
    result = await db.execute(
        select(DatasetModel).where(DatasetModel.id == payload.dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset or not dataset.has_cleaned_data or not dataset.s3_key_cleaned:
        raise HTTPException(
            status_code=400,
            detail="Dataset not ready. Clean your data first."
        )

    response = s3.get_object(Bucket=S3_BUCKET, Key=dataset.s3_key_cleaned)
    content = response["Body"].read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8", errors="replace")))

    if payload.model_name == "PCA_KMeans":
        output = run_model_pca_kmeans(
            df, n_clusters=payload.n_clusters
        )

    elif payload.model_name == "RandomForest":
        output = run_model_random_forest(
            df, payload.target_column,
            payload.n_estimators,
            payload.max_depth
        )

    elif payload.model_name == "LogisticRegression":
        output = run_model_logistic_regression(
            df, payload.target_column, C=payload.C
        )

    elif payload.model_name == "Sentiment":
        if not payload.target_column:
            raise HTTPException(status_code=400, detail="Target column is required for Sentiment")
        texts = df[payload.target_column].dropna().astype(str).tolist()[:100]
        output = run_model_northflank(
            payload.dataset_id,
            payload.model_name,
            params={"texts": texts}
    )

    elif payload.model_name == "AnomalyDetection":
        records = df.to_dict(orient="records")
        if not records:
            raise HTTPException(status_code=400, detail="Empty dataset for anomaly detection")
        print(f"[DEBUG] Sending {len(records)} records to GPU service")
        output = run_model_northflank(
            payload.dataset_id,
            payload.model_name,
            params={"records": records}
    )


    elif payload.model_name == "TimeSeriesForecasting":
        if not payload.target_column:
            raise HTTPException(
                status_code=400,
                detail="Target column is required for TimeSeriesForecasting"
            )
        if "timestamp" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Dataset must include a 'timestamp' column for forecasting"
            )
        series = df[["timestamp", payload.target_column]].dropna().to_dict(orient="records")
        output = run_model_northflank(
            payload.dataset_id,
            payload.model_name,
            params={"series": series}
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Model {payload.model_name} not supported"
        )

    return {
        "dataset_id": int(payload.dataset_id),
        "model": payload.model_name,
        "status": "success",
        **output
    }








      



