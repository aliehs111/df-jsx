# server/routers/modelrunner.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
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

from server.database import get_async_db
from server.models import Dataset as DatasetModel
from server.auth.userroutes import current_user

router = APIRouter(prefix="/models", tags=["models"])
logger = logging.getLogger("server.main")
s3 = boto3.client("s3")
S3_BUCKET = "dfjsx-uploads"  # Ensure this is correct


class ModelRunRequest(BaseModel):
    dataset_id: int
    model_name: str
    target_column: str | None = None
    n_estimators: int | None = 100
    max_depth: int | None = None
    C: float | None = 1.0
    n_clusters: int | None = 3


# server/routers/modelrunner.py (only run_model_pca_kmeans shown, keep others as-is)
def run_model_pca_kmeans(df: pd.DataFrame, n_clusters=3):
    logger.info(
        f"PCA_KMeans: Starting with n_clusters={n_clusters}, df shape={df.shape}"
    )
    df_numeric = df.select_dtypes(include=["int64", "float64"])
    logger.info(
        f"PCA_KMeans: Numeric columns before filtering: {df_numeric.columns.tolist()}"
    )
    valid_cols = [c for c in df_numeric.columns if df_numeric[c].notna().any()]
    df_numeric = df_numeric[valid_cols]
    logger.info(
        f"PCA_KMeans: Numeric columns after filtering NaNs: {df_numeric.columns.tolist()}"
    )
    if len(valid_cols) < 2:
        raise ValueError(
            f"PCA requires at least 2 numeric columns with non-NaN values, found {len(valid_cols)}. "
            "Please preprocess the dataset to remove or impute missing values in Data Cleaning."
        )
    df_numeric = df_numeric.dropna()
    logger.info(
        f"PCA_KMeans: Numeric df shape after dropna: {df_numeric.shape}, columns={df_numeric.columns.tolist()}"
    )
    if df_numeric.empty:
        raise ValueError(
            "No samples remain after dropping rows with missing values. "
            "Please preprocess the dataset to impute or remove missing values in Data Cleaning."
        )
    logger.info(f"PCA_KMeans: Running PCA with n_components=2")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df_numeric)
    logger.info(f"PCA_KMeans: PCA transformed shape={reduced.shape}")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(reduced)
    logger.info(f"PCA_KMeans: KMeans clusters={clusters[:10].tolist()}...")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap="viridis", s=50)
    ax.set_title("PCA + KMeans Clustering")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {
        "input_shape": [int(x) for x in df.shape],
        "numeric_shape": [int(x) for x in df_numeric.shape],
        "n_clusters": int(n_clusters),
        "cluster_counts": {
            str(k): int(v)
            for k, v in pd.Series(clusters).value_counts().sort_index().items()
        },
        "pca_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "image_base64": img_base64,
    }


def run_model_random_forest(
    df: pd.DataFrame, target_column, n_estimators=100, max_depth=None
):
    logger.info(
        f"RandomForest: Target column '{target_column}', columns: {df.columns.tolist()}"
    )
    if target_column not in df.columns:
        raise ValueError(f"Missing '{target_column}' column.")
    y = df[target_column].dropna()
    logger.info(
        f"RandomForest: Target unique values: {y.nunique()}, values: {y.unique().tolist()}"
    )
    if y.nunique() < 2:
        raise ValueError(
            f"Classification requires at least 2 classes, found {y.nunique()}."
        )
    feature_cols = [
        c
        for c in df.select_dtypes(include=["int64", "float64"]).columns
        if c != target_column and df[c].notna().any()
    ]
    if not feature_cols:
        raise ValueError("No valid numeric feature columns found.")
    X = df[feature_cols].dropna()
    if X.empty:
        raise ValueError(
            f"No samples remain after dropping NaNs in features: {feature_cols}"
        )
    y = y.loc[X.index]
    logger.info(
        f"RandomForest: Features: {feature_cols}, X shape: {X.shape}, y shape: {y.shape}"
    )
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    rf.fit(X, y)
    preds = rf.predict(X)
    report = classification_report(y, preds, output_dict=True)
    conf_mat = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(feature_cols, rf.feature_importances_)
    ax.set_title("Random Forest Feature Importances")
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {
        "input_shape": [int(x) for x in df.shape],
        "numeric_shape": [int(x) for x in X.shape],
        "target_classes": [
            int(x) if isinstance(x, np.integer) else x for x in y.unique().tolist()
        ],
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "feature_importances": rf.feature_importances_.tolist(),
        "image_base64": img_base64,
    }


def run_model_logistic_regression(
    df: pd.DataFrame, target_column, C=1.0, max_iter=1000
):
    logger.info(
        f"LogisticRegression: Target column '{target_column}', columns: {df.columns.tolist()}"
    )
    if target_column not in df.columns:
        raise ValueError(f"Missing '{target_column}' column.")
    y = df[target_column].dropna()
    logger.info(
        f"LogisticRegression: Target unique values: {y.nunique()}, values: {y.unique().tolist()}"
    )
    if y.nunique() < 2:
        raise ValueError(
            f"Classification requires at least 2 classes, found {y.nunique()}."
        )
    feature_cols = [
        c
        for c in df.select_dtypes(include=["int64", "float64"]).columns
        if c != target_column and df[c].notna().any()
    ]
    if not feature_cols:
        raise ValueError("No valid numeric feature columns found.")
    X = df[feature_cols].dropna()
    if X.empty:
        raise ValueError(
            f"No samples remain after dropping NaNs in features: {feature_cols}"
        )
    y = y.loc[X.index]
    logger.info(
        f"LogisticRegression: Features: {feature_cols}, X shape: {X.shape}, y shape: {y.shape}"
    )
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
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "ROC not available", horizontalalignment="center")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {
        "input_shape": [int(x) for x in df.shape],
        "numeric_shape": [int(x) for x in X.shape],
        "target_classes": [
            int(x) if isinstance(x, np.integer) else x for x in y.unique().tolist()
        ],
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "coefficients": lr.coef_.tolist(),
        "intercept": lr.intercept_.tolist(),
        "image_base64": img_base64,
    }


@router.post("/run")
async def run_model(
    payload: ModelRunRequest,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(current_user),
):
    logger.info(
        f"Model run request: payload={payload.dict()}, user={user.id if user else None}"
    )
    dataset_id = payload.dataset_id
    model_name = payload.model_name
    target_column = payload.target_column
    n_estimators = payload.n_estimators
    max_depth = payload.max_depth
    C = payload.C
    n_clusters = payload.n_clusters

    if not dataset_id or not model_name:
        raise HTTPException(status_code=400, detail="Missing dataset_id or model_name")

    # Fetch dataset
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.has_cleaned_data or not dataset.s3_key_cleaned:
        raise HTTPException(status_code=400, detail="No cleaned data available")

    # Load cleaned data from S3
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=dataset.s3_key_cleaned)
        content = response["Body"].read()
        encodings = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    io.StringIO(content.decode(encoding, errors="replace"))
                )
                logger.info(
                    f"Loaded CSV {dataset.s3_key_cleaned} with encoding {encoding}"
                )
                break
            except Exception as e:
                logger.warning(
                    f"Failed with encoding {encoding} for {dataset.s3_key_cleaned}: {str(e)}"
                )
        if df is None:
            raise Exception("Failed to parse CSV with any encoding")
    except Exception as e:
        logger.error(
            f"Failed to load CSV {dataset.s3_key_cleaned} for dataset {dataset_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=400, detail=f"Failed to load cleaned data: {str(e)}"
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="Empty dataset")

    try:
        if model_name == "PCA_KMeans":
            output = run_model_pca_kmeans(df, n_clusters=n_clusters)
        elif model_name == "RandomForest":
            if not target_column:
                raise HTTPException(status_code=400, detail="Missing target_column")
            output = run_model_random_forest(
                df,
                target_column=target_column,
                n_estimators=n_estimators,
                max_depth=max_depth,
            )
        elif model_name == "LogisticRegression":
            if not target_column:
                raise HTTPException(status_code=400, detail="Missing target_column")
            output = run_model_logistic_regression(
                df, target_column=target_column, C=C, max_iter=1000
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Model '{model_name}' not implemented"
            )
        return {
            "dataset_id": int(dataset_id),
            "model": model_name,
            "status": "success",
            **output,
        }
    except ValueError as e:
        logger.error(f"Model {model_name} failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Model {model_name} failed for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model run failed: {str(e)}")
