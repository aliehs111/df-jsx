# server/routers/modelrunner.py

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from server.database import get_async_db as get_db
from server.models import Dataset
from sqlalchemy import select
import pandas as pd
import io, base64
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from server.utils.encoders import _to_py

router = APIRouter(prefix="/models", tags=["models"])


def run_model_pca_kmeans(df: pd.DataFrame, n_clusters=3):
    # Diagnostics
    rows, cols = df.shape
    df_numeric = df.select_dtypes(include="number")
    numeric_rows, numeric_cols = df_numeric.dropna().shape
    if numeric_cols < 2:
        raise ValueError(
            f"PCA requires at least 2 numeric features, found {numeric_cols}."
        )

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df_numeric.dropna())
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    clusters = kmeans.fit_predict(reduced)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap="viridis", s=50)
    ax.set_title("PCA + KMeans Clustering")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "input_shape": [rows, cols],
        "numeric_shape": [numeric_rows, numeric_cols],
        "n_clusters": n_clusters,
        "cluster_counts": dict(pd.Series(clusters).value_counts().sort_index()),
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "image_base64": img_base64,
    }


def run_model_random_forest(df: pd.DataFrame, n_estimators=100, max_depth=None):
    # Diagnostics
    rows, cols = df.shape
    if "target" not in df.columns:
        raise ValueError(
            "Missing 'target' column: classification models require a target column."
        )
    df_numeric = df.select_dtypes(include="number")
    numeric_rows, numeric_cols = df_numeric.dropna().shape
    feature_cols = [c for c in df_numeric.columns if c != "target"]
    if len(feature_cols) == 0:
        raise ValueError(
            "No numeric feature columns found. Random Forest requires input features."
        )
    y = df_numeric["target"].dropna()
    if y.nunique() < 2:
        raise ValueError(
            f"Classification requires at least 2 classes, found {y.nunique()}."
        )
    X = df_numeric[feature_cols].loc[y.index]

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(X, y)
    preds = rf.predict(X)

    report = classification_report(y, preds, output_dict=True)
    conf_mat = confusion_matrix(y, preds)

    # Feature importances plot
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
        "input_shape": [rows, cols],
        "numeric_shape": [numeric_rows, numeric_cols],
        "target_classes": y.unique().tolist(),
        "class_counts": y.value_counts().to_dict(),
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "feature_importances": rf.feature_importances_.tolist(),
        "image_base64": img_base64,
    }


def run_model_logistic_regression(df: pd.DataFrame, C=1.0, max_iter=100):
    # Diagnostics
    rows, cols = df.shape
    if "target" not in df.columns:
        raise ValueError(
            "Missing 'target' column: classification models require a target column."
        )
    df_numeric = df.select_dtypes(include="number")
    numeric_rows, numeric_cols = df_numeric.dropna().shape
    feature_cols = [c for c in df_numeric.columns if c != "target"]
    if len(feature_cols) == 0:
        raise ValueError(
            "No numeric feature columns found. Logistic Regression requires input features."
        )
    y = df_numeric["target"].dropna()
    if y.nunique() < 2:
        raise ValueError(
            f"Classification requires at least 2 classes, found {y.nunique()}."
        )
    X = df_numeric[feature_cols].loc[y.index]

    lr = LogisticRegression(C=C, max_iter=max_iter)
    lr.fit(X, y)
    preds = lr.predict(X)
    probs = lr.predict_proba(X) if hasattr(lr, "predict_proba") else None

    report = classification_report(y, preds, output_dict=True)
    conf_mat = confusion_matrix(y, preds)

    # ROC curve plot
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
        "input_shape": [rows, cols],
        "numeric_shape": [numeric_rows, numeric_cols],
        "target_classes": y.unique().tolist(),
        "class_counts": y.value_counts().to_dict(),
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "coefficients": lr.coef_.tolist(),
        "intercept": lr.intercept_.tolist(),
        "image_base64": img_base64,
    }


@router.post("/run")
async def run_model(payload: dict, db: AsyncSession = Depends(get_db)):
    dataset_id = payload.get("dataset_id")
    model_name = payload.get("model_name")

    if not dataset_id or not model_name:
        raise HTTPException(status_code=400, detail="Missing dataset_id or model_name")

    # Fetch dataset from DB
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.cleaned_data:
        raise HTTPException(
            status_code=400,
            detail="The selected dataset is not suitable for this model. Please ensure it has valid cleaned data.",
        )

    df = pd.DataFrame(dataset.cleaned_data)

    try:
        if model_name == "PCA_KMeans":
            n_clusters = int(payload.get("n_clusters", 3))
            output = run_model_pca_kmeans(df, n_clusters=n_clusters)
        elif model_name == "RandomForest":
            n_estimators = int(payload.get("n_estimators", 100))
            max_depth = payload.get("max_depth")
            max_depth = int(max_depth) if max_depth is not None else None
            output = run_model_random_forest(
                df, n_estimators=n_estimators, max_depth=max_depth
            )
        elif model_name == "LogisticRegression":
            C = float(payload.get("C", 1.0))
            max_iter = int(payload.get("max_iter", 100))
            output = run_model_logistic_regression(df, C=C, max_iter=max_iter)
        else:
            raise HTTPException(
                status_code=400, detail=f"Model '{model_name}' not implemented"
            )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _to_py(
        {
            "dataset_id": int(dataset_id),
            "model": model_name,
            "status": "success",
            **output,
        }
    )
