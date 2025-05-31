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
from server.utils.encoders import _to_py

router = APIRouter(prefix="/models", tags=["models"])


def run_model_pca_kmeans(df: pd.DataFrame, n_clusters=3):
    df_numeric = df.select_dtypes(include="number").dropna()
    if df_numeric.shape[1] < 2:
        raise ValueError("Not enough numeric features for PCA")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df_numeric)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    clusters = kmeans.fit_predict(reduced)

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
        "n_clusters": n_clusters,
        "cluster_counts": dict(pd.Series(clusters).value_counts().sort_index()),
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "image_base64": img_base64,
    }


@router.post("/run")
async def run_model(payload: dict, db: AsyncSession = Depends(get_db)):
    dataset_id = payload.get("dataset_id")
    model_name = payload.get("model_name")

    if not dataset_id or not model_name:
        raise HTTPException(status_code=400, detail="Missing dataset_id or model_name")

    print("🧪 Payload received:", payload)

    # Fetch dataset from DB
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.cleaned_data:
        raise HTTPException(
            status_code=400, detail="No cleaned data available. Please clean the dataset first."
        )

    df = pd.DataFrame(dataset.cleaned_data)

    if model_name == "PCA_KMeans":
        try:
            n_clusters = int(payload.get("n_clusters", 3))  # fallback default
            output = run_model_pca_kmeans(df, n_clusters=n_clusters)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return _to_py({
            "dataset_id": int(dataset_id),
            "model": model_name,
            "status": "success",
            **output
        })

    # Optional: fallback if model_name is not supported
    raise HTTPException(status_code=400, detail=f"Model '{model_name}' not implemented")
