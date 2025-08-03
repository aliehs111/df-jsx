import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os

# ✅ Safe Matplotlib setup for headless environments
import matplotlib
try:
    matplotlib.use("Agg")  # headless-safe backend
except Exception as e:
    print(f"[Warning] Could not set matplotlib backend: {e}")
import matplotlib.pyplot as plt

from transformers import pipeline
import io, base64
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

API_KEY = os.getenv("API_KEY")

app = FastAPI(title="GPU Inference Service")

class InferenceRequest(BaseModel):
    dataset_id: str
    model: str
    params: dict | None = None

# Load pipelines once (so GPU warms up)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/infer")
async def infer(req: InferenceRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

    if req.model == "Sentiment":
        texts = req.params.get("texts") if req.params else []
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided for sentiment analysis")

        results = sentiment_pipeline(texts)
        labels = [r['label'] for r in results]
        counts = {label: labels.count(label) for label in set(labels)}

        fig, ax = plt.subplots()
        ax.bar(counts.keys(), counts.values())
        ax.set_title("Sentiment Counts")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "dataset_id": req.dataset_id,
            "model": req.model,
            "status": "success",
            "results": results,
            "sentiment_counts": counts,
            "image_base64": img_base64,
        }

        elif req.model == "AnomalyDetection":
        records = req.params.get("records") if req.params else []
        if not records:
            raise HTTPException(status_code=400, detail="No records provided for anomaly detection")

        df = pd.DataFrame(records)
        print(f"[DEBUG] AnomalyDetection received {len(df)} records")
        print(f"[DEBUG] Columns: {df.columns.tolist()}")

        if df.empty:
            raise HTTPException(status_code=400, detail="Empty dataset received for anomaly detection")

        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Encode categorical columns dynamically
        for col in categorical_cols:
            try:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                numeric_cols.append(col)  # now encoded, treat as numeric
            except Exception as e:
                print(f"[WARN] Could not encode column {col}: {e}")

        # Use all numeric features
        feature_cols = [col for col in numeric_cols if col in df.columns]

        if not feature_cols:
            raise HTTPException(
                status_code=400,
                detail="No valid numeric feature columns found for anomaly detection"
            )

        model = IsolationForest(
            contamination=req.params.get("contamination", 0.05),
            random_state=42
        )
        df["anomaly"] = model.fit_predict(df[feature_cols])

        anomalies = df[df["anomaly"] == -1].to_dict(orient="records")

        # Plot bar chart of anomaly vs normal
        fig, ax = plt.subplots()
        counts = df["anomaly"].value_counts().to_dict()
        ax.bar(["Normal", "Anomaly"], [counts.get(1, 0), counts.get(-1, 0)])
        ax.set_title("Anomaly Detection Results")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "dataset_id": req.dataset_id,
            "model": req.model,
            "status": "success",
            "results": df.to_dict(orient="records"),
            "anomalies": anomalies,
            "sentiment_counts": {},  # keep response contract consistent
            "image_base64": img_base64,
        }


    elif req.model == "TimeSeriesForecasting":
        return {"status": "error", "note": "TimeSeriesForecasting not implemented yet"}

    raise HTTPException(status_code=400, detail=f"Model {req.model} not supported")


@app.on_event("startup")
async def startup_event():
    port = os.getenv("PORT", "8080")
    print(f"[INFO] GPU service starting on port {port}")



