import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

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
        # Expect dataset rows in req.params.records
        records = req.params.get("records") if req.params else []
        if not records:
            raise HTTPException(
                status_code=400,
                detail="No records provided for anomaly detection"
            )

        df = pd.DataFrame(records)

        # Only keep numeric columns
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            raise HTTPException(
                status_code=400,
                detail="No numeric columns available for anomaly detection"
            )

        try:
            # Isolation Forest anomaly detection
            model = IsolationForest(contamination=0.1, random_state=42)
            preds = model.fit_predict(numeric_df)

            df["anomaly"] = preds
            anomalies = df[df["anomaly"] == -1].to_dict(orient="records")

            return {
                "dataset_id": req.dataset_id,
                "model": req.model,
                "status": "success",
                "n_records": len(df),
                "n_anomalies": len(anomalies),
                "anomalies": anomalies[:20],  # limit to first 20 for safety
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

    elif req.model == "TimeSeriesForecasting":
        # placeholder
        return {"status": "error", "note": "TimeSeriesForecasting not implemented yet"}

    else:
        raise HTTPException(status_code=400, detail=f"Model {req.model} not supported")


@app.on_event("startup")
async def startup_event():
    port = os.getenv("PORT", "8080")
    print(f"[INFO] GPU service starting on port {port}")



