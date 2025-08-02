from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os
import base64

API_KEY = os.getenv("API_KEY")

app = FastAPI(title="GPU Inference Service")

class InferenceRequest(BaseModel):
    dataset_id: str
    model: str
    params: dict | None = None

import matplotlib.pyplot as plt
import io, base64
import random

@app.post("/infer")
async def infer(req: InferenceRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Example: create a dummy sentiment distribution plot
    sentiments = ["Positive", "Neutral", "Negative"]
    counts = [random.randint(10, 100) for _ in sentiments]

    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(sentiments, counts, color=["green", "gray", "red"])
    ax.set_title(f"Sentiment Analysis for dataset {req.dataset_id}")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "dataset_id": req.dataset_id,
        "model": req.model,
        "status": "success",
        "image_base64": img_base64,
        "results": {"sentiment_counts": dict(zip(sentiments, counts))}
    }


