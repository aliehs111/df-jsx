from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os

API_KEY = os.getenv("API_KEY")

app = FastAPI(title="GPU Inference Service")

class InferenceRequest(BaseModel):
    dataset_id: str
    model: str
    params: dict | None = None

# ✅ Add this health check route
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/infer")
async def infer(req: InferenceRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Placeholder for model logic
    return {
        "dataset_id": req.dataset_id,
        "model": req.model,
        "status": "success",
        "results": {"note": "Replace this with actual GPU inference logic"}
    }



