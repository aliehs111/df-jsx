# server/main.py
from database import SessionLocal
from models import Dataset

# Already at the top? If not, add these:
from pydantic import BaseModel
from typing import List, Dict, Any

class CleanRequest(BaseModel):
    data: List[Dict[str, Any]]
    operations: Dict[str, Any]



import io, base64
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import matplotlib.pyplot as plt

app = FastAPI()

# Allow your React dev server (http://localhost:5173) to call these APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend is alive!"}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    data = await file.read()
    df = pd.read_csv(io.BytesIO(data))

    # initial cleanup
    df = df.dropna()
    df.columns = df.columns.str.strip().str.lower()

    return {
        "columns": df.columns.tolist(),
        "head":    df.head().to_dict(orient="records")
    }


# ─── New: Cleaning endpoint ────────────────────────────────────────────────────

from database import SessionLocal
from models import Dataset

@app.post("/clean")
def clean_data(req: CleanRequest):
    df = pd.DataFrame(req.data)

    # Clean the data
    if req.operations.get("dropna"):
        df = df.dropna()
    for col, val in req.operations.get("fillna", {}).items():
        df[col] = df[col].fillna(val)
    if req.operations.get("lowercase_headers"):
        df.columns = [c.lower() for c in df.columns]

    cleaned_dict = df.to_dict(orient="records")
    filename = req.operations.get("filename", "unknown.csv")

    # Save to database
    db = SessionLocal()
    try:
        new_dataset = Dataset(
            filename=filename,
            cleaned_data=cleaned_dict
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)
    finally:
        db.close()

    return {"data": cleaned_dict}



@app.get("/plot")
def get_plot():
    plt.figure()
    pd.Series([1, 3, 2, 5]).plot(kind="bar")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return {"plot": f"data:image/png;base64,{img_b64}"}
