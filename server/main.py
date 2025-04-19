# server/main.py

# Standard library
import io
import base64
from datetime import datetime
from typing import List, Dict, Any

# Third-party
from fastapi import FastAPI, File, UploadFile, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel

# Internal modules
from database import SessionLocal
from models import Dataset
from database import engine
from models import Base

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CleanRequest(BaseModel):
    data: List[Dict[str, Any]]
    operations: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Backend is alive!"}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Capture df.info()
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_output = buffer.getvalue()

    # Build the insights dictionary
    insights = {
        "preview": df.head().to_dict(orient="records"),
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "summary_stats": df.describe(include="all").fillna("").to_dict(),
        "info_output": info_output
    }

    return insights



    # Save to database
    db = SessionLocal()
    try:
        new_dataset = Dataset(
            title=title,
            description=description,
            filename=file.filename,
            raw_data=df.to_dict(orient="records")
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)
    finally:
        db.close()

    return {
        "message": "Upload successful",
        "columns": df.columns.tolist(),
        "head": df.head().to_dict(orient="records")
    }



@app.post("/clean")
def clean_data(req: CleanRequest):
    df = pd.DataFrame(req.data)

    if req.operations.get("dropna"):
        df = df.dropna()
    for col, val in req.operations.get("fillna", {}).items():
        df[col] = df[col].fillna(val)
    if req.operations.get("lowercase_headers"):
        df.columns = [c.lower() for c in df.columns]

    cleaned_dict = df.to_dict(orient="records")
    filename = req.operations.get("filename", "unknown.csv")

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




# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/datasets")
def get_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "uploaded_at": d.uploaded_at,
            "cleaned_data": d.cleaned_data
        } for d in datasets
    ]



@app.get("/plot")
def get_plot():
    plt.figure()
    pd.Series([1, 3, 2, 5]).plot(kind="bar")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return {"plot": f"data:image/png;base64,{img_b64}"}

