# server/main.py

# Standard library
import io
import base64
from datetime import datetime
from typing import List, Dict, Any

# Third-party
from fastapi import FastAPI, File, UploadFile, Depends, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import matplotlib
matplotlib.use("Agg")  # Avoid macOS GUI crash
import matplotlib.pyplot as plt
import seaborn as sns  # ✅ Add this


app = FastAPI()

origins = [
    "http://localhost:5173",  # Vite frontend
    "http://127.0.0.1:5173",  # Sometimes needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of trusted dev origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from sqlalchemy.orm import Session
from typing import List, Dict, Any
from sqlalchemy.orm import Session
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel

# Internal modules
from database import SessionLocal, get_db
from models import Dataset
from database import engine
from models import Base
import schemas
import models

Base.metadata.create_all(bind=engine)



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

class DatasetCreate(BaseModel):
    title: str
    description: str
    filename: str
    raw_data: list[dict]

@app.post("/datasets/save")
def save_dataset(data: DatasetCreate, db: Session = Depends(get_db)):
    try:
        dataset = Dataset(
            title=data.title,
            description=data.description,
            filename=data.filename,
            raw_data=data.raw_data,
            cleaned_data=None,
            categorical_mappings=None,
            normalization_params=None,
            column_renames=None
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        return {"message": "Dataset saved", "id": dataset.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving dataset: {e}")


@app.get("/datasets", response_model=List[schemas.DatasetSummary])
def get_all_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()

@app.get("/datasets/{dataset_id}", response_model=schemas.Dataset)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset



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


# fallback option for CORS
from fastapi.responses import JSONResponse

@app.options("/{full_path:path}")
async def preflight_handler():
    return JSONResponse(content={"detail": "CORS preflight OK"})



# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/datasets/{dataset_id}/heatmap")
def get_heatmap(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset or not dataset.raw_data:
        raise HTTPException(status_code=404, detail="Dataset not found or no raw data")
    
    df = pd.DataFrame(dataset.raw_data)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    img_b64 = base64.b64encode(buf.read()).decode()

    return {"plot": f"data:image/png;base64,{img_b64}"}


@app.get("/datasets/{dataset_id}/correlation")
def get_correlation_heatmap(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset or not dataset.raw_data:
        raise HTTPException(status_code=404, detail="Dataset or raw data not found")

    try:
        df = pd.DataFrame(dataset.raw_data)

        # Drop non-numeric columns if any
        df_numeric = df.select_dtypes(include=['number'])
        if df_numeric.empty:
            raise HTTPException(status_code=400, detail="No numeric data available for correlation")

        # Create correlation matrix
        corr = df_numeric.corr()

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.colorbar()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")

        return {"heatmap": f"data:image/png;base64,{encoded}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate heatmap: {e}")



@app.get("/plot")
def get_plot():
    plt.figure()
    pd.Series([1, 3, 2, 5]).plot(kind="bar")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return {"plot": f"data:image/png;base64,{img_b64}"}

