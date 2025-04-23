# server/main.py
import sys
print("✅ Running Python from:", sys.executable)


from models import Base, Dataset
from auth import userbase  # Import this too so the table gets created
from database import engine


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
# --- Setup FastAPI app ---
from fastapi import FastAPI, File, UploadFile, Depends, Form, Request, HTTPException, APIRouter



from auth.userroutes import router as user_router
from auth.userroutes import fastapi_users  
current_user = fastapi_users.current_user()


# --- Unique ID generator (define BEFORE FastAPI instance)
def custom_generate_unique_id(route: APIRoute):
    tag = route.tags[0] if route.tags else "default"
    return f"{tag}_{route.name}"

# --- Create FastAPI app with custom ID function
app = FastAPI(generate_unique_id_function=custom_generate_unique_id)

# --- CORS settings ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Include routers ---
app.include_router(user_router)

from auth.userroutes import router as user_router

# Standard library
import io
import base64
from datetime import datetime
from typing import List, Dict, Any

# Third-party


import matplotlib
matplotlib.use("Agg")  # Avoid macOS GUI crash
import matplotlib.pyplot as plt
import seaborn as sns  # ✅ Add this










def custom_generate_unique_id(route: APIRoute):
    tag = route.tags[0] if route.tags else "default"
    return f"{tag}_{route.name}"




app.include_router(user_router)

from sqlalchemy import select 
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel

# Internal modules
from database import AsyncSessionLocal, get_async_db

from models import Dataset
from database import engine
from models import Base
import schemas
import models





class CleanRequest(BaseModel):
    data: List[Dict[str, Any]]
    operations: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Backend is alive!"}


from sqlalchemy.ext.asyncio import create_async_engine
from database import engine, Base
from auth.userbase import User  # Make sure this import is here
from models import Dataset  # Same for any other models

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.on_event("startup")
async def on_startup():
    await init_models()


@app.post("/upload-csv", dependencies=[Depends(current_user)])
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("ISO-8859-1")))



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

@app.post("/datasets/save", dependencies=[Depends(current_user)])
async def save_dataset(
        data: DatasetCreate,
        db: AsyncSession = Depends(get_async_db)):   # ✅ AsyncSession here
    try:
        dataset = Dataset(
            title=data.title,
            description=data.description,
            filename=data.filename,
            raw_data=data.raw_data,
        )

        db.add(dataset)
        await db.flush()          # ✦ 1. push to DB, id is generated
        await db.refresh(dataset) # ✦ 2. pull the PK back
        await db.commit()         # ✦ 3. finalize transaction

        return {"id": dataset.id}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/datasets",
    response_model=list[schemas.DatasetSummary],
    dependencies=[Depends(current_user)],
)
async def list_datasets(db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(
        select(Dataset).order_by(Dataset.uploaded_at.desc())
    )
    rows = result.scalars().all()
    return [schemas.DatasetSummary.from_orm(row) for row in rows]




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
# from fastapi.responses import JSONResponse

# @app.options("/{full_path:path}")
# async def preflight_handler():
#     return JSONResponse(
#         content={"detail": "CORS preflight OK"},
#         headers={
#             "Access-Control-Allow-Origin": ", ".join(origins),
#             "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
#             "Access-Control-Allow-Headers": "*",
#             "Access-Control-Allow-Credentials": "true",
#         },
#     )



# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/datasets/{dataset_id}/heatmap")
def get_heatmap(dataset_id: int, db: Session = Depends(get_async_db)):
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
def get_correlation_heatmap(dataset_id: int, db: Session = Depends(get_async_db)):
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

@app.post("/clean-preview")
def preview_cleaning(data: Dict[str, Any], db: Session = Depends(get_async_db)):
    dataset_id = data.get("dataset_id")
    operations = data.get("operations", {})

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.DataFrame(dataset.raw_data)
    df_cleaned = df.copy()

    # --- Before Stats ---
    before = {
        "shape": df.shape,
        "null_counts": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

    # --- Lowercase Headers ---
    if operations.get("lowercase_headers"):
        df_cleaned.columns = [c.lower() for c in df_cleaned.columns]

    # --- Handle Missing Values ---
    strategy = operations.get("fillna_strategy")
    if strategy in {"mean", "median", "mode"}:
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().any():
                if strategy == "mean":
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                elif strategy == "median":
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                elif strategy == "mode":
                    mode_val = df_cleaned[col].mode()
                    if not mode_val.empty:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
    elif strategy == "zero":
        df_cleaned = df_cleaned.fillna(0)

    # --- Scale Data ---
    scale_method = operations.get("scale")
    numeric_cols = df_cleaned.select_dtypes(include="number").columns

    if scale_method == "normalize":
        for col in numeric_cols:
            min_val = df_cleaned[col].min()
            max_val = df_cleaned[col].max()
            if min_val != max_val:
                df_cleaned[col] = (df_cleaned[col] - min_val) / (max_val - min_val)
    elif scale_method == "standardize":
        for col in numeric_cols:
            mean = df_cleaned[col].mean()
            std = df_cleaned[col].std()
            if std > 0:
                df_cleaned[col] = (df_cleaned[col] - mean) / std

    # --- Encode Categorical Variables ---
    encoding = operations.get("encoding")
    if encoding in {"onehot", "label"}:
        cat_cols = df_cleaned.select_dtypes(include=["object", "category"]).columns.tolist()
        if encoding == "onehot":
            df_cleaned = pd.get_dummies(df_cleaned, columns=cat_cols)
        elif encoding == "label":
            from sklearn.preprocessing import LabelEncoder
            for col in cat_cols:
                le = LabelEncoder()
                df_cleaned[col] = le.fit_transform(df_cleaned[col])

    # --- After Stats ---
    after = {
        "shape": df_cleaned.shape,
        "null_counts": df_cleaned.isnull().sum().to_dict(),
        "dtypes": df_cleaned.dtypes.astype(str).to_dict()
    }

    return {
        "before_stats": before,
        "after_stats": after,
    }


@app.get("/plot")
def get_plot():
    plt.figure()
    pd.Series([1, 3, 2, 5]).plot(kind="bar")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return {"plot": f"data:image/png;base64,{img_b64}"}



