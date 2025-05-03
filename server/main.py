# ─────────────────────────────────────────────────────────────────────────────
#  server/main.py  –  CLEAN IMPORTS
#    • Std-lib   → no external deps
#    • 3rd-party → wheels from requirements.txt
#    • Local     → always prefix with  “server.”  so they resolve everywhere
# ─────────────────────────────────────────────────────────────────────────────

# ---------- Standard library ----------
import sys
import io
import base64
from datetime import datetime
from typing import List, Dict, Any

# ---------- Third-party ----------
from dotenv import load_dotenv
load_dotenv()                          # load .env first

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel

# ---------- Local / project ----------
from server.aws_client import get_s3, upload_bytes, S3_BUCKET
from server.database import get_async_db, engine, AsyncSessionLocal
from server.models import Dataset, Base
from server.schemas import Dataset as DatasetSchema, DatasetSummary
import server.schemas as schemas
from server.auth.userroutes import router as user_router, fastapi_users
from server.auth.userbase import User

# ─────────────────────────────────────────────────────────────────────────────
#  END IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

current_user = fastapi_users.current_user()
s3 = get_s3()

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
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router)


class CleanRequest(BaseModel):
    data: List[Dict[str, Any]]
    operations: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Backend is alive!"}


async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.on_event("startup")
async def on_startup():
    await init_models()


@app.post("/upload-csv", dependencies=[Depends(current_user)])
async def upload_csv(file: UploadFile = File(...)):
    # 1️⃣ read the bytes
    contents = await file.read()

    # 2️⃣ push to S3  (helper imported from aws_client.py)
    s3_key = upload_bytes(contents, file.filename)   # <-- NEW

    # 3️⃣ DataFrame / preview
    df = pd.read_csv(io.StringIO(contents.decode("ISO-8859-1")))

    buf = io.StringIO()
    df.info(buf=buf)
    info_output = buf.getvalue()

    insights = {
        "preview"       : df.head().to_dict(orient="records"),
        "shape"         : list(df.shape),
        "columns"       : df.columns.tolist(),
        "dtypes"        : df.dtypes.astype(str).to_dict(),
        "null_counts"   : df.isnull().sum().to_dict(),
        "summary_stats" : df.describe(include="all").fillna("").to_dict(),
        "info_output"   : info_output,
        "s3_key"        : s3_key,                    # <-- NEW
    }
    return insights


class DatasetCreate(BaseModel):
    title: str
    description: str
    filename: str
    raw_data: list[dict]
    s3_key: str


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
            s3_key=data.s3_key
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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get(
    "/datasets/{dataset_id}",
    response_model=DatasetSchema,                    # ← Pydantic schema here
    dependencies=[Depends(current_user)],
)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    row = await db.get(DatasetModel, dataset_id)     # async load
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return row                                       # FastAPI converts via orm_mode
                                



@app.get(
    "/datasets/{dataset_id}/heatmap",
    dependencies=[Depends(current_user)]
)
async def get_heatmap(
    dataset_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    obj = await db.get(DatasetModel, dataset_id)
    if not obj or not obj.raw_data:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")

    df = pd.DataFrame(obj.raw_data)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True,
                cmap="coolwarm", fmt=".2f")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()

    img_b64 = base64.b64encode(buf.read()).decode()
    return {"plot": f"data:image/png;base64,{img_b64}"}


@app.get(
    "/datasets/{dataset_id}/correlation",
    dependencies=[Depends(current_user)]
)
async def correlation_matrix(
    dataset_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    obj = await db.get(DatasetModel, dataset_id)
    if not obj or not obj.raw_data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.DataFrame(obj.raw_data).select_dtypes("number")
    if df.empty:
        raise HTTPException(status_code=400, detail="No numeric columns")

    corr = df.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar();  plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png"); buf.seek(0);  plt.close()

    img_b64 = base64.b64encode(buf.read()).decode()
    return {"heatmap": f"data:image/png;base64,{img_b64}"}

@app.post(
    "/clean-preview",
    dependencies=[Depends(current_user)],
    response_model=dict   
)
def preview_cleaning(data: Dict[str, Any], db: AsyncSession = Depends(get_async_db)):
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

app.mount(
    "/",
    StaticFiles(directory="client/dist", html=True),
    name="static",
)


