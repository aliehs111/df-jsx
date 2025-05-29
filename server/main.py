# ─────────────────────────────────────────────────────────────────────────────
#  server/main.py  –  CLEAN IMPORTS
#    • Std-lib   → no external deps
#    • 3rd-party → wheels from requirements.txt
#    • Local     → always prefix with  “server.”  so they resolve everywhere
# ─────────────────────────────────────────────────────────────────────────────


# ---------- Standard library ----------
import sys
import os
print("📢 Connecting to database at:", os.getenv("DATABASE_URL") or os.getenv("JAWSDB_URL"))
from pathlib import Path
import io
import base64
from datetime import datetime
from typing import List, Dict, Any

# ---------- Third-party ----------
from dotenv import load_dotenv
load_dotenv()                          # load .env first

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from botocore.exceptions import ClientError

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel

# ---------- Local / project ----------
from server.aws_client import get_s3, upload_bytes, S3_BUCKET
from server.database import get_async_db, engine, AsyncSessionLocal
from server.models import Dataset       as DatasetModel
from server.models import Base
import server.schemas      as schemas
import server.schemas as schemas
from server.auth.userroutes import router as user_router, fastapi_users
from server.auth.userbase import User
from server.auth.userroutes import current_user
from server.schemas       import ProcessRequest
from server.routers import insights

# ─────────────────────────────────────────────────────────────────────────────
#  END IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
def _to_py(obj):
    # """
    # Recursively convert numpy scalars to native Python types.
    # """
    if isinstance(obj, dict):
        return { _to_py(k): _to_py(v) for k, v in obj.items() }
    if isinstance(obj, list):
        return [ _to_py(v) for v in obj ]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

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
app.include_router(insights.router)

class CleanRequest(BaseModel):
    data: List[Dict[str, Any]]
    operations: Dict[str, Any]

@app.get("/health")
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

    # build a “safe” summary_stats
    summary = df.describe(include="all")
    # replace infinities with NaN, then turn all NaNs into empty string (or null)
    summary = summary.replace([np.inf, -np.inf], np.nan).fillna("")
    summary_dict = summary.astype(str).to_dict()   # cast every cell to string

    insights = {
        "preview"       : df.head().to_dict(orient="records"),
        "records"       : df.to_dict(orient="records"), 
        "shape"         : list(df.shape),
        "columns"       : df.columns.tolist(),
        "dtypes"        : df.dtypes.astype(str).to_dict(),
        "null_counts"   : df.isnull().sum().to_dict(),
        "summary_stats" : df.describe(include="all").fillna("").to_dict(),
        "info_output"   : info_output,
        "s3_key"        : s3_key,                    # <-- NEW
    }
    
      # ensure all numpy types are finally plain Python
    return jsonable_encoder(insights)


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
        dataset = DatasetModel(
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
    response_model=List[schemas.DatasetSummary],
    dependencies=[Depends(current_user)],
)
async def list_datasets(db: AsyncSession = Depends(get_async_db)):
    # only pull the lightweight fields, not the full JSON columns
    stmt = (
        select(
            DatasetModel.id,
            DatasetModel.title,
            DatasetModel.description,
            DatasetModel.filename,
            DatasetModel.s3_key,
            DatasetModel.uploaded_at,
        )
        .order_by(DatasetModel.uploaded_at.desc())
    )
    result = await db.execute(stmt)
    rows = result.all()  # a list of Row(id=…, title=…, …)

    # manually build your Pydantic summaries
    return [
        schemas.DatasetSummary(
            id=row.id,
            title=row.title,
            description=row.description,
            filename=row.filename,
            s3_key=row.s3_key,
            uploaded_at=row.uploaded_at,
        )
        for row in rows
    ]

@app.get(
    "/datasets/{dataset_id}",
    response_model=schemas.Dataset,       # Pydantic schema for a single dataset
    dependencies=[Depends(current_user)],
)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    # now DatasetModel is defined
    row = await db.get(DatasetModel, dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return row   # FastAPI will convert it to DatasetSchema via orm_mode

@app.get(
    "/datasets/{dataset_id}/insights",
    dependencies=[Depends(current_user)],
)
async def get_dataset_insights(
    dataset_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    # fetch the saved raw_data
    ds = await db.get(DatasetModel, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # build a DataFrame
    df = pd.DataFrame(ds.raw_data)

    # run df.info()
    buf = io.StringIO()
    df.info(buf=buf)
    info_output = buf.getvalue()

    # assemble exactly the same insights shape as /upload-csv
    return {
        "preview":       df.head().to_dict(orient="records"),
        "shape":         list(df.shape),
        "columns":       df.columns.tolist(),
        "dtypes":        df.dtypes.astype(str).to_dict(),
        "null_counts":   df.isnull().sum().to_dict(),
        "summary_stats": df.describe(include="all").fillna("").to_dict(),
        "info_output":   info_output,
    }

from fastapi import status

@app.delete(
    "/datasets/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(current_user)],
)
async def delete_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    print(f"🔴 delete_dataset called for id={dataset_id}")

    ds = await db.get(DatasetModel, dataset_id)
    if not ds:
        print("🔴 Dataset not found (404)")
        raise HTTPException(status_code=404, detail="Dataset not found")

    # delete the S3 object if there is one
    if ds.s3_key:
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=ds.s3_key)
            print(f"🔴 Deleted S3 key {ds.s3_key}")
        except Exception as e:
            print("🔴 S3 delete error:", e)

    # do the DB delete + commit
    await db.delete(ds)   
    await db.commit()
    print("🔴 Committed delete")

    # **debug step: try to fetch it again**
    still = await db.get(DatasetModel, dataset_id)
    print("🔴 After commit, db.get returned:", still)

    return




@app.post("/datasets/{dataset_id}/clean")
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

    db = AsyncSession = Depends(get_async_db),
    try:
        new_dataset = DatasetModel(
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
    db = AsyncSession = Depends(get_async_db),
    try:
        yield db
    finally:
        db.close()
                              
@app.post(
    "/datasets/{dataset_id}/process",
    dependencies=[Depends(current_user)],
)
async def process_dataset(
    dataset_id: int,
    payload: ProcessRequest,
    db: AsyncSession = Depends(get_async_db),
):
    # 1) load dataset
    ds = await db.get(DatasetModel, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    df = pd.DataFrame(ds.raw_data)

    # ─── Cleaning ───────────────────────────────────────────────────
    ops = payload.clean
    if ops.dropna:
        df = df.dropna()
    for col, val in ops.fillna.items():
        df[col] = df[col].fillna(val)
    if ops.lowercase_headers:
        df.columns = [c.lower() for c in df.columns]
    if ops.remove_duplicates:
        df = df.drop_duplicates()

    renames = (
        {old: new for old, new in zip(ds.raw_data[0].keys(), df.columns)}
        if ops.lowercase_headers else {}
    )

    # ─── Preprocessing ──────────────────────────────────────────────
    norm_params = {}
    if payload.preprocess.scale in {"normalize", "standardize"}:
        for c in df.select_dtypes("number"):
            if payload.preprocess.scale == "normalize":
                mn, mx = df[c].min(), df[c].max()
                norm_params[c] = {"min": mn, "max": mx}
                if mx != mn:
                    df[c] = (df[c] - mn) / (mx - mn)
            else:  # standardize
                mean, std = df[c].mean(), df[c].std()
                norm_params[c] = {"mean": mean, "std": std}
                if std:
                    df[c] = (df[c] - mean) / std

    cat_maps = {}
    if payload.preprocess.encoding == "label":
        from sklearn.preprocessing import LabelEncoder
        for c in df.select_dtypes("object"):
            le = LabelEncoder().fit(df[c].fillna(""))
            cat_maps[c] = dict(zip(le.classes_, le.transform(le.classes_)))
            df[c] = le.transform(df[c].fillna(""))
    elif payload.preprocess.encoding == "onehot":
        for c in df.select_dtypes("object"):
            cols = df[c].fillna("").unique().tolist()
            cat_maps[c] = cols
        df = pd.get_dummies(df, columns=list(cat_maps.keys()), dummy_na=False)

    # ─── Persist in DB and S3 ───────────────────────────────────────
    # Convert every nested  into plain Python
    cleaned_records = _to_py(df.to_dict(orient="records"))
    norm_py         = _to_py(norm_params)
    cat_py          = _to_py(cat_maps)

    # quick debug to verify types
    print("DEBUG types:", 
          "cleaned_records[0] types:", {k:type(v) for k,v in cleaned_records[0].items()},
          "norm_py types:", {k:type(v["min"]) for k,v in norm_py.items()} if norm_py else {},
          "cat_py types:", {k:type(v) for k,v in cat_py.items()} )

    ds.column_renames        = renames
    ds.cleaned_data          = cleaned_records
    ds.normalization_params  = norm_py
    ds.categorical_mappings  = cat_py

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    key = upload_bytes(buf.getvalue().encode("utf-8"), f"final_{ds.filename}")
    ds.s3_key = key

    await db.commit()
    await db.refresh(ds)

    return {
        "s3_key": key,
        "column_renames": renames,
        "normalization_params": norm_py,
        "categorical_mappings": cat_py,
    }

@app.get(
    "/datasets/{dataset_id}/download",
    dependencies=[Depends(current_user)],
)
async def download_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    ds = await db.get(DatasetModel, dataset_id)
    if not ds or not ds.s3_key:
        raise HTTPException(status_code=404, detail="Dataset or file not found")

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": ds.s3_key},
            ExpiresIn=3600,  # link valid for 1 hour
        )
    except ClientError as e:
        logger.error(f"Presigned URL generation failed: {e}")
        raise HTTPException(status_code=500, detail="Could not generate download link")

    return {"url": url}


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
    "/datasets/{dataset_id}/clean-preview",
    dependencies=[Depends(current_user)],
    response_model=dict   
)
async def preview_cleaning(data: Dict[str, Any], db: AsyncSession = Depends(get_async_db)):
    dataset_id = data.get("dataset_id")
    operations = data.get("operations", {})

    stmt = select(DatasetModel).filter(DatasetModel.id == dataset_id)
    result = await db.execute(stmt)
    dataset = result.scalars().first()
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







# Only mount the React build when running on Heroku (DYNO env var present)
if "DYNO" in os.environ:
    DIST = Path(__file__).resolve().parent.parent / "client" / "dist"
    if DIST.exists():
        app.mount("/", StaticFiles(directory=str(DIST), html=True), name="static")
    else:
        print(f"⚠️  No frontend build found at {DIST}, skipping static mount")
else:
    print("⚠️  Development mode: skipping static mount")


