# server/main.py
# ---------- Standard library ----------
import sys
import os

print(
    "📢 Connecting to database at:",
    os.getenv("DATABASE_URL") or os.getenv("JAWSDB_URL"),
)
from pathlib import Path
import io
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
import time

# ---------- Third-party ----------
from dotenv import load_dotenv


load_dotenv()  # load .env first

import logging

logger = logging.getLogger(__name__)

import json
import pandas as pd
import numpy as np
import matplotlib
import math
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
from fastapi.encoders import jsonable_encoder
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Depends,
    HTTPException,
    status,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.requests import Request
from fastapi.routing import APIRoute
from botocore.exceptions import ClientError
from fastapi import status

from sqlalchemy import select, case
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel

from starlette.responses import StreamingResponse
from io import BytesIO

# ---------- Local / project ----------
from server.schemas import DatasetCreate

from server.aws_client import get_s3, upload_bytes, S3_BUCKET
from server.database import get_async_db, engine, AsyncSessionLocal
from server.models import Dataset as DatasetModel
from server.models import Base
import server.schemas as schemas
from server.schemas import (
    ProcessRequest,
    CleanOps,
    PreprocessOps,
    DownloadURLResponse,
    CleanPreviewRequest,
)

from server.auth.userroutes import router as user_router, fastapi_users
from server.auth.userbase import User
from server.auth.userroutes import current_user
from server.schemas import ProcessRequest

from server.utils.encoders import _to_py
from server.auth.userroutes import router as user_router

from server.routers.datasets import router as datasets_router
from server.routers.modelrunner import router as model_runner_router
from server.routers.insights import router as insights_router
from server.routers import modelrunner
from server.routers import databot
from server.routers import datarows
from openai import OpenAI

from server.schemas import CleanPreviewRequest
from server.models import Dataset as DatasetModel
from server.auth.userroutes import current_user


# ─────────────────────────────────────────────────────────────────────────────
#  END IMPORTS
# ─────────────────────────────────────────────────────────────────────────────





client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

current_user = fastapi_users.current_user()
s3 = get_s3()


# --- Unique ID generator (define BEFORE FastAPI instance)
def custom_generate_unique_id(route: APIRoute):
    tag = route.tags[0] if route.tags else "default"
    return f"{tag}_{route.name}"


# -- for cleaned datasets
def sanitize_floats(o):
    if isinstance(o, dict):
        return {k: sanitize_floats(v) for k, v in o.items()}
    if isinstance(o, float):
        # turn any NaN or Infinity into None
        return None if (math.isnan(o) or math.isinf(o)) else o
    return o


# --- Create FastAPI app with custom ID function
app = FastAPI(generate_unique_id_function=custom_generate_unique_id)

# --- CORS settings ---
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

PROD_ORIGINS = [
    "https://df-jsx-ab06705b49fb.herokuapp.com",
]

ENV = os.getenv("ENV", "development")  # set ENV=production in Heroku config

origins = PROD_ORIGINS if ENV == "production" else DEV_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(user_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(insights_router, prefix="/api", tags=["insights"])
app.include_router(model_runner_router, prefix="/api", tags=["models"])
app.include_router(databot.router, prefix="/api/databot")
app.include_router(datarows.router, prefix="/api")

class CleanRequest(BaseModel):
    data: List[Dict[str, Any]]
    operations: Dict[str, Any]


@app.get("/api/health")
def read_root():
    return {"message": "Backend is alive!"}


async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.on_event("startup")
async def on_startup():
    await init_models()
    # ✅ Check OpenAI API key is loaded
    openai_key_loaded = bool(os.getenv("OPENAI_API_KEY"))
    if not openai_key_loaded:
        print("⚠️  WARNING: OPENAI_API_KEY not found. Databot will not work.")
    else:
        print("✅ OpenAI API key loaded successfully.")
        # Optionally, instantiate the client so it's ready
        global openai_client
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.post("/api/upload-csv", dependencies=[Depends(current_user)])
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    contents = await file.read()
    s3_key = upload_bytes(contents, file.filename)

    # Try multiple encodings to read the CSV
    df = None
    encodings = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
    for encoding in encodings:
        try:
            df = pd.read_csv(io.StringIO(contents.decode(encoding)))
            break
        except Exception:
            continue

    if df is None:
        raise HTTPException(status_code=400, detail="Could not decode CSV with common encodings.")

    # Clean up infinities and NaNs
    df = df.replace([np.inf, -np.inf], np.nan)

    # Capture df.info() output
    buf = io.StringIO()
    df.info(buf=buf)
    info_output = buf.getvalue()

    # Summary statistics
    summary = df.describe(include="all").replace([np.inf, -np.inf], np.nan).fillna("")

    # Column metadata
    column_metadata = {
        col: {
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "null_count": int(df[col].isnull().sum()),
        }
        for col in df.columns
    }

    # Sanitize preview for JSON
    def sanitize_value(val):
        if isinstance(val, (np.floating, float)):
            return None if (math.isnan(val) or math.isinf(val)) else float(val)
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            return str(val)
        return str(val) if val is not None else None

    preview = [
        {k: sanitize_value(v) for k, v in row.items()}
        for row in df.head(10).to_dict(orient="records")
    ]

    insights = {
        "preview": preview,
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "summary_stats": summary.astype(str).to_dict(),
        "info_output": info_output,
        "s3_key": s3_key,
        "s3_key_cleaned": None,
        "column_metadata": column_metadata,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "has_missing_values": bool(df.isnull().values.any()),
        "has_cleaned_data": False,
    }

    return jsonable_encoder(insights)



@app.post("/api/datasets/save", dependencies=[Depends(current_user)])
async def save_dataset(data: DatasetCreate, db: AsyncSession = Depends(get_async_db)):
    try:
        dataset = DatasetModel(
            title=data.title,
            description=data.description,
            filename=data.filename,
            s3_key=data.s3_key,
            s3_key_cleaned=None,
            categorical_mappings=data.categorical_mappings,
            normalization_params=data.normalization_params,
            column_renames=data.column_renames,
            target_column=data.target_column,
            selected_features=data.selected_features,
            excluded_columns=data.excluded_columns,
            feature_engineering_notes=data.feature_engineering_notes,
            column_metadata=data.column_metadata,     # ✅ from upload_csv
            n_rows=data.n_rows,                      # ✅ from upload_csv
            n_columns=data.n_columns,                # ✅ from upload_csv
            has_missing_values=bool(data.has_missing_values)
            if data.has_missing_values is not None else None,
            processing_log=(
                " | ".join(data.processing_log) if data.processing_log else None
            ),
            current_stage=data.current_stage or "uploaded",  # ✅ default stage
            has_cleaned_data=False,
            extra_json_1=data.extra_json_1,
            extra_txt_1=data.extra_txt_1,
        )

        db.add(dataset)
        await db.flush()     # ensures ID is generated
        await db.refresh(dataset)
        await db.commit()

        return {"id": dataset.id}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))



@app.get(
    "/api/datasets",
    response_model=List[schemas.DatasetSummary],
    dependencies=[Depends(current_user)],
)
async def list_datasets(db: AsyncSession = Depends(get_async_db)):
    stmt = (
        select(
            DatasetModel.id,
            DatasetModel.title,
            DatasetModel.description,
            DatasetModel.filename,
            DatasetModel.s3_key,
            DatasetModel.s3_key_cleaned,
            DatasetModel.uploaded_at,
            DatasetModel.has_cleaned_data,
        )
        .order_by(DatasetModel.uploaded_at.desc())
        .limit(100)
    )

    result = await db.execute(stmt)
    rows = result.fetchall()

    return [
        schemas.DatasetSummary(
            id=row.id,
            title=row.title,
            description=row.description,
            filename=row.filename,
            s3_key=row.s3_key,
            s3_key_cleaned=row.s3_key_cleaned,
            uploaded_at=row.uploaded_at,
            has_cleaned_data=row.has_cleaned_data,
        )
        for row in rows
    ]


@app.get(
    "/api/datasets/{dataset_id}",
    response_model=schemas.DatasetOut,
    dependencies=[Depends(current_user)],
)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    row = await db.get(DatasetModel, dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Fetch CSV preview from S3 if s3_key exists
    preview_data = None
    if row.s3_key:
        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key=row.s3_key)
            content = response["Body"].read()
            encodings = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(
                        io.StringIO(content.decode(encoding, errors="replace"))
                    )
                    logger.info(f"Loaded CSV {row.s3_key} with encoding {encoding}")
                    break
                except Exception as e:
                    logger.warning(
                        f"Failed with encoding {encoding} for {row.s3_key}: {str(e)}"
                    )
            if df is None:
                raise Exception("Failed to parse CSV with any encoding")

            # Log raw data
            logger.info(f"Dataset {dataset_id} columns: {df.columns.tolist()}")
            logger.info(f"Dataset {dataset_id} dtypes: {df.dtypes.to_dict()}")
            logger.info(
                f"Dataset {dataset_id} head: {df.head(5).to_dict(orient='records')}"
            )

            # Sanitize preview data
            def sanitize_value(val):
                if isinstance(val, (np.floating, float)):
                    if math.isnan(val) or math.isinf(val):
                        return None
                    return float(val)
                if isinstance(val, (np.integer, int)):
                    return int(val)
                if isinstance(val, (np.bool_, bool)):
                    return bool(val)
                if isinstance(val, (pd.Timestamp, np.datetime64)):
                    return str(val)
                return str(val) if val is not None else None

            preview_data = [
                {k: sanitize_value(v) for k, v in record.items()}
                for record in df.head(5).to_dict(orient="records")
            ]
        except Exception as e:
            logger.error(
                f"Failed to fetch or process S3 data for {row.s3_key}: {str(e)}"
            )
            alerts.append(f"Failed to load preview data: {str(e)}")

    # Prepare response
    response = schemas.DatasetOut(
        id=row.id,
        title=row.title,
        description=row.description,
        filename=row.filename,
        s3_key=row.s3_key,
        s3_key_cleaned=row.s3_key_cleaned,
        uploaded_at=row.uploaded_at,
        categorical_mappings=row.categorical_mappings,
        normalization_params=row.normalization_params,
        column_renames=row.column_renames,
        target_column=row.target_column,
        selected_features=row.selected_features,
        excluded_columns=row.excluded_columns,
        feature_engineering_notes=row.feature_engineering_notes,
        column_metadata=row.column_metadata,
        n_rows=row.n_rows,
        n_columns=row.n_columns,
        has_missing_values=row.has_missing_values,
        processing_log=row.processing_log,
        current_stage=row.current_stage,
        has_cleaned_data=row.has_cleaned_data,
        extra_json_1=row.extra_json_1,
        extra_txt_1=row.extra_txt_1,
        preview_data=preview_data,
    )

    # Log response
    logger.info(f"Dataset {dataset_id} response: {jsonable_encoder(response)}")

    return response


@app.get(
    "/api/datasets/{dataset_id}/insights",
    dependencies=[Depends(current_user)],
)
async def get_dataset_insights(
    dataset_id: int,
    which: str = Query("raw", regex="^(raw|cleaned)$"),
    limit: int = Query(5, ge=1, le=100),
    db: AsyncSession = Depends(get_async_db),
):
    # Fetch dataset
    ds = await db.get(DatasetModel, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Pick raw vs. cleaned
    s3_key = ds.s3_key if which == "raw" else ds.s3_key_cleaned
    if which == "cleaned" and (not ds.has_cleaned_data or not ds.s3_key_cleaned):
        logger.warning(
            f"Dataset {dataset_id}: No cleaned data available (has_cleaned_data={ds.has_cleaned_data}, s3_key_cleaned={ds.s3_key_cleaned})"
        )
        raise HTTPException(
            status_code=400, detail="No cleaned data available for this dataset"
        )
    if not s3_key:
        logger.warning(
            f"Dataset {dataset_id}: No {'raw' if which == 'raw' else 'cleaned'} data available"
        )
        raise HTTPException(
            status_code=400,
            detail=f"No {'raw' if which == 'raw' else 'cleaned'} data available for this dataset",
        )

    # Fetch CSV from S3
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        content = response["Body"].read()
        encodings = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    io.StringIO(content.decode(encoding, errors="replace"))
                )
                logger.info(f"Loaded CSV {s3_key} with encoding {encoding}")
                break
            except Exception as e:
                logger.warning(
                    f"Failed with encoding {encoding} for {s3_key}: {str(e)}"
                )
        if df is None:
            raise Exception("Failed to parse CSV with any encoding")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.error(f"Dataset {dataset_id}: S3 key {s3_key} does not exist")
            raise HTTPException(
                status_code=400,
                detail=f"No {'raw' if which == 'raw' else 'cleaned'} data file exists in S3",
            )
        logger.error(f"Failed to load CSV {s3_key} for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Failed to load CSV from S3: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to load CSV {s3_key} for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Failed to load CSV from S3: {str(e)}"
        )

    # Validate DataFrame
    if df.empty:
        logger.warning(f"Dataset {dataset_id}: Empty DataFrame")
        raise HTTPException(status_code=400, detail="Empty DataFrame")

    # Sanitize values
    def sanitize_value(val):
        if isinstance(val, (np.floating, float)):
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            return str(val)
        return str(val) if val is not None else None

    # Run df.info()
    buf = io.StringIO()
    df.info(buf=buf)
    info_output = buf.getvalue()

    # Prepare response
    response = {
        "preview": [
            {k: sanitize_value(v) for k, v in record.items()}
            for record in df.head(limit).to_dict(orient="records")
        ],
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": {
            col: sanitize_value(val) for col, val in df.isnull().sum().to_dict().items()
        },
        "summary_stats": {},  # Skip to avoid serialization issues
        "info_output": info_output,
    }

    logger.info(
        f"Dataset {dataset_id} insights response: {json.dumps(response, default=str)}"
    )
    return response


@app.delete(
    "/api/datasets/{dataset_id}",
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


@app.post(
    "/api/datasets/{dataset_id}/clean",
    dependencies=[Depends(current_user)],
)
async def clean_data(
    dataset_id: int,
    req: CleanRequest,
    db: AsyncSession = Depends(get_async_db),
):
    ds = await db.get(DatasetModel, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.DataFrame(req.data)

    if req.operations.get("dropna"):
        df = df.dropna()

    for col, val in req.operations.get("fillna", {}).items():
        df[col] = df[col].fillna(val)

    if req.operations.get("lowercase_headers"):
        df.columns = [c.lower() for c in df.columns]

    # Save cleaned data back to the existing dataset
    cleaned_dict = df.to_dict(orient="records")
    ds.cleaned_data = cleaned_dict

    await db.commit()
    await db.refresh(ds)

    return {"id": ds.id, "cleaned_row_count": len(cleaned_dict)}


@app.post(
    "/api/datasets/{dataset_id}/process",
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

    if ops.fillna_strategy:
        strat = ops.fillna_strategy
        if strat == "mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif strat == "median":
            df = df.fillna(df.median(numeric_only=True))
        elif strat == "mode":
            modes = df.mode(dropna=True).iloc[0].to_dict()
            df = df.fillna(modes)
        elif strat == "zero":
            df = df.fillna(0)

    if ops.lowercase_headers:
        df.columns = [c.lower() for c in df.columns]

    if ops.remove_duplicates:
        df = df.drop_duplicates()

    renames = (
        {old: new for old, new in zip(ds.raw_data[0].keys(), df.columns)}
        if ops.lowercase_headers
        else {}
    )

    # ─── Preprocessing ──────────────────────────────────────────────
    norm_params = {}
    if payload.preprocess.scale in {"normalize", "standardize"}:
        for c in df.select_dtypes(include="number").columns:
            if payload.preprocess.scale == "normalize":
                mn, mx = df[c].min(), df[c].max()
                norm_params[c] = {"min": mn, "max": mx}
                if mx != mn:
                    df[c] = (df[c] - mn) / (mx - mn)
            else:
                mean, std = df[c].mean(), df[c].std()
                norm_params[c] = {"mean": mean, "std": std}
                if std:
                    df[c] = (df[c] - mean) / std

    norm_params = sanitize_floats(norm_params)
    cat_maps = {}
    if payload.preprocess.encoding == "label":
        from sklearn.preprocessing import LabelEncoder

        for c in df.select_dtypes(include="object").columns:
            le = LabelEncoder().fit(df[c].fillna(""))
            cat_maps[c] = dict(zip(le.classes_, le.transform(le.classes_)))
            df[c] = le.transform(df[c].fillna(""))
    elif payload.preprocess.encoding == "onehot":
        for c in df.select_dtypes(include="object").columns:
            cols = df[c].fillna("").unique().tolist()
            cat_maps[c] = cols
        df = pd.get_dummies(df, columns=list(cat_maps.keys()), dummy_na=False)

    # ─── Persist in DB and S3 ───────────────────────────────────────
    ds.normalization_params = norm_params
    cleaned_records = df.to_dict(orient="records")
    ds.column_renames = renames
    ds.cleaned_data = cleaned_records
    ds.normalization_params = norm_params
    ds.categorical_mappings = cat_maps

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    key = upload_bytes(buf.getvalue().encode("utf-8"), f"final_{ds.filename}")
    ds.s3_key = key

    # 🔽 🔽 🔽 INSERT THIS SECTION RIGHT HERE 🔽 🔽 🔽
    ds.n_rows, ds.n_columns = df.shape
    ds.has_missing_values = df.isnull().values.any()
    ds.current_stage = "processed"
    ds.processing_log = ds.processing_log or []
    ds.processing_log.append("Processed with clean+preprocess step")

    ds.column_metadata = {
        col: {
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "null_count": int(df[col].isnull().sum()),
        }
        for col in df.columns
    }
    # 🔼 🔼 🔼 END OF METADATA SECTION 🔼 🔼 🔼

    await db.commit()
    await db.refresh(ds)

    return {"id": ds.id, "s3_key": key}


@app.get(
    "/api/datasets/{dataset_id}/download",
    response_model=DownloadURLResponse,
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
            ExpiresIn=3600,
        )
    except ClientError as e:
        logger.error(f"Presigned URL generation failed: {e}")
        raise HTTPException(status_code=500, detail="Could not generate download link")

    return {"url": url}


logger = logging.getLogger("server.main")


@app.get(
    "/api/datasets/{dataset_id}/heatmap",
    dependencies=[Depends(current_user)],
)
async def get_dataset_heatmap(
    dataset_id: int,
    which: str = Query("raw", regex="^(raw|cleaned)$"),
    db: AsyncSession = Depends(get_async_db),
):
    # Fetch dataset
    ds = await db.get(DatasetModel, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Pick raw vs. cleaned
    s3_key = ds.s3_key if which == "raw" else ds.s3_key_cleaned
    if which == "cleaned" and (not ds.has_cleaned_data or not ds.s3_key_cleaned):
        logger.warning(f"Dataset {dataset_id}: No cleaned data available")
        raise HTTPException(status_code=400, detail="No cleaned data available")
    if not s3_key:
        logger.warning(f"Dataset {dataset_id}: No {which} data available")
        raise HTTPException(status_code=400, detail=f"No {which} data available")

    # Fetch CSV from S3
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        content = response["Body"].read()
        encodings = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    io.StringIO(content.decode(encoding, errors="replace"))
                )
                logger.info(f"Loaded CSV {s3_key} with encoding {encoding}")
                break
            except Exception as e:
                logger.warning(
                    f"Failed with encoding {encoding} for {s3_key}: {str(e)}"
                )
        if df is None:
            raise Exception("Failed to parse CSV with any encoding")
    except Exception as e:
        logger.error(f"Failed to load CSV {s3_key} for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Failed to load CSV from S3: {str(e)}"
        )

    # Validate DataFrame
    if df.empty:
        logger.warning(f"Dataset {dataset_id}: Empty DataFrame")
        raise HTTPException(status_code=400, detail="Empty DataFrame")

    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not numeric_cols:
        logger.warning(f"Dataset {dataset_id}: No numeric columns for heatmap")
        raise HTTPException(
            status_code=400, detail="No numeric columns available for heatmap"
        )

    # Generate heatmap
    try:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
    except Exception as e:
        logger.error(f"Failed to generate heatmap for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate heatmap: {str(e)}"
        )

    # Upload plot to S3
    plot_key = f"plots/heatmap_{dataset_id}_{which}_{int(time.time())}.png"
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=plot_key,
            Body=buf.getvalue(),
            ContentType="image/png",
        )
        logger.info(f"Dataset {dataset_id}: Uploaded heatmap to S3: {plot_key}")
    except Exception as e:
        logger.error(
            f"Failed to upload heatmap to S3 for dataset {dataset_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to upload heatmap: {str(e)}"
        )

    # Generate presigned URL
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": plot_key},
            ExpiresIn=3600,
        )
        logger.info(f"Dataset {dataset_id}: Generated presigned URL for heatmap")
    except Exception as e:
        logger.error(
            f"Failed to generate presigned URL for dataset {dataset_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to generate presigned URL: {str(e)}"
        )

    return {"plot": url}


@app.get("/api/datasets/{dataset_id}/correlation", dependencies=[Depends(current_user)])
async def correlation_matrix(dataset_id: int, db: AsyncSession = Depends(get_async_db)):
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
    plt.colorbar()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    img_b64 = base64.b64encode(buf.read()).decode()
    return {"heatmap": f"data:image/png;base64,{img_b64}"}




from fastapi import Request

@datasets_router.post("/datasets/{dataset_id}/clean-preview")
async def clean_preview(
    dataset_id: int,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(current_user),
):
    req = await request.json()  # ✅ Explicitly load JSON

    """Generate preview of dataset before and after cleaning operations."""
    logger.info(f"Dataset {dataset_id} clean-preview operations: {req.get('operations')}")
    ds = await db.get(DatasetModel, dataset_id)
    if not ds or not ds.s3_key:
        raise HTTPException(404, "Dataset or raw data not found")

    # Load CSV from S3
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=ds.s3_key)
        content = response["Body"].read()
        encodings = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(io.StringIO(content.decode(encoding, errors="replace")))
                logger.info(f"Loaded CSV {ds.s3_key} with encoding {encoding}")
                break
            except Exception as e:
                logger.warning(f"Failed with encoding {encoding}: {str(e)}")
        if df is None:
            raise Exception("Failed to parse CSV")
    except Exception as e:
        logger.error(f"Failed to load CSV {ds.s3_key}: {str(e)}")
        raise HTTPException(400, f"Failed to load CSV: {str(e)}")

    # Compute before_stats on original data
    def get_stats(df):
        return {
            "shape": list(df.shape),
            "null_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
    before_stats = get_stats(df)
    logger.info(f"Before stats dtypes: {before_stats['dtypes']}")  # Debug log

    # Create copy for cleaning
    df_cleaned = df.copy()
    alerts = []
    ops = req.get("operations", {})

    # Lowercase Headers
    if ops.get("lowercase_headers"):
        df_cleaned.columns = [c.lower() for c in df_cleaned.columns]
        ds.column_renames = {old: new for old, new in zip(df.columns, df_cleaned.columns)}
        alerts.append("Applied lowercase headers.")

    # Drop NA
    if ops.get("dropna"):
        df_cleaned = df_cleaned.dropna()
        alerts.append(f"Dropped {before_stats['shape'][0] - df_cleaned.shape[0]} rows with NA.")

    # Remove Duplicates
    if ops.get("remove_duplicates"):
        before_count = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        alerts.append(f"Removed {before_count - len(df_cleaned)} duplicates.")

    # Column-Specific Imputation
    if ops.get("fillna_strategy") and ops.get("selected_columns", {}).get("fillna"):
        for col in ops["selected_columns"]["fillna"]:
            if col not in df_cleaned.columns:
                alerts.append(f"Column '{col}' not found for imputation.")
                continue
            try:
                if ops["fillna_strategy"] == "mean" and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                    alerts.append(f"Filled '{col}' with mean.")
                elif ops["fillna_strategy"] == "median" and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                    alerts.append(f"Filled '{col}' with median.")
                elif ops["fillna_strategy"] == "mode":
                    mode_val = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else None
                    if mode_val is not None:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                        alerts.append(f"Filled '{col}' with mode.")
                elif ops["fillna_strategy"] == "zero":
                    df_cleaned[col] = df_cleaned[col].fillna(0)
                    alerts.append(f"Filled '{col}' with zero.")
                elif ops["fillna_strategy"] == "knn" and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    numeric_cols = [c for c in ops["selected_columns"]["fillna"] if c in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[c])]
                    if numeric_cols:
                        imputer = KNNImputer(n_neighbors=5)
                        df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])
                        alerts.append(f"Applied KNN imputation to '{col}'.")
                else:
                    alerts.append(f"Invalid imputation strategy for '{col}'.")
            except Exception as e:
                alerts.append(f"Failed to impute '{col}': {str(e)}")

    # Column-Specific Scaling
    norm_params = {}
    if ops.get("scale") and ops.get("selected_columns", {}).get("scale"):
        for col in ops["selected_columns"]["scale"]:
            if col not in df_cleaned.columns or not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                alerts.append(f"Column '{col}' not numeric or not found for scaling.")
                continue
            try:
                if ops["scale"] == "normalize":
                    mn, mx = df_cleaned[col].min(), df_cleaned[col].max()
                    if mx != mn:
                        df_cleaned[col] = (df_cleaned[col] - mn) / (mx - mn)
                        norm_params[col] = {"min": float(mn), "max": float(mx)}
                        alerts.append(f"Normalized '{col}'.")
                elif ops["scale"] == "standardize":
                    mean, std = df_cleaned[col].mean(), df_cleaned[col].std()
                    if std:
                        df_cleaned[col] = (df_cleaned[col] - mean) / std
                        norm_params[col] = {"mean": float(mean), "std": float(std)}
                        alerts.append(f"Standardized '{col}'.")
                elif ops["scale"] == "robust":
                    Q1, Q3 = df_cleaned[col].quantile(0.25), df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR:
                        df_cleaned[col] = (df_cleaned[col] - df_cleaned[col].median()) / IQR
                        norm_params[col] = {"Q1": float(Q1), "Q3": float(Q3), "median": float(df_cleaned[col].median())}
                        alerts.append(f"Robust-scaled '{col}'.")
            except Exception as e:
                alerts.append(f"Failed to scale '{col}': {str(e)}")

    # Column-Specific Encoding
    cat_maps = {}
    if ops.get("encoding") and ops.get("selected_columns", {}).get("encoding"):
        valid_cols = [
            c for c in ops["selected_columns"]["encoding"]
            if c in df_cleaned.columns and not pd.api.types.is_numeric_dtype(df_cleaned[c])
        ]
        if not valid_cols:
            alerts.append("No valid categorical columns selected for encoding.")
        else:
            try:
                if ops["encoding"] == "onehot":
                    cat_maps = {c: df_cleaned[c].fillna("").unique().tolist() for c in valid_cols}
                    df_cleaned = pd.get_dummies(df_cleaned, columns=valid_cols, dummy_na=False)
                    alerts.append(f"One-hot encoded {valid_cols}.")
                elif ops["encoding"] in ["label", "ordinal"]:
                    for col in valid_cols:
                        le = LabelEncoder().fit(df_cleaned[col].fillna(""))
                        cat_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
                        df_cleaned[col] = le.transform(df_cleaned[col].fillna(""))
                        alerts.append(f"{'Label' if ops['encoding'] == 'label' else 'Ordinal'}-encoded '{col}'.")
            except Exception as e:
                alerts.append(f"Failed to encode columns: {str(e)}")

    # Column-Specific Outlier Handling
    if ops.get("outlier_method") and ops.get("selected_columns", {}).get("outliers"):
        for col in ops["selected_columns"]["outliers"]:
            if col not in df_cleaned.columns or not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                alerts.append(f"Column '{col}' not numeric or not found for outlier handling.")
                continue
            try:
                if ops["outlier_method"] == "iqr":
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df_cleaned = df_cleaned[~((df_cleaned[col] < (Q1 - 1.5 * IQR)) | (df_cleaned[col] > (Q3 + 1.5 * IQR)))]
                    alerts.append(f"Removed outliers in '{col}' using IQR.")
                elif ops["outlier_method"] == "zscore":
                    df_cleaned = df_cleaned[df_cleaned[col].notna()]
                    df_cleaned = df_cleaned[abs(stats.zscore(df_cleaned[col])) < 3]
                    alerts.append(f"Removed outliers in '{col}' using Z-score.")
                elif ops["outlier_method"] == "cap":
                    p5, p95 = df_cleaned[col].quantile(0.05), df_cleaned[col].quantile(0.95)
                    df_cleaned[col] = df_cleaned[col].clip(lower=p5, upper=p95)
                    alerts.append(f"Capped outliers in '{col}' at 5th-95th percentiles.")
            except Exception as e:
                alerts.append(f"Failed to handle outliers in '{col}': {str(e)}")

    # Data Type Conversions
    for col, dtype in ops.get("conversions", {}).items():
        if col not in df_cleaned.columns:
            alerts.append(f"Column '{col}' not found for conversion.")
            continue
        try:
            if dtype == "numeric":
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
                alerts.append(f"Converted '{col}' to numeric.")
            elif dtype == "date":
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors="coerce")
                alerts.append(f"Converted '{col}' to date.")
            elif dtype == "category":
                df_cleaned[col] = df_cleaned[col].astype("category")
                alerts.append(f"Converted '{col}' to category.")
        except Exception as e:
            alerts.append(f"Failed to convert '{col}' to {dtype}: {str(e)}")

    # Feature Binning
    for col, bins in ops.get("binning", {}).items():
        if col not in df_cleaned.columns or not pd.api.types.is_numeric_dtype(df_cleaned[col]) or not bins:
            alerts.append(f"Invalid binning for '{col}': not numeric or bins not specified.")
            continue
        try:
            df_cleaned[f"{col}_binned"] = pd.cut(df_cleaned[col], bins=bins, labels=False, include_lowest=True)
            alerts.append(f"Binned '{col}' into {bins} bins.")
        except Exception as e:
            alerts.append(f"Failed to bin '{col}' with {bins} bins: {str(e)}")

    # Visualization
    vis_image_base64 = None
    if len(df_cleaned.select_dtypes(include=["number"]).columns) > 1:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            corr = df_cleaned.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            vis_image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            alerts.append("Generated correlation heatmap.")
        except Exception as e:
            alerts.append(f"Failed to generate visualization: {str(e)}")

    # Validation Checks
    if df_cleaned.isnull().values.any():
        alerts.append("Warning: Missing values remain after cleaning.")
    if df_cleaned.duplicated().any():
        alerts.append("Warning: Duplicates remain after processing.")

    after_stats = get_stats(df_cleaned)
    logger.info(f"After stats dtypes: {after_stats['dtypes']}")  # Debug log

    # Sanitize Preview
    def sanitize_value(val):
        if isinstance(val, (np.floating, float)):
            return None if (math.isnan(val) or math.isinf(val)) else float(val)
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            return str(val)
        return str(val) if val is not None else None

    preview = [
        {k: sanitize_value(v) for k, v in record.items()}
        for record in df_cleaned.head(10).to_dict(orient="records")
    ]

    # Save to DB and S3 if requested
    if req.get("save"):
        try:
            ds.cleaned_data = df_cleaned.to_dict(orient="records")
            ds.n_rows, ds.n_columns = df_cleaned.shape
            ds.has_missing_values = df_cleaned.isnull().values.any()
            ds.current_stage = "cleaned"
            ds.processing_log = ds.processing_log or ""
            ds.processing_log += f"\nCleaned with operations: {json.dumps(ops, default=str)}\n" + "\n".join(alerts)
            ds.column_metadata = {
                col: {
                    "dtype": str(df_cleaned[col].dtype),
                    "n_unique": int(df_cleaned[col].nunique()),
                    "null_count": int(df_cleaned[col].isnull().sum()),
                    "min": float(df_cleaned[col].min()) if pd.api.types.is_numeric_dtype(df_cleaned[col]) else None,
                    "max": float(df_cleaned[col].max()) if pd.api.types.is_numeric_dtype(df_cleaned[col]) else None,
                    "mean": float(df_cleaned[col].mean()) if pd.api.types.is_numeric_dtype(df_cleaned[col]) else None,
                    "std": float(df_cleaned[col].std()) if pd.api.types.is_numeric_dtype(df_cleaned[col]) else None,
                }
                for col in df_cleaned.columns
            }
            ds.normalization_params = norm_params
            ds.categorical_mappings = cat_maps
            buf = io.StringIO()
            df_cleaned.to_csv(buf, index=False)
            ds.s3_key_cleaned = upload_bytes(buf.getvalue().encode("utf-8"), f"cleaned_{ds.filename}")
            await db.commit()
            await db.refresh(ds)
            alerts.append("Saved cleaned dataset to S3 and database.")
        except Exception as e:
            await db.rollback()
            alerts.append(f"Failed to save cleaned data: {str(e)}")
            raise HTTPException(500, f"Failed to save cleaned data: {str(e)}")

    response = {
        "before_stats": before_stats,
        "after_stats": after_stats,
        "alerts": alerts,
        "preview": preview,
        "vis_image_base64": vis_image_base64,
        "saved": req.get("save", False),
    }
    logger.info(f"Dataset {dataset_id} clean-preview response: {json.dumps(response, default=str)}")
    return response


@app.get("/api/plot")
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




for route in app.routes:
    if isinstance(route, APIRoute):
        print(f"{route.path} → {route.methods}")
    else:
        print(f"{route.path} → [non-APIRoute: {type(route).__name__}]")

