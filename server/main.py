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
app.include_router(databot.router, prefix="/api", tags=["Databot"])
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

    try:
        df = pd.read_csv(io.StringIO(contents.decode("ISO-8859-1")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    df = df.replace([np.inf, -np.inf], np.nan).where(pd.notna(df), None)
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].astype(object).where(df[col].notnull(), None)

    buf = io.StringIO()
    df.info(buf=buf)
    info_output = buf.getvalue()

    summary = df.describe(include="all").replace([np.inf, -np.inf], np.nan).fillna("")
    column_metadata = {
        col: {
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "null_count": int(df[col].isnull().sum()),
        }
        for col in df.columns
    }

    insights = {
        "preview": df.head().to_dict(orient="records"),
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
        "has_missing_values": bool(df.isnull().values.any()),  # Cast to Python bool
        "has_cleaned_data": False,
    }

    return jsonable_encoder(insights)


# @app.post("/api/datasets/save", dependencies=[Depends(current_user)])
# async def save_dataset(data: DatasetCreate, db: AsyncSession = Depends(get_async_db)):
#     try:
#         dataset = DatasetModel(
#             title=data.title,
#             description=data.description,
#             filename=data.filename,
#             s3_key=data.s3_key,
#             s3_key_cleaned=None,
#             categorical_mappings=data.categorical_mappings,
#             normalization_params=data.normalization_params,
#             column_renames=data.column_renames,
#             target_column=data.target_column,
#             selected_features=data.selected_features,
#             excluded_columns=data.excluded_columns,
#             feature_engineering_notes=data.feature_engineering_notes,
#             column_metadata=data.column_metadata,
#             n_rows=data.n_rows,
#             n_columns=data.n_columns,
#             has_missing_values=data.has_missing_values,
#             processing_log=(
#                 " | ".join(data.processing_log) if data.processing_log else None
#             ),
#             current_stage=data.current_stage,
#             has_cleaned_data=False,
#             extra_json_1=data.extra_json_1,
#             extra_txt_1=data.extra_txt_1,
#         )

#         db.add(dataset)
#         await db.flush()
#         await db.refresh(dataset)
#         await db.commit()

#         return {"id": dataset.id}
#     except Exception as e:
#         await db.rollback()
#         raise HTTPException(status_code=500, detail=str(e))
# server/main.py
# server/main.py
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


@app.post(
    "/api/datasets/{dataset_id}/clean-preview",
    dependencies=[Depends(current_user)],
    response_model=dict,
)
async def preview_cleaning(
    payload: CleanPreviewRequest, db: AsyncSession = Depends(get_async_db)
):
    dataset_id = payload.dataset_id
    operations = payload.operations or {}
    save = payload.save

    # Log operations
    logger.info(f"Dataset {dataset_id} operations: {operations}")

    # Fetch dataset
    stmt = select(DatasetModel).filter(DatasetModel.id == dataset_id)
    result = await db.execute(stmt)
    dataset = result.scalars().first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Fetch CSV from S3
    if not dataset.s3_key:
        raise HTTPException(status_code=400, detail="No raw data available")
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=dataset.s3_key)
        content = response["Body"].read()
        encodings = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    io.StringIO(content.decode(encoding, errors="replace"))
                )
                logger.info(f"Loaded CSV {dataset.s3_key} with encoding {encoding}")
                break
            except Exception as e:
                logger.warning(
                    f"Failed with encoding {encoding} for {dataset.s3_key}: {str(e)}"
                )
        if df is None:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load CSV from S3: unable to parse with encodings {encodings}",
            )
    except Exception as e:
        logger.error(
            f"Failed to fetch CSV {dataset.s3_key} for dataset {dataset_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=400, detail=f"Failed to fetch CSV from S3: {str(e)}"
        )

    df_cleaned = df.copy()
    alerts = []

    # Log original columns
    logger.info(f"Dataset {dataset_id} original columns: {df.columns.tolist()}")

    # Sanitization
    def sanitize_value(val):
        if isinstance(val, (np.floating, float)):
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        if isinstance(val, (np.integer, int)):
            return int(val)  # Convert np.int64 to int
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            return str(val)
        return str(val) if val is not None else None

    # Generate column_metadata if null
    if not dataset.column_metadata:
        try:
            dataset.column_metadata = {
                col: {
                    "dtype": str(df[col].dtype),
                    "n_unique": int(df[col].nunique(dropna=True)),
                    "null_count": int(df[col].isnull().sum()),
                }
                for col in df.columns
            }
            dataset.n_rows, dataset.n_columns = df.shape
            dataset.has_missing_values = bool(df.isnull().values.any())
            await db.commit()
            await db.refresh(dataset)
        except Exception as e:
            alerts.append(f"Failed to generate column_metadata: {str(e)}")
            logger.error(
                f"Metadata generation failed for dataset {dataset_id}: {str(e)}"
            )

    # Before stats
    before = {
        "shape": list(df.shape),
        "null_counts": {
            col: sanitize_value(val) for col, val in df.isnull().sum().to_dict().items()
        },
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    logger.info(f"Dataset {dataset_id} before_stats: {before}")

    # Apply cleaning operations
    if operations.get("lowercase_headers", False):
        logger.info(f"Dataset {dataset_id}: Applying lowercase_headers")
        df_cleaned.columns = [c.lower() for c in df_cleaned.columns]
        logger.info(
            f"Dataset {dataset_id} cleaned columns: {df_cleaned.columns.tolist()}"
        )

    strategy = operations.get("fillna_strategy")
    selected_columns = operations.get("selected_columns", [])
    if strategy in {"mean", "median", "mode", "zero"}:
        logger.info(
            f"Dataset {dataset_id}: Applying fillna_strategy={strategy} to columns={selected_columns or 'all'}"
        )
        for col in selected_columns or df_cleaned.columns:
            if col not in df_cleaned.columns:
                alerts.append(f"Column '{col}' not found")
                continue
            if strategy in {"mean", "median"} and not pd.api.types.is_numeric_dtype(
                df_cleaned[col]
            ):
                alerts.append(
                    f"Cannot apply {strategy} fillna to non-numeric column '{col}'"
                )
                continue
            try:
                if strategy == "mean":
                    fill_value = df_cleaned[col].mean()
                    df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                    logger.info(
                        f"Dataset {dataset_id}: Filled '{col}' with mean={fill_value}"
                    )
                elif strategy == "median":
                    fill_value = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                    logger.info(
                        f"Dataset {dataset_id}: Filled '{col}' with median={fill_value}"
                    )
                elif strategy == "mode":
                    mode_val = df_cleaned[col].mode()
                    if not mode_val.empty:
                        fill_value = mode_val[0]
                        df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                        logger.info(
                            f"Dataset {dataset_id}: Filled '{col}' with mode={fill_value}"
                        )
                elif strategy == "zero":
                    df_cleaned[col] = df_cleaned[col].fillna(0)
                    logger.info(f"Dataset {dataset_id}: Filled '{col}' with zero")
            except Exception as e:
                alerts.append(f"Failed to apply {strategy} to column '{col}': {str(e)}")
                logger.error(
                    f"Cleaning failed for column '{col}' in dataset {dataset_id}: {str(e)}"
                )

    encoding = operations.get("encoding")
    categorical_mappings = {}
    if encoding in {"onehot", "label"}:
        logger.info(f"Dataset {dataset_id}: Applying encoding={encoding}")
        categorical_cols = df_cleaned.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if not categorical_cols:
            alerts.append("No categorical columns found for encoding")
            logger.info(f"Dataset {dataset_id}: No categorical columns for encoding")
        else:
            try:
                if encoding == "label":
                    for col in categorical_cols:
                        le = LabelEncoder()
                        df_cleaned[col] = le.fit_transform(df_cleaned[col].fillna(""))
                        # Convert np.int64 to int for JSON serialization
                        categorical_mappings[col] = {
                            str(k): int(v)
                            for k, v in zip(le.classes_, le.transform(le.classes_))
                        }
                        logger.info(
                            f"Dataset {dataset_id}: Label encoded column '{col}', mapping={categorical_mappings}"
                        )
                elif encoding == "onehot":
                    df_cleaned = pd.get_dummies(
                        df_cleaned, columns=categorical_cols, dummy_na=False
                    )
                    logger.info(
                        f"Dataset {dataset_id}: One-hot encoded columns {categorical_cols}"
                    )
            except Exception as e:
                alerts.append(f"Failed to apply {encoding} encoding: {str(e)}")
                logger.error(f"Encoding failed for dataset {dataset_id}: {str(e)}")
        logger.info(
            f"Dataset {dataset_id} columns after encoding: {df_cleaned.columns.tolist()}"
        )

    if operations.get("dropna", False):
        logger.info(f"Dataset {dataset_id}: Applying dropna")
        row_count_before = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        logger.info(
            f"Dataset {dataset_id}: Dropped {row_count_before - len(df_cleaned)} rows"
        )

    if operations.get("remove_duplicates", False):
        logger.info(f"Dataset {dataset_id}: Applying remove_duplicates")
        row_count_before = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        logger.info(
            f"Dataset {dataset_id}: Removed {row_count_before - len(df_cleaned)} duplicates"
        )

    scale = operations.get("scale")
    normalization_params = {}
    if scale in {"normalize", "standardize"}:
        logger.info(f"Dataset {dataset_id}: Applying scale={scale}")
        numeric_cols = df_cleaned.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        if not numeric_cols:
            alerts.append("No numeric columns found for scaling")
            logger.info(f"Dataset {dataset_id}: No numeric columns for scaling")
        else:
            try:
                for col in numeric_cols:
                    if scale == "normalize":
                        min_val, max_val = df_cleaned[col].min(), df_cleaned[col].max()
                        if max_val != min_val:
                            df_cleaned[col] = (df_cleaned[col] - min_val) / (
                                max_val - min_val
                            )
                            normalization_params[col] = {
                                "min": float(min_val),
                                "max": float(max_val),
                            }
                            logger.info(
                                f"Dataset {dataset_id}: Normalized column '{col}', min={min_val}, max={max_val}"
                            )
                    elif scale == "standardize":
                        mean_val, std_val = (
                            df_cleaned[col].mean(),
                            df_cleaned[col].std(),
                        )
                        if std_val != 0:
                            df_cleaned[col] = (df_cleaned[col] - mean_val) / std_val
                            normalization_params[col] = {
                                "mean": float(mean_val),
                                "std": float(std_val),
                            }
                            logger.info(
                                f"Dataset {dataset_id}: Standardized column '{col}', mean={mean_val}, std={std_val}"
                            )
            except Exception as e:
                alerts.append(f"Failed to apply {scale} scaling: {str(e)}")
                logger.error(f"Scaling failed for dataset {dataset_id}: {str(e)}")

    # After stats
    after = {
        "shape": list(df_cleaned.shape),
        "null_counts": {
            col: sanitize_value(val)
            for col, val in df_cleaned.isnull().sum().to_dict().items()
        },
        "dtypes": {col: str(dtype) for col, dtype in df_cleaned.dtypes.items()},
    }
    logger.info(f"Dataset {dataset_id} after_stats: {after}")

    # Preview data
    preview = [
        {k: sanitize_value(v) for k, v in record.items()}
        for record in df_cleaned.head(5).to_dict(orient="records")
    ]
    logger.info(f"Dataset {dataset_id} preview: {preview}")

    # Save to S3 if save=True
    if save:
        try:
            buf = io.StringIO()
            df_cleaned.to_csv(buf, index=False)
            s3_key_cleaned = f"cleaned/final_{dataset.filename}"
            response = s3.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key_cleaned,
                Body=buf.getvalue().encode("utf-8"),
            )
            logger.info(
                f"Dataset {dataset_id} uploaded to S3: {s3_key_cleaned}, response: {response}"
            )
            # Verify upload
            try:
                s3.head_object(Bucket=S3_BUCKET, Key=s3_key_cleaned)
                logger.info(f"Dataset {dataset_id} verified in S3: {s3_key_cleaned}")
                dataset.s3_key_cleaned = s3_key_cleaned
                dataset.has_cleaned_data = True
            except ClientError as e:
                logger.error(
                    f"Failed to verify S3 upload for {s3_key_cleaned}: {str(e)}"
                )
                raise Exception(f"Failed to verify S3 upload: {str(e)}")
            dataset.column_metadata = {
                col: {
                    "dtype": str(df_cleaned[col].dtype),
                    "n_unique": int(df_cleaned[col].nunique(dropna=True)),
                    "null_count": int(df_cleaned[col].isnull().sum()),
                }
                for col in df_cleaned.columns
            }
            dataset.n_rows, dataset.n_columns = df_cleaned.shape
            dataset.has_missing_values = bool(df_cleaned.isnull().values.any())
            dataset.categorical_mappings = (
                categorical_mappings
                if categorical_mappings
                else dataset.categorical_mappings
            )
            dataset.normalization_params = (
                normalization_params
                if normalization_params
                else dataset.normalization_params
            )
            dataset.processing_log = dataset.processing_log or ""
            dataset.processing_log += (
                f" | Cleaned with operations: {json.dumps(operations)}"
            )
            await db.commit()
            await db.refresh(dataset)
            logger.info(
                f"Dataset {dataset_id} saved: s3_key_cleaned={s3_key_cleaned}, has_cleaned_data={dataset.has_cleaned_data}"
            )
        except Exception as e:
            await db.rollback()
            logger.error(f"Save failed for dataset {dataset_id}: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to save cleaned data: {str(e)}"
            )

    # Response
    response = {
        "before_stats": before,
        "after_stats": after,
        "preview": preview,
        "alerts": alerts,
        "saved": save,
    }

    # Log response
    logger.info(f"Dataset {dataset_id} response: {json.dumps(response, default=str)}")

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
