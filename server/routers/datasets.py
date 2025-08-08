# server/routers/datasets.py

import io
import boto3
import logging
import base64

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer



from server.database import get_async_db
from server.models import Dataset as DatasetModel
from server.schemas import DatasetSummary
from server.auth.userroutes import current_user
from server.aws_client import get_s3, S3_BUCKET
from server.schemas import CleanRequest

logger = logging.getLogger("server.main")
router = APIRouter(prefix="/datasets", tags=["datasets"])

s3 = boto3.client("s3")
S3_BUCKET = "dfjsx-uploads"




@router.get(
    "/cleaned",
    response_model=list[DatasetSummary],
    dependencies=[Depends(current_user)],
)
async def list_cleaned_datasets(db: AsyncSession = Depends(get_async_db)):
    try:
        stmt = select(DatasetModel).filter(
            DatasetModel.has_cleaned_data == True, DatasetModel.s3_key_cleaned != None
        )
        result = await db.execute(stmt)
        datasets = result.scalars().all()
        logger.info(f"Fetched {len(datasets)} cleaned datasets")
        return datasets
    except Exception as e:
        logger.error(f"Failed to fetch cleaned datasets: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch cleaned datasets: {str(e)}"
        )


@router.get("/{dataset_id}/columns", dependencies=[Depends(current_user)])
async def get_dataset_columns(
    dataset_id: int, db: AsyncSession = Depends(get_async_db)
):
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.has_cleaned_data or not dataset.s3_key_cleaned:
        raise HTTPException(
            status_code=400,
            detail="No cleaned data available. Please clean the dataset in Data Cleaning.",
        )
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=dataset.s3_key_cleaned)
        content = response["Body"].read()
        encodings = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    io.StringIO(content.decode(encoding, errors="replace"))
                )
                logger.info(
                    f"Loaded CSV {dataset.s3_key_cleaned} with encoding {encoding}"
                )
                break
            except Exception as e:
                logger.warning(
                    f"Failed with encoding {encoding} for {dataset.s3_key_cleaned}: {str(e)}"
                )
        if df is None:
            raise Exception("Failed to parse CSV with any encoding")
        return {"columns": df.columns.tolist()}
    except Exception as e:
        logger.error(f"Failed to fetch columns for dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load cleaned data: {str(e)}. Please ensure the dataset is cleaned and saved in Data Cleaning.",
        )


@router.get(
    "/{dataset_id}/column/{column_name}/unique", dependencies=[Depends(current_user)]
)
async def get_column_unique_values(
    dataset_id: int, column_name: str, db: AsyncSession = Depends(get_async_db)
):
    result = await db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.has_cleaned_data or not dataset.s3_key_cleaned:
        raise HTTPException(
            status_code=400,
            detail="No cleaned data available. Please clean the dataset in Data Cleaning.",
        )
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=dataset.s3_key_cleaned)
        content = response["Body"].read()
        encodings = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    io.StringIO(content.decode(encoding, errors="replace"))
                )
                logger.info(
                    f"Loaded CSV {dataset.s3_key_cleaned} with encoding {encoding}"
                )
                break
            except Exception as e:
                logger.warning(
                    f"Failed with encoding {encoding} for {dataset.s3_key_cleaned}: {str(e)}"
                )
        if df is None:
            raise Exception("Failed to parse CSV with any encoding")
        if column_name not in df.columns:
            raise HTTPException(
                status_code=400, detail=f"Column '{column_name}' not found"
            )
        unique_count = df[column_name].dropna().nunique()
        return {"unique_count": unique_count}
    except Exception as e:
        logger.error(
            f"Failed to fetch unique values for dataset {dataset_id}, column {column_name}: {str(e)}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load column data: {str(e)}. Please ensure the dataset is cleaned in Data Cleaning.",
        )



@router.post("/{dataset_id}/clean-preview")
async def clean_preview(dataset_id: int, request: Request, db: AsyncSession = Depends(get_async_db), user=Depends(current_user)):
    req = await request.json()
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
            "column_metadata": {
                col: {
                    "dtype": str(dtype),
                    "n_unique": int(df[col].nunique()),
                    "null_count": int(df[col].isnull().sum())
                }
                for col, dtype in df.dtypes.items()
            }
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
            return None if (np.isnan(val) or np.isinf(val)) else float(val)
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
