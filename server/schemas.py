# server/schemas.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class DatasetCreate(BaseModel):
    title: str
    description: Optional[str] = None
    filename: str
    s3_key: Optional[str] = None
    s3_key_cleaned: Optional[str] = None
    categorical_mappings: Optional[Dict[str, Any]] = None
    normalization_params: Optional[Dict[str, Any]] = None
    column_renames: Optional[Dict[str, Any]] = None
    target_column: Optional[str] = None
    selected_features: Optional[List[str]] = None
    excluded_columns: Optional[List[str]] = None
    feature_engineering_notes: Optional[str] = None
    column_metadata: Optional[Dict[str, Any]] = None
    n_rows: Optional[int] = None
    n_columns: Optional[int] = None
    has_missing_values: Optional[bool] = None
    processing_log: Optional[str] = None  # Changed to string
    current_stage: Optional[str] = None
    has_cleaned_data: bool = False
    extra_json_1: Optional[Dict[str, Any]] = None
    extra_txt_1: Optional[str] = None


class DatasetSummary(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    filename: str
    s3_key: Optional[str] = None
    s3_key_cleaned: Optional[str] = None
    uploaded_at: datetime
    has_cleaned_data: bool

    class Config:
        orm_mode = True


class DatasetOut(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    filename: str
    s3_key: Optional[str] = None
    s3_key_cleaned: Optional[str] = None
    uploaded_at: datetime
    categorical_mappings: Optional[Dict[str, Any]] = None
    normalization_params: Optional[Dict[str, Any]] = None
    column_renames: Optional[Dict[str, Any]] = None
    target_column: Optional[str] = None
    selected_features: Optional[List[str]] = None
    excluded_columns: Optional[List[str]] = None
    feature_engineering_notes: Optional[str] = None
    column_metadata: Optional[Dict[str, Any]] = None
    n_rows: Optional[int] = None
    n_columns: Optional[int] = None
    has_missing_values: Optional[bool] = None
    processing_log: Optional[str] = None  # Changed to string
    current_stage: Optional[str] = None
    has_cleaned_data: bool
    extra_json_1: Optional[Dict[str, Any]] = None
    extra_txt_1: Optional[str] = None
    preview_data: Optional[List[Dict]] = None

    class Config:
        orm_mode = True


class CleanOps(BaseModel):
    dropna: Optional[bool] = False
    fillna: Optional[Dict[str, Any]] = {}
    lowercase_headers: Optional[bool] = False
    remove_duplicates: Optional[bool] = False
    selected_columns: Optional[List[str]] = None


class PreprocessOps(BaseModel):
    scale: Optional[str] = None
    encoding: Optional[str] = None
    selected_columns: Optional[List[str]] = None


class ProcessRequest(BaseModel):
    clean: CleanOps
    preprocess: PreprocessOps


class DownloadURLResponse(BaseModel):
    url: str


class CleanPreviewRequest(BaseModel):
    dataset_id: int
    operations: Dict[str, Any]
    save: Optional[bool] = False
