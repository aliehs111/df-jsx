# server/schemas.py
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime

class DatasetSummary(BaseModel):
    id: int
    title: str
    description: str
    filename: str
    uploaded_at: datetime
    s3_key: Optional[str] = None

    class Config:
       orm_mode = True  # Pydantic v1 syntax

class Dataset(BaseModel):
    id: int
    title: str
    description: str
    filename: str
    raw_data: Any  # ✅ <- allow any JSON-serializable structure
    s3_key: Optional[str] = None
    cleaned_data: Optional[Any] = None
    categorical_mappings: Optional[Dict[str, Any]] = None
    normalization_params: Optional[Dict[str, Any]] = None
    column_renames: Optional[Dict[str, Any]] = None
    uploaded_at: datetime

    class Config:
        orm_mode = True  # Pydantic v1 syntax