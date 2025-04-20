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

    class Config:
        from_attributes = True  # <-- Replaces orm_mode in Pydantic v2


class Dataset(BaseModel):
    id: int
    title: str
    description: str
    filename: str
    raw_data: Any  # ✅ <- allow any JSON-serializable structure
    cleaned_data: Optional[Any] = None
    categorical_mappings: Optional[Dict[str, Any]] = None
    normalization_params: Optional[Dict[str, Any]] = None
    column_renames: Optional[Dict[str, Any]] = None
    uploaded_at: datetime

    class Config:
        from_attributes = True  # Pydantic v2