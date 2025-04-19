# server/schemas.py
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
    raw_data: dict | None
    cleaned_data: dict | None
    categorical_mappings: dict | None
    normalization_params: dict | None
    column_renames: dict | None
    uploaded_at: datetime

    class Config:
        from_attributes = True  # replaces orm_mode in Pydantic v2
