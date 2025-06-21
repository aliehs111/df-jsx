from sqlalchemy import Column, Integer, String, Text, JSON, TIMESTAMP, Boolean, func
from datetime import datetime
from server.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    filename = Column(String(255), nullable=False)
    s3_key = Column(String(512), nullable=True)
    s3_key_cleaned = Column(String(512), nullable=True)
    uploaded_at = Column(TIMESTAMP, server_default=func.now())
    categorical_mappings = Column(JSON, nullable=True)
    normalization_params = Column(JSON, nullable=True)
    column_renames = Column(JSON, nullable=True)
    target_column = Column(String(255), nullable=True)
    selected_features = Column(JSON, nullable=True)
    excluded_columns = Column(JSON, nullable=True)
    feature_engineering_notes = Column(Text, nullable=True)
    column_metadata = Column(JSON, nullable=True)
    n_rows = Column(Integer, nullable=True)
    n_columns = Column(Integer, nullable=True)
    has_missing_values = Column(Boolean, nullable=True)
    processing_log = Column(Text, nullable=True)
    current_stage = Column(String(255), nullable=True)
    has_cleaned_data = Column(Boolean, nullable=False, default=False)
    extra_json_1 = Column(JSON, nullable=True)
    extra_txt_1 = Column(Text, nullable=True)
