from sqlalchemy import Column, Integer, String, Text, JSON, TIMESTAMP, func
from datetime import datetime
from database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    description = Column(Text)
    filename = Column(String(255))
    raw_data = Column(JSON)
    cleaned_data = Column(JSON)
    categorical_mappings = Column(JSON)
    normalization_params = Column(JSON)
    column_renames = Column(JSON)
    uploaded_at = Column(TIMESTAMP, server_default=func.now())

