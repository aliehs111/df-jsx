from sqlalchemy import Column, Integer, String, Text, JSON, TIMESTAMP, func
from datetime import datetime
from server.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)            # PK index implicit
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    filename = Column(String(255), nullable=False)
    s3_key = Column(String(512), nullable=True)
    # keep JSON if on MySQL 5.7+/8, otherwise switch to Text
    raw_data = Column(JSON, nullable=True)
    cleaned_data = Column(JSON, nullable=True)
    categorical_mappings = Column(JSON, nullable=True)
    normalization_params = Column(JSON, nullable=True)
    column_renames = Column(JSON, nullable=True)

    uploaded_at = Column(TIMESTAMP, server_default=func.now())

