# server/models.py
from sqlalchemy import Column, Integer, String, DateTime, JSON
from datetime import datetime
from database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    cleaned_data = Column(JSON)  # or just store an S3 URL if you want
