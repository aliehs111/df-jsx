# server/models.py

from sqlalchemy import Column, Integer, String, DateTime, JSON
from datetime import datetime
from database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    cleaned_data = Column(JSON)  # Can store cleaned data directly or a reference (e.g., S3 URL)

    def __repr__(self):
        return f"<Dataset(id={self.id}, filename='{self.filename}', uploaded_at={self.uploaded_at})>"

