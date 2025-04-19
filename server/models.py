from sqlalchemy import Column, Integer, String, DateTime, JSON
from datetime import datetime
from database import Base

class Dataset(Base):
    __tablename__ = 'datasets'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)  # New
    description = Column(String, nullable=True)  # New
    filename = Column(String, nullable=False)
    cleaned_data = Column(JSON)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


