from sqlalchemy import Boolean, Column, Integer, String
from server.database import Base  # ✅ Use shared Base
from fastapi_users.db import SQLAlchemyBaseUserTable

class User(SQLAlchemyBaseUserTable[int], Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)          # ✅ Add length
    hashed_password = Column(String(255), nullable=False)                         # ✅ Add length
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)


