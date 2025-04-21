# server/database.py

import os
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# Load from .env
load_dotenv()

# Replace sync URL with async-compatible driver (aiomysql)
DATABASE_URL = os.getenv("DATABASE_URL").replace("mysql+pymysql://", "mysql+aiomysql://")

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Async session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Declare base
Base = declarative_base()

# Dependency for FastAPI to use in routes
async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session
