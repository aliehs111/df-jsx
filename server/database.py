import os

# ──────────── only load .env when running locally ────────────
if os.environ.get("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv()

# ──────────── now import SQLAlchemy pieces ────────────
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# ──────────── pick up your Heroku URL (or local .env) ────────────
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

# ──────────── swap in the async driver if it wasn’t already ────────────
if DATABASE_URL.startswith("mysql+pymysql://"):
    # only replace first occurrence
    DATABASE_URL = DATABASE_URL.replace(
        "mysql+pymysql://", 
        "mysql+aiomysql://", 
        1
    )

# ──────────── create your async engine & session factory ────────────
engine = create_async_engine(DATABASE_URL, echo=True)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ──────────── your declarative base ────────────
Base = declarative_base()

# ──────────── FastAPI dependency ────────────
async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session
