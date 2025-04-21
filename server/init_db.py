# server/init_db.py

import asyncio
from database import engine, Base
from auth.userbase import User  # make sure to import all models
from models import Dataset

async def ():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tables created!")

if __name__ == "__main__":
    asyncio.run(init_models())

