# server/auth/userdatabase.py
from fastapi import Depends
from sqlalchemy.orm import Session  # <-- make sure it's NOT from sqlalchemy.ext.asyncio
from fastapi_users.db import SQLAlchemyUserDatabase

from server.database import get_async_db  # ✅ correct


from server.auth.userbase import User


async def get_user_db(session: Session = Depends(get_async_db)):
    yield SQLAlchemyUserDatabase(session, User)


