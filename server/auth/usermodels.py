from fastapi_users import BaseUserManager, IntegerIDMixin
from fastapi import Depends
from typing import AsyncGenerator

from server.auth.userdatabase import get_user_db
from server.auth.userbase import User

class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    user_db_model = User

    async def on_after_register(self, user: User, request=None):
        print(f"User {user.email} has registered.")

async def get_user_manager(user_db=Depends(get_user_db)) -> AsyncGenerator[UserManager, None]:
    yield UserManager(user_db)

