# server/auth/userroutes.py
from fastapi import APIRouter, Depends
from fastapi_users import FastAPIUsers

from .usermodels import get_user_manager
from .userschemas import UserRead, UserCreate, UserUpdate
from .authbackend import auth_backend
from .userbase import User

router = APIRouter()

fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

current_user = fastapi_users.current_user()

# ✅ Protected route
@router.get("/protected", response_model=UserRead)
async def protected_route(user: User = Depends(current_user)):
    return {"message": f"Hello {user.email}, you're authenticated!"}

# 👇 Auth and user management routes
router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"]
)

router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"]
)

router.include_router(
    fastapi_users.get_users_router(
        UserRead,
        UserUpdate,
        requires_verification=False,   # ← add this
    ),
    prefix="/users",
    tags=["users"],
)


