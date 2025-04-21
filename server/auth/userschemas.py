from fastapi_users import schemas
from pydantic import EmailStr

class UserRead(schemas.BaseUser[int]):
    pass

class UserCreate(schemas.BaseUserCreate):
    email: EmailStr
    password: str  # `BaseUserCreate` already includes this, but you can redefine it if needed

class UserUpdate(schemas.BaseUserUpdate):
    pass


