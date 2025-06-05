# server/auth/authbackend.py

import os
from dotenv import load_dotenv
from fastapi_users.authentication import (
    AuthenticationBackend,
    JWTStrategy,
    CookieTransport,
)

# 1) Load environment variables from server/.env
load_dotenv()

# 2) Retrieve and validate JWT_SECRET
SECRET = os.getenv("JWT_SECRET", "")
if not SECRET:
    raise RuntimeError("JWT_SECRET is not set in server/.env")

# 3) Retrieve SECURE_COOKIE from .env (as a string "True"/"False") and convert to boolean
_raw_secure = os.getenv("SECURE_COOKIE", "False")
SECURE_COOKIE = _raw_secure.strip().lower() in ("true", "1", "yes")

# 4) Configure CookieTransport using SECURE_COOKIE
cookie_transport = CookieTransport(
    cookie_name="access-token",
    cookie_max_age=3600,
    cookie_secure=SECURE_COOKIE,  # False locally; set True in production
    cookie_httponly=True,
)


# 5) JWT strategy factory
def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)


# 6) Assemble the AuthenticationBackend
auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)
