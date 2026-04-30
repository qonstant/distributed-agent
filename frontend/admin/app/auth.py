import os
import hmac
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV in {"prod", "production"}


def _required_env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    if IS_PRODUCTION:
        raise RuntimeError(f"{name} is required when APP_ENV=production")
    return default


SECRET_KEY = _required_env("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ADMIN_USERNAME = _required_env(
    "ADMIN_USERNAME",
    os.getenv("INITIAL_ADMIN_USERNAME", "Unibothelper"),
)
ADMIN_PASSWORD = _required_env(
    "ADMIN_PASSWORD",
    os.getenv("INITIAL_ADMIN_PASSWORD", "unibot123456"),
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/token")


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


async def authenticate_admin(db: AsyncSession, username: str, password: str):
    if not _constant_time_equal(username, ADMIN_USERNAME):
        return None
    if not _constant_time_equal(password, ADMIN_PASSWORD):
        return None

    result = await db.execute(
        select(User).where(User.username == username, User.is_admin.is_(True))
    )
    user = result.scalars().first()

    if not user:
        return None

    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        is_admin = payload.get("is_admin", False)

        if username is None or not is_admin:
            raise credentials_exception

        return payload
    except JWTError:
        raise credentials_exception


async def get_current_admin(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    return {"username": payload["sub"], "is_admin": True}
