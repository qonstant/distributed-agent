import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User

SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", os.getenv("INITIAL_ADMIN_USERNAME", "Unibothelper"))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", os.getenv("INITIAL_ADMIN_PASSWORD", "unibot123456"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/token")


async def authenticate_admin(db: AsyncSession, username: str, password: str):
    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        return None

    result = await db.execute(
        select(User).where(User.username == username)
    )
    user = result.scalars().first()

    if not user:
        return None

    if not user.is_admin:
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
