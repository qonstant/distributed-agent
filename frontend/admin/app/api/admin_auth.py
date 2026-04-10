from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import authenticate_admin, create_access_token
from ..db import get_db

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    admin = await authenticate_admin(db, form_data.username, form_data.password)

    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    access_token = create_access_token({
        "sub": admin.username,
        "is_admin": True
    })

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }
