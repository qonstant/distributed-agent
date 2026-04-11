from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..crud import get_user_by_telegram_id

router = APIRouter(prefix="/public", tags=["public"])


@router.get("/access/{telegram_id}")
async def check_access(telegram_id: int, db: AsyncSession = Depends(get_db)):
    user = await get_user_by_telegram_id(db, telegram_id)
    return {
        "telegram_id": telegram_id,
        "exists": bool(user),
        "has_access": bool(user and user.has_access),
        "is_blocked": bool(user and user.is_blocked),
        "access_expires_at": user.access_expires_at if user else None,
    }
