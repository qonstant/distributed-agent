from datetime import date
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import get_current_admin
from ..crud import (
    classify_message_by_admin,
    extend_user_access,
    get_all_admin_actions,
    get_message_by_id,
    get_user_by_telegram_id,
    get_user_by_username,
    get_user_messages,
    get_user_usage_events,
    search_users_by_username,
    set_user_access,
    set_user_blocked,
    verify_user_payment,
)
from ..db import get_db
from ..schemas import (
    AdminActionOut,
    ClassifierIntent,
    MessageClassificationOut,
    MessageOut,
    UsageEventOut,
    UserOut,
)

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(get_current_admin)])


@router.get("/users/search", response_model=List[UserOut])
async def search_users(q: str, db: AsyncSession = Depends(get_db)):
    users = await search_users_by_username(db, q)
    return users[:50]


@router.post("/users/{telegram_id}/grant", response_model=UserOut)
async def grant_access(
    telegram_id: int,
    notes: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin),
):
    admin_user = await get_user_by_username(db, admin["username"])
    if not admin_user:
        raise HTTPException(status_code=401, detail="Admin not found")

    user = await set_user_access(
        db,
        telegram_id,
        True,
        admin_user_id=admin_user.id,
        notes=notes,
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@router.post("/users/{telegram_id}/revoke", response_model=UserOut)
async def revoke_access(
    telegram_id: int,
    notes: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin),
):
    admin_user = await get_user_by_username(db, admin["username"])
    if not admin_user:
        raise HTTPException(status_code=401, detail="Admin not found")

    user = await set_user_access(
        db,
        telegram_id,
        False,
        admin_user_id=admin_user.id,
        notes=notes,
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@router.post("/users/{telegram_id}/block", response_model=UserOut)
async def block_user(
    telegram_id: int,
    notes: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin),
):
    admin_user = await get_user_by_username(db, admin["username"])
    if not admin_user:
        raise HTTPException(status_code=401, detail="Admin not found")

    user = await set_user_blocked(
        db,
        telegram_id,
        True,
        admin_user_id=admin_user.id,
        notes=notes,
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@router.post("/users/{telegram_id}/unblock", response_model=UserOut)
async def unblock_user(
    telegram_id: int,
    notes: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin),
):
    admin_user = await get_user_by_username(db, admin["username"])
    if not admin_user:
        raise HTTPException(status_code=401, detail="Admin not found")

    user = await set_user_blocked(
        db,
        telegram_id,
        False,
        admin_user_id=admin_user.id,
        notes=notes,
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@router.post("/users/{telegram_id}/verify-payment", response_model=UserOut)
async def verify_payment(
    telegram_id: int,
    notes: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin),
):
    admin_user = await get_user_by_username(db, admin["username"])
    if not admin_user:
        raise HTTPException(status_code=401, detail="Admin not found")

    user = await verify_user_payment(
        db,
        telegram_id,
        admin_user_id=admin_user.id,
        notes=notes,
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@router.post("/users/{telegram_id}/extend-access", response_model=UserOut)
async def extend_access(
    telegram_id: int,
    days: int = 30,
    notes: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin),
):
    admin_user = await get_user_by_username(db, admin["username"])
    if not admin_user:
        raise HTTPException(status_code=401, detail="Admin not found")

    user = await extend_user_access(
        db,
        telegram_id,
        admin_user_id=admin_user.id,
        days=days,
        notes=notes,
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@router.get("/actions", response_model=List[AdminActionOut])
async def get_admin_actions(
    admin_user_id: int | None = None,
    target_user_id: int | None = None,
    entity_type: str | None = None,
    entity_id: int | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    db: AsyncSession = Depends(get_db),
):
    return await get_all_admin_actions(
        db=db,
        admin_user_id=admin_user_id,
        target_user_id=target_user_id,
        entity_type=entity_type,
        entity_id=entity_id,
        date_from=date_from,
        date_to=date_to,
    )


@router.get("/usage-events", response_model=List[UsageEventOut])
async def get_admin_usage_events(
    user_id: int | None = None,
    conversation_id: int | None = None,
    message_id: int | None = None,
    event_type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    db: AsyncSession = Depends(get_db),
):
    return await get_user_usage_events(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
        message_id=message_id,
        event_type=event_type,
        date_from=date_from,
        date_to=date_to,
    )


@router.get("/users/{telegram_id}/messages", response_model=List[MessageOut])
async def get_admin_user_messages(
    telegram_id: int,
    db: AsyncSession = Depends(get_db),
):
    user = await get_user_by_telegram_id(db, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return await get_user_messages(db, user_id=user.id)


@router.get("/messages/{message_id}", response_model=MessageOut)
async def get_admin_message(
    message_id: int,
    db: AsyncSession = Depends(get_db),
):
    message = await get_message_by_id(db, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    return message


@router.post("/messages/{message_id}/classify", response_model=MessageClassificationOut)
async def classify_message(
    message_id: int,
    intent: ClassifierIntent,
    explanation: str | None = None,
    detected_language: str | None = None,
    classifier_model: str | None = None,
    classifier_version: str | None = None,
    notes: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(get_current_admin),
):
    admin_user = await get_user_by_username(db, admin["username"])
    if not admin_user:
        raise HTTPException(status_code=401, detail="Admin not found")

    classification = await classify_message_by_admin(
        db=db,
        message_id=message_id,
        admin_user_id=admin_user.id,
        intent=intent.value,
        explanation=explanation,
        detected_language=detected_language,
        classifier_model=classifier_model,
        classifier_version=classifier_version,
        notes=notes,
    )
    if not classification:
        raise HTTPException(status_code=404, detail="Message not found")

    return classification
