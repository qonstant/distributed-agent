from datetime import date, datetime, time, timedelta, timezone
from typing import List, Optional

from sqlalchemy import String, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import (
    AdminAction,
    Conversation,
    Message,
    MessageClassification,
    UsageEvent,
    User,
)


async def get_user_by_telegram_id(db: AsyncSession, telegram_id: int) -> Optional[User]:
    result = await db.execute(select(User).where(User.telegram_id == telegram_id))
    return result.scalars().first()


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.username == username))
    return result.scalars().first()


async def get_all_users(db: AsyncSession) -> List[User]:
    result = await db.execute(select(User).order_by(User.id.desc()))
    return result.scalars().all()


async def search_users_by_username(db: AsyncSession, username_substr: str) -> List[User]:
    if not username_substr:
        return []

    pattern = f"%{username_substr}%"
    result = await db.execute(
        select(User)
        .where(
            or_(
                User.username.ilike(pattern),
                cast(User.telegram_id, String).ilike(pattern),
            )
        )
        .order_by(User.id.desc())
    )
    return result.scalars().all()


async def create_or_update_user(
    db: AsyncSession,
    telegram_id: int,
    username: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    is_blocked: bool | None = None,
    is_admin: bool | None = None,
    access_expires_at=None,
    admin_user_id: int | None = None,
):
    user = await get_user_by_telegram_id(db, telegram_id)

    if not user:
        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            is_blocked=is_blocked if is_blocked is not None else False,
            is_admin=is_admin if is_admin is not None else False,
            access_expires_at=access_expires_at,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

        if admin_user_id is not None:
            await create_admin_action(
                db=db,
                admin_user_id=admin_user_id,
                target_user_id=user.id,
                action_type="other",
                entity_type="user",
                entity_id=user.id,
                notes="create_user",
            )

        return user

    previous_has_access = user.has_access
    changed = False
    blocked_changed = False
    access_changed = False

    if username is not None and username != user.username:
        user.username = username
        changed = True

    if first_name is not None and first_name != user.first_name:
        user.first_name = first_name
        changed = True

    if last_name is not None and last_name != user.last_name:
        user.last_name = last_name
        changed = True

    if is_blocked is not None and is_blocked != user.is_blocked:
        user.is_blocked = is_blocked
        changed = True
        blocked_changed = True

    if is_admin is not None and is_admin != user.is_admin:
        user.is_admin = is_admin
        changed = True

    if access_expires_at is not None and access_expires_at != user.access_expires_at:
        user.access_expires_at = access_expires_at
        changed = True
        access_changed = True

    if changed:
        await db.commit()
        await db.refresh(user)

        if admin_user_id is not None:
            if blocked_changed:
                await create_admin_action(
                    db=db,
                    admin_user_id=admin_user_id,
                    target_user_id=user.id,
                    action_type="block_user" if user.is_blocked else "unblock_user",
                    entity_type="user",
                    entity_id=user.id,
                )

            if access_changed and previous_has_access != user.has_access:
                await create_admin_action(
                    db=db,
                    admin_user_id=admin_user_id,
                    target_user_id=user.id,
                    action_type="grant_access" if user.has_access else "revoke_access",
                    entity_type="user",
                    entity_id=user.id,
                )

    return user


async def set_user_access(
    db: AsyncSession,
    telegram_id: int,
    has_access: bool,
    admin_user_id: int | None = None,
    access_expires_at=None,
    notes: str | None = None,
):
    user = await get_user_by_telegram_id(db, telegram_id)
    if not user:
        return None

    new_access_expires_at = access_expires_at
    if has_access and access_expires_at is None:
        new_access_expires_at = None
    if not has_access and access_expires_at is None:
        new_access_expires_at = datetime.now(timezone.utc)

    previous_has_access = user.has_access
    if user.access_expires_at != new_access_expires_at or previous_has_access != has_access:
        user.access_expires_at = new_access_expires_at
        await db.commit()
        await db.refresh(user)

        if admin_user_id is not None and previous_has_access != user.has_access:
            await create_admin_action(
                db=db,
                admin_user_id=admin_user_id,
                target_user_id=user.id,
                action_type="grant_access" if has_access else "revoke_access",
                entity_type="user",
                entity_id=user.id,
                notes=notes,
            )

    return user


async def set_user_blocked(
    db: AsyncSession,
    telegram_id: int,
    blocked: bool,
    admin_user_id: int | None = None,
    notes: str | None = None,
):
    user = await get_user_by_telegram_id(db, telegram_id)
    if not user:
        return None

    if user.is_blocked != blocked:
        user.is_blocked = blocked
        await db.commit()
        await db.refresh(user)

        if admin_user_id is not None:
            await create_admin_action(
                db=db,
                admin_user_id=admin_user_id,
                target_user_id=user.id,
                action_type="block_user" if blocked else "unblock_user",
                entity_type="user",
                entity_id=user.id,
                notes=notes,
            )

    return user


async def verify_user_payment(
    db: AsyncSession,
    telegram_id: int,
    admin_user_id: int,
    notes: str | None = None,
):
    user = await get_user_by_telegram_id(db, telegram_id)
    if not user:
        return None

    await create_admin_action(
        db=db,
        admin_user_id=admin_user_id,
        target_user_id=user.id,
        action_type="verify_payment",
        entity_type="user",
        entity_id=user.id,
        notes=notes,
    )
    return user


async def extend_user_access(
    db: AsyncSession,
    telegram_id: int,
    admin_user_id: int,
    days: int = 30,
    notes: str | None = None,
):
    user = await get_user_by_telegram_id(db, telegram_id)
    if not user:
        return None

    normalized_days = max(days, 1)
    now = datetime.now(timezone.utc)
    expires_at = user.access_expires_at
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    base_time = expires_at if expires_at and expires_at > now else now
    user.access_expires_at = base_time + timedelta(days=normalized_days)
    await db.commit()
    await db.refresh(user)

    await create_admin_action(
        db=db,
        admin_user_id=admin_user_id,
        target_user_id=user.id,
        action_type="extend_access",
        entity_type="user",
        entity_id=user.id,
        notes=notes or f"Extended by {normalized_days} days until {user.access_expires_at.isoformat()}",
    )
    return user


async def get_conversation_by_key(db: AsyncSession, conversation_key: str) -> Optional[Conversation]:
    result = await db.execute(
        select(Conversation).where(Conversation.conversation_key == conversation_key)
    )
    return result.scalars().first()


async def get_or_create_conversation(
    db: AsyncSession,
    user_id: int,
    conversation_key: str,
    summary: str | None = None,
):
    conversation = await get_conversation_by_key(db, conversation_key)
    if conversation:
        if summary is not None and summary != conversation.summary:
            conversation.summary = summary
            await db.commit()
            await db.refresh(conversation)
        return conversation

    conversation = Conversation(
        user_id=user_id,
        conversation_key=conversation_key,
        summary=summary,
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def get_user_conversations(db: AsyncSession, user_id: int):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
    )
    return result.scalars().all()


async def get_user_profile(db: AsyncSession, telegram_id: int) -> Optional[User]:
    result = await db.execute(
        select(User)
        .options(
            selectinload(User.conversations)
            .selectinload(Conversation.messages)
            .selectinload(Message.classification),
            selectinload(User.conversations).selectinload(Conversation.usage_events),
            selectinload(User.usage_events),
            selectinload(User.admin_actions_as_admin),
            selectinload(User.admin_actions_as_target),
        )
        .where(User.telegram_id == telegram_id)
    )
    return result.scalars().first()


async def create_usage_event(
    db: AsyncSession,
    user_id: int,
    event_type,
    conversation_id: int | None = None,
    message_id: int | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    estimated_cost: float = 0.0,
):
    usage_event = UsageEvent(
        user_id=user_id,
        conversation_id=conversation_id,
        message_id=message_id,
        event_type=event_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=estimated_cost,
    )
    db.add(usage_event)
    await db.commit()
    await db.refresh(usage_event)
    return usage_event


async def get_user_usage_events(
    db: AsyncSession,
    user_id: int | None = None,
    conversation_id: int | None = None,
    message_id: int | None = None,
    event_type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
):
    stmt = select(UsageEvent)

    if user_id is not None:
        stmt = stmt.where(UsageEvent.user_id == user_id)

    if conversation_id is not None:
        stmt = stmt.where(UsageEvent.conversation_id == conversation_id)

    if message_id is not None:
        stmt = stmt.where(UsageEvent.message_id == message_id)

    if event_type:
        stmt = stmt.where(UsageEvent.event_type == event_type)

    if date_from:
        stmt = stmt.where(UsageEvent.created_at >= datetime.combine(date_from, time.min))

    if date_to:
        stmt = stmt.where(UsageEvent.created_at <= datetime.combine(date_to, time.max))

    stmt = stmt.order_by(UsageEvent.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()


async def create_message(
    db: AsyncSession,
    conversation_id: int,
    message_text: str = "",
):
    message = Message(
        conversation_id=conversation_id,
        message_text=message_text,
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def get_user_messages(
    db: AsyncSession,
    user_id: int | None = None,
    conversation_id: int | None = None,
):
    stmt = (
        select(Message)
        .options(selectinload(Message.classification))
        .join(Message.conversation)
    )

    if conversation_id is not None:
        stmt = stmt.where(Message.conversation_id == conversation_id)

    if user_id is not None:
        stmt = stmt.where(Conversation.user_id == user_id)

    stmt = stmt.order_by(Message.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_message_by_id(db: AsyncSession, message_id: int):
    result = await db.execute(
        select(Message)
        .options(
            selectinload(Message.classification),
            selectinload(Message.conversation).selectinload(Conversation.user),
        )
        .where(Message.id == message_id)
    )
    return result.scalars().first()


async def create_message_classification(
    db: AsyncSession,
    message_id: int,
    intent,
    explanation: str | None = None,
    detected_language: str | None = None,
    classifier_model: str | None = None,
    classifier_version: str | None = None,
):
    existing = await get_message_classification(db, message_id)
    if existing:
        existing.intent = intent
        existing.explanation = explanation
        existing.detected_language = detected_language
        existing.classifier_model = classifier_model
        existing.classifier_version = classifier_version
        await db.commit()
        await db.refresh(existing)
        return existing

    classification = MessageClassification(
        message_id=message_id,
        intent=intent,
        explanation=explanation,
        detected_language=detected_language,
        classifier_model=classifier_model,
        classifier_version=classifier_version,
    )
    db.add(classification)
    await db.commit()
    await db.refresh(classification)
    return classification


async def get_message_classification(db: AsyncSession, message_id: int):
    result = await db.execute(
        select(MessageClassification).where(MessageClassification.message_id == message_id)
    )
    return result.scalars().first()


async def classify_message_by_admin(
    db: AsyncSession,
    message_id: int,
    admin_user_id: int,
    intent,
    explanation: str | None = None,
    detected_language: str | None = None,
    classifier_model: str | None = None,
    classifier_version: str | None = None,
    notes: str | None = None,
):
    message = await get_message_by_id(db, message_id)
    if not message:
        return None

    classification = await create_message_classification(
        db=db,
        message_id=message_id,
        intent=intent,
        explanation=explanation,
        detected_language=detected_language,
        classifier_model=classifier_model,
        classifier_version=classifier_version,
    )

    await create_admin_action(
        db=db,
        admin_user_id=admin_user_id,
        target_user_id=message.conversation.user_id,
        action_type="classify_message",
        entity_type="message",
        entity_id=message.id,
        notes=notes,
    )
    return classification


async def create_admin_action(
    db: AsyncSession,
    admin_user_id: int,
    target_user_id: int | None = None,
    action_type: str = "other",
    entity_type: str | None = None,
    entity_id: int | None = None,
    notes: str | None = None,
):
    action = AdminAction(
        admin_user_id=admin_user_id,
        target_user_id=target_user_id,
        action_type=action_type,
        entity_type=entity_type,
        entity_id=entity_id,
        notes=notes,
    )
    db.add(action)
    await db.commit()
    await db.refresh(action)
    return action


async def get_user_admin_actions(db: AsyncSession, user_id: int):
    result = await db.execute(
        select(AdminAction)
        .where(
            or_(
                AdminAction.admin_user_id == user_id,
                AdminAction.target_user_id == user_id,
            )
        )
        .order_by(AdminAction.created_at.desc())
    )
    return result.scalars().all()


async def get_all_admin_actions(
    db: AsyncSession,
    admin_user_id: int | None = None,
    target_user_id: int | None = None,
    entity_type: str | None = None,
    entity_id: int | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
):
    stmt = select(AdminAction)

    if admin_user_id is not None:
        stmt = stmt.where(AdminAction.admin_user_id == admin_user_id)

    if target_user_id is not None:
        stmt = stmt.where(AdminAction.target_user_id == target_user_id)

    if entity_type:
        stmt = stmt.where(AdminAction.entity_type == entity_type)

    if entity_id is not None:
        stmt = stmt.where(AdminAction.entity_id == entity_id)

    if date_from:
        stmt = stmt.where(AdminAction.created_at >= datetime.combine(date_from, time.min))

    if date_to:
        stmt = stmt.where(AdminAction.created_at <= datetime.combine(date_to, time.max))

    stmt = stmt.order_by(AdminAction.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()
