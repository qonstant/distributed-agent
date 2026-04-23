import asyncio
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import AsyncSessionLocal
from app.models import User

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV in {"prod", "production"}


def required_admin_username() -> str:
    username = os.getenv("ADMIN_USERNAME") or os.getenv("INITIAL_ADMIN_USERNAME")
    username = (username or "").strip()
    if username:
        return username
    if IS_PRODUCTION:
        raise RuntimeError("ADMIN_USERNAME is required when APP_ENV=production")
    return "Unibothelper"


INITIAL_ADMIN_USERNAME = required_admin_username()


def required_admin_telegram_id() -> int:
    raw = os.getenv("ADMIN_TELEGRAM_ID") or os.getenv("INITIAL_ADMIN_TELEGRAM_ID")
    raw = (raw or "").strip()
    if not raw:
        if IS_PRODUCTION:
            raise RuntimeError("ADMIN_TELEGRAM_ID is required when APP_ENV=production")
        return -1

    try:
        telegram_id = int(raw)
    except ValueError as exc:
        raise RuntimeError("ADMIN_TELEGRAM_ID must be an integer") from exc

    if telegram_id == 0:
        raise RuntimeError("ADMIN_TELEGRAM_ID must not be 0")
    if IS_PRODUCTION and telegram_id < 0:
        raise RuntimeError("ADMIN_TELEGRAM_ID must be a real positive Telegram user ID")

    return telegram_id


ADMIN_TELEGRAM_ID = required_admin_telegram_id()


async def save_admin(session, user: User, action: str) -> None:
    if not user.username:
        user.username = INITIAL_ADMIN_USERNAME
    user.telegram_id = ADMIN_TELEGRAM_ID
    user.is_admin = True
    user.is_blocked = False

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise RuntimeError(
            "Failed to save admin user. Check that ADMIN_TELEGRAM_ID is not "
            "already used by another user."
        ) from exc

    print(action)
    print(f"Username: {INITIAL_ADMIN_USERNAME}")
    print(f"Telegram ID: {ADMIN_TELEGRAM_ID}")


async def create_admin():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.telegram_id == ADMIN_TELEGRAM_ID)
        )
        existing = result.scalars().first()

        if existing:
            await save_admin(session, existing, "Existing Telegram user promoted to admin")
            return

        result = await session.execute(
            select(User).where(
                User.username == INITIAL_ADMIN_USERNAME,
                User.is_admin.is_(True),
            )
        )
        existing_admin = result.scalars().first()
        if existing_admin:
            await save_admin(
                session,
                existing_admin,
                "Existing admin user updated with configured Telegram ID",
            )
            return

        admin_user = User(
            username=INITIAL_ADMIN_USERNAME,
            telegram_id=ADMIN_TELEGRAM_ID,
            is_admin=True,
            is_blocked=False,
        )

        session.add(admin_user)
        await save_admin(session, admin_user, "Admin created successfully")


if __name__ == "__main__":
    asyncio.run(create_admin())
