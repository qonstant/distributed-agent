import asyncio
import os
import random
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


async def generate_fake_telegram_id(session) -> int:
    while True:
        fake_id = random.randint(10**9, 10**10 - 1)
        result = await session.execute(
            select(User).where(User.telegram_id == fake_id)
        )
        if not result.scalars().first():
            return fake_id


async def create_admin():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.username == INITIAL_ADMIN_USERNAME)
        )
        existing = result.scalars().first()

        if existing:
            if not existing.is_admin:
                existing.is_admin = True
                await session.commit()
                print(f"Existing user '{INITIAL_ADMIN_USERNAME}' promoted to admin")
                return

            print(f"Admin '{INITIAL_ADMIN_USERNAME}' already exists")
            return

        telegram_id = await generate_fake_telegram_id(session)

        admin_user = User(
            username=INITIAL_ADMIN_USERNAME,
            telegram_id=telegram_id,
            is_admin=True,
            is_blocked=False,
        )

        session.add(admin_user)

        try:
            await session.commit()
            print("Admin created successfully")
            print(f"Username: {INITIAL_ADMIN_USERNAME}")
            print(f"Telegram ID (fake): {telegram_id}")
        except IntegrityError as exc:
            await session.rollback()
            print("Error creating admin:", exc)


if __name__ == "__main__":
    asyncio.run(create_admin())
