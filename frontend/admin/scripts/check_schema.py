import asyncio
import os
from typing import Iterable

from dotenv import load_dotenv
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine


load_dotenv()

DATABASE_URL = os.getenv("ADMIN_DATABASE_URL") or os.getenv("DATABASE_URL")

REQUIRED_COLUMNS = {
    "users": {
        "id",
        "telegram_id",
        "username",
        "first_name",
        "last_name",
        "is_blocked",
        "is_admin",
        "access_expires_at",
        "created_at",
        "updated_at",
    },
    "conversations": {
        "id",
        "user_id",
        "conversation_key",
        "summary",
        "created_at",
        "updated_at",
    },
    "messages": {
        "id",
        "conversation_id",
        "message_text",
        "created_at",
    },
    "message_classifications": {
        "id",
        "message_id",
        "intent",
        "explanation",
        "detected_language",
        "classifier_model",
        "classifier_version",
        "created_at",
    },
    "usage_events": {
        "id",
        "user_id",
        "conversation_id",
        "message_id",
        "event_type",
        "input_tokens",
        "output_tokens",
        "estimated_cost",
        "created_at",
    },
    "admin_actions": {
        "id",
        "admin_user_id",
        "target_user_id",
        "action_type",
        "entity_type",
        "entity_id",
        "notes",
        "created_at",
    },
}


def _format_missing(name: str, values: Iterable[str]) -> str:
    return f"{name}: {', '.join(sorted(values))}"


def inspect_schema(connection) -> list[str]:
    inspector = inspect(connection)
    existing_tables = set(inspector.get_table_names())
    errors = []

    missing_tables = set(REQUIRED_COLUMNS) - existing_tables
    if missing_tables:
        errors.append(_format_missing("missing tables", missing_tables))

    for table_name, required_columns in REQUIRED_COLUMNS.items():
        if table_name not in existing_tables:
            continue

        existing_columns = {
            column["name"] for column in inspector.get_columns(table_name)
        }
        missing_columns = required_columns - existing_columns
        if missing_columns:
            errors.append(
                _format_missing(f"missing columns in {table_name}", missing_columns)
            )

    return errors


async def main() -> int:
    if not DATABASE_URL:
        raise RuntimeError("ADMIN_DATABASE_URL or DATABASE_URL is not set")

    engine = create_async_engine(DATABASE_URL, future=True, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            errors = await connection.run_sync(inspect_schema)
    finally:
        await engine.dispose()

    if errors:
        print("[schema] admin schema check failed")
        for error in errors:
            print(f"[schema] {error}")
        print("[schema] Run the Go database migrations before deploying admin.")
        return 1

    print("[schema] admin schema check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
