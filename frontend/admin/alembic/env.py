import asyncio
import os
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine
from alembic import context
from app.models import Base
from dotenv import load_dotenv

load_dotenv()

config = context.config
fileConfig(config.config_file_name)

ALLOW_SCHEMA_MIGRATIONS = os.getenv("ADMIN_ALLOW_SCHEMA_MIGRATIONS", "").strip().lower()
if ALLOW_SCHEMA_MIGRATIONS not in {"1", "true", "yes"}:
    raise RuntimeError(
        "Admin Alembic migrations are disabled. The admin panel uses the "
        "existing Go-managed database schema and must not create or alter "
        "tables. Run `python scripts/check_schema.py` to validate the schema. "
        "Set ADMIN_ALLOW_SCHEMA_MIGRATIONS=1 only for explicit legacy/local "
        "migration maintenance."
    )

DATABASE_URL = os.getenv("DATABASE_URL")
target_metadata = Base.metadata

def run_migrations_offline():
    context.configure(url=DATABASE_URL, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online():
    connectable = create_async_engine(DATABASE_URL, future=True, echo=True, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
