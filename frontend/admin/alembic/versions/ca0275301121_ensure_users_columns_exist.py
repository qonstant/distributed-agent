"""ensure users columns exist

Revision ID: ca0275301121
Revises: 09119a175e96
Create Date: 2026-03-01 15:37:40.419768

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ca0275301121'
down_revision: Union[str, Sequence[str], None] = '09119a175e96'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Добавляем колонки только если их нет (Postgres поддерживает IF NOT EXISTS).
    op.execute("""
        ALTER TABLE users ADD COLUMN IF NOT EXISTS telegram_user_id BIGINT;
    """)
    op.execute("""
        ALTER TABLE users ADD COLUMN IF NOT EXISTS telegram_username TEXT;
    """)
    op.execute("""
        ALTER TABLE users ADD COLUMN IF NOT EXISTS access_granted BOOLEAN DEFAULT false;
    """)
    op.execute("""
        ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT now();
    """)
    op.execute("""
        ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT now();
    """)
    # Добавим индексы/уникальности, если их нет
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_telegram_user_id ON users (telegram_user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_users_telegram_username ON users (telegram_username);")


def downgrade() -> None:
    # Откат — удаляем добавленные колонки (если нужно)
    op.execute("DROP INDEX IF EXISTS ix_users_telegram_user_id;")
    op.execute("DROP INDEX IF EXISTS ix_users_telegram_username;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS updated_at;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS created_at;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS access_granted;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS telegram_username;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS telegram_user_id;")
