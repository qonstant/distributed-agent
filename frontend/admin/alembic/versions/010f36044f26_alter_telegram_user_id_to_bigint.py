"""alter telegram_user_id to bigint

Revision ID: 010f36044f26
Revises: ca0275301121
Create Date: 2026-03-01 15:51:34.285868

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '010f36044f26'
down_revision: Union[str, Sequence[str], None] = 'ca0275301121'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Преобразуем строковую колонку в BIGINT (используя явное приведение)
    op.execute("""
        ALTER TABLE users
        ALTER COLUMN telegram_user_id TYPE BIGINT USING (telegram_user_id::bigint)
    """)
    # При необходимости — добавим недостающие колонки
    op.execute("""
        ALTER TABLE users
        ADD COLUMN IF NOT EXISTS access_granted BOOLEAN DEFAULT false
    """)
    op.execute("""
        ALTER TABLE users
        ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    """)
    op.execute("""
        ALTER TABLE users
        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    """)
    # индексы
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_telegram_user_id ON users (telegram_user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_users_telegram_username ON users (telegram_username);")


def downgrade() -> None:
    # Откат: преобразуем обратно в varchar (если нужно)
    op.execute("""
        ALTER TABLE users
        ALTER COLUMN telegram_user_id TYPE VARCHAR USING (telegram_user_id::varchar)
    """)
    # откат добавления колонок
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS updated_at;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS created_at;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS access_granted;")
    op.execute("DROP INDEX IF EXISTS ix_users_telegram_user_id;")
    op.execute("DROP INDEX IF EXISTS ix_users_telegram_username;")