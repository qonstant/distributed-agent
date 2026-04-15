"""rename telegram index

Revision ID: 516cff0fe1d2
Revises: ef3ecb2d404c
Create Date: 2026-03-29 14:25:16.664590

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'rename_index_telegram'
down_revision: Union[str, Sequence[str], None] = 'ef3ecb2d404c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade():
    op.execute("""
    DO $$
    BEGIN
        -- если старый индекс есть И нового нет
        IF EXISTS (
            SELECT 1 FROM pg_class WHERE relname = 'ix_users_telegram_user_id'
        ) AND NOT EXISTS (
            SELECT 1 FROM pg_class WHERE relname = 'ix_users_telegram_id'
        ) THEN
            ALTER INDEX ix_users_telegram_user_id RENAME TO ix_users_telegram_id;
        END IF;
    END$$;
    """)


def downgrade():
    op.execute("""
        ALTER INDEX ix_users_telegram_id
        RENAME TO ix_users_telegram_user_id;
    """)
