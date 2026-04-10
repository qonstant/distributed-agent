"""rename telegram_user_id to telegram_id

Revision ID: ef3ecb2d404c
Revises: 010f36044f26
Create Date: 2026-03-29 14:08:09.733098

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ef3ecb2d404c'
down_revision: Union[str, Sequence[str], None] = '010f36044f26'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.execute("""
        ALTER TABLE users
        RENAME COLUMN telegram_user_id TO telegram_id;
    """)



def downgrade():
    op.execute("""
        ALTER TABLE users
        RENAME COLUMN telegram_id TO telegram_user_id;
    """)


