"""add password to users

Revision ID: 16facfd39709
Revises: 9aba0c23f67b
Create Date: 2026-03-30 11:36:57.464393

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '16facfd39709'
down_revision: Union[str, Sequence[str], None] = '9aba0c23f67b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Добавляем password безопасно
    op.add_column('users', sa.Column('password', sa.String(), nullable=True))

    op.execute("UPDATE users SET password = 'temp_password'")

    op.alter_column('users', 'password', nullable=False)