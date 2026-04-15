"""add telegram_username column to users

Revision ID: 0cb38886d8a7
Revises: 2978c106969f
Create Date: 2026-03-01 15:20:36.553154

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0cb38886d8a7'
down_revision: Union[str, Sequence[str], None] = '2978c106969f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('users', sa.Column('telegram_username', sa.String(), nullable=True))

def downgrade():
    op.drop_column('users', 'telegram_username')
