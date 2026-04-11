"""rename telegram_id to telegram_user_id

Revision ID: 2978c106969f
Revises: cf97a40f8c02
Create Date: 2026-03-01 15:13:14.469654

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2978c106969f'
down_revision: Union[str, Sequence[str], None] = 'cf97a40f8c02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Переименовываем колонку
    op.alter_column('users', 'telegram_id', new_column_name='telegram_user_id')

def downgrade():
    # Откат — возвращаем старое имя
    op.alter_column('users', 'telegram_user_id', new_column_name='telegram_id')
