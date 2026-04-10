"""drop username column

Revision ID: 09119a175e96
Revises: 0cb38886d8a7
Create Date: 2026-03-01 15:29:25.276265

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '09119a175e96'
down_revision: Union[str, Sequence[str], None] = '0cb38886d8a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: drop 'username' column."""
    op.drop_column('users', 'username')


def downgrade() -> None:
    """Downgrade schema: add 'username' column back."""
    op.add_column('users', sa.Column('username', sa.String(), nullable=True))