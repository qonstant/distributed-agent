"""sync indexes

Revision ID: 9aba0c23f67b
Revises: bd8d1c23e5c7
Create Date: 2026-03-29 22:57:41.652365

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9aba0c23f67b'
down_revision: Union[str, Sequence[str], None] = 'bd8d1c23e5c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    pass

def downgrade():
    pass