"""drop password from users for env auth

Revision ID: 7c9e2f4a1b55
Revises: 4b8f0d1c2e3f
Create Date: 2026-04-03 10:30:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7c9e2f4a1b55"
down_revision: Union[str, Sequence[str], None] = "4b8f0d1c2e3f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    users_columns = {column["name"] for column in inspector.get_columns("users")}

    if "password" in users_columns:
        op.drop_column("users", "password")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    users_columns = {column["name"] for column in inspector.get_columns("users")}

    if "password" not in users_columns:
        op.add_column("users", sa.Column("password", sa.String(), nullable=True))
