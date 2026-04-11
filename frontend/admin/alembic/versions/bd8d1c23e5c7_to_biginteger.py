"""to biginteger

Revision ID: bd8d1c23e5c7
Revises: 828c8098f121
Create Date: 2026-03-29 16:58:18.963497

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = 'bd8d1c23e5c7'
down_revision: Union[str, Sequence[str], None] = '828c8098f121'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(bind, table_name: str) -> bool:
    return table_name in inspect(bind).get_table_names()


def upgrade() -> None:
    bind = op.get_bind()

    if not _has_table(bind, "users"):
        return

    # 1) Сначала снимаем FK-ограничения, чтобы можно было менять типы
    op.drop_constraint("usage_events_user_id_fkey", "usage_events", type_="foreignkey")
    op.drop_constraint("messages_user_id_fkey", "messages", type_="foreignkey")
    op.drop_constraint("admin_actions_admin_user_id_fkey", "admin_actions", type_="foreignkey")
    op.drop_constraint("admin_actions_target_user_id_fkey", "admin_actions", type_="foreignkey")

    # 2) Меняем типы в users
    op.alter_column(
        "users",
        "id",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=False,
        postgresql_using="id::bigint",
    )

    op.alter_column(
        "users",
        "telegram_username",
        existing_type=sa.Text(),
        type_=sa.String(length=255),
        existing_nullable=True,
        postgresql_using="telegram_username::varchar(255)",
    )

    # 3) Меняем типы FK-колонок
    op.alter_column(
        "usage_events",
        "user_id",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=False,
        postgresql_using="user_id::bigint",
    )

    op.alter_column(
        "messages",
        "user_id",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=True,
        postgresql_using="user_id::bigint",
    )

    op.alter_column(
        "admin_actions",
        "admin_user_id",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=False,
        postgresql_using="admin_user_id::bigint",
    )

    op.alter_column(
        "admin_actions",
        "target_user_id",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=True,
        postgresql_using="target_user_id::bigint",
    )

    # 4) Возвращаем FK
    op.create_foreign_key(
        "usage_events_user_id_fkey",
        "usage_events",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_foreign_key(
        "messages_user_id_fkey",
        "messages",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_foreign_key(
        "admin_actions_admin_user_id_fkey",
        "admin_actions",
        "users",
        ["admin_user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_foreign_key(
        "admin_actions_target_user_id_fkey",
        "admin_actions",
        "users",
        ["target_user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    # Обратный порядок: FK снимаем, типы возвращаем назад, FK ставим снова

    op.drop_constraint("usage_events_user_id_fkey", "usage_events", type_="foreignkey")
    op.drop_constraint("messages_user_id_fkey", "messages", type_="foreignkey")
    op.drop_constraint("admin_actions_admin_user_id_fkey", "admin_actions", type_="foreignkey")
    op.drop_constraint("admin_actions_target_user_id_fkey", "admin_actions", type_="foreignkey")

    op.alter_column(
        "admin_actions",
        "target_user_id",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=True,
        postgresql_using="target_user_id::integer",
    )

    op.alter_column(
        "admin_actions",
        "admin_user_id",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=False,
        postgresql_using="admin_user_id::integer",
    )

    op.alter_column(
        "messages",
        "user_id",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=True,
        postgresql_using="user_id::integer",
    )

    op.alter_column(
        "usage_events",
        "user_id",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=False,
        postgresql_using="user_id::integer",
    )

    op.alter_column(
        "users",
        "telegram_username",
        existing_type=sa.String(length=255),
        type_=sa.Text(),
        existing_nullable=True,
        postgresql_using="telegram_username::text",
    )

    op.alter_column(
        "users",
        "id",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=False,
        postgresql_using="id::integer",
    )

    op.create_foreign_key(
        "usage_events_user_id_fkey",
        "usage_events",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_foreign_key(
        "messages_user_id_fkey",
        "messages",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_foreign_key(
        "admin_actions_admin_user_id_fkey",
        "admin_actions",
        "users",
        ["admin_user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_foreign_key(
        "admin_actions_target_user_id_fkey",
        "admin_actions",
        "users",
        ["target_user_id"],
        ["id"],
        ondelete="SET NULL",
    )