"""new table and change access_granted

Revision ID: 828c8098f121
Revises: rename_index_telegram
Create Date: 2026-03-29 16:27:02.719570

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '828c8098f121'
down_revision: Union[str, Sequence[str], None] = 'rename_index_telegram'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_columns(bind, table_name: str) -> set[str]:
    insp = inspect(bind)
    if table_name not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table_name)}


def _table_exists(bind, table_name: str) -> bool:
    return table_name in inspect(bind).get_table_names()


def _create_enum_type(name: str, values: list[str]) -> None:
    values_sql = ", ".join(f"'{v}'" for v in values)
    op.execute(
        f"""
DO $$
BEGIN
    CREATE TYPE {name} AS ENUM ({values_sql});
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;
"""
    )


def upgrade() -> None:
    bind = op.get_bind()

    # ------------------------------------------------------------
    # STAGE 1 — users: access_granted -> has_access + new columns
    # ------------------------------------------------------------
    users_cols = _table_columns(bind, "users")

    if "access_granted" in users_cols and "has_access" not in users_cols:
        op.execute("ALTER TABLE users RENAME COLUMN access_granted TO has_access;")
        users_cols = _table_columns(bind, "users")

    elif "access_granted" in users_cols and "has_access" in users_cols:
        # если оба поля уже есть, переносим значение и удаляем старое
        op.execute(
            """
            UPDATE users
            SET has_access = COALESCE(has_access, access_granted)
            """
        )
        op.drop_column("users", "access_granted")
        users_cols = _table_columns(bind, "users")

    elif "has_access" not in users_cols:
        op.add_column(
            "users",
            sa.Column(
                "has_access",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("true"),
            ),
        )
        users_cols = _table_columns(bind, "users")

    # новые поля users
    if "first_name" not in users_cols:
        op.add_column("users", sa.Column("first_name", sa.String(255), nullable=True))

    if "last_name" not in users_cols:
        op.add_column("users", sa.Column("last_name", sa.String(255), nullable=True))

    if "is_blocked" not in users_cols:
        op.add_column(
            "users",
            sa.Column(
                "is_blocked",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )

    if "is_admin" not in users_cols:
        op.add_column(
            "users",
            sa.Column(
                "is_admin",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )

    if "access_expires_at" not in users_cols:
        op.add_column(
            "users",
            sa.Column("access_expires_at", sa.TIMESTAMP(timezone=True), nullable=True),
        )

    # индексы users
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_telegram_id ON users (telegram_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_telegram_username ON users (telegram_username);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_is_blocked ON users (is_blocked);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users (is_admin);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_has_access ON users (has_access);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_access_expires_at ON users (access_expires_at);"
    )

    # ------------------------------------------------------------
    # STAGE 2 — ENUM types
    # ------------------------------------------------------------
    _create_enum_type(
        "usage_event_type",
        [
            "message",
            "rag_query",
            "chat_completion",
            "embedding",
            "classification",
            "admin_action",
            "other",
        ],
    )

    _create_enum_type(
        "classifier_intent",
        [
            "GREETING",
            "CHIT_CHAT",
            "FACTUAL_QUESTION",
            "GUIDANCE",
            "DOCUMENT_REQUEST",
            "OTHER",
        ],
    )

    # ------------------------------------------------------------
    # STAGE 3 — usage_events
    # ------------------------------------------------------------
    if not _table_exists(bind, "usage_events"):
        usage_event_type = postgresql.ENUM(
            "message",
            "rag_query",
            "chat_completion",
            "embedding",
            "classification",
            "admin_action",
            "other",
            name="usage_event_type",
            create_type=False,
        )

        op.create_table(
            "usage_events",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "user_id",
                sa.Integer(),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("event_type", usage_event_type, nullable=False),
            sa.Column(
                "input_tokens",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            sa.Column(
                "output_tokens",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            sa.Column(
                "total_tokens",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            sa.Column(
                "estimated_cost",
                sa.Numeric(14, 6),
                nullable=False,
                server_default=sa.text("0.000000"),
            ),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )

        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_events_user_created_at ON usage_events (user_id, created_at);"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_events_event_type ON usage_events (event_type);"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_events_created_at ON usage_events (created_at);"
        )

    # ------------------------------------------------------------
    # STAGE 4 — messages
    # ------------------------------------------------------------
    if not _table_exists(bind, "messages"):
        op.create_table(
            "messages",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "user_id",
                sa.Integer(),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column("message_text", sa.Text(), nullable=False),
            sa.Column("language_code", sa.String(16), nullable=True),
            sa.Column("telegram_message_id", sa.BigInteger(), nullable=True),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )

        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages (user_id);"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_telegram_message_id ON messages (telegram_message_id);"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages (created_at);"
        )

    # ------------------------------------------------------------
    # STAGE 5 — message_classifications (1:1)
    # ------------------------------------------------------------
    if not _table_exists(bind, "message_classifications"):
        classifier_intent = postgresql.ENUM(
            "GREETING",
            "CHIT_CHAT",
            "FACTUAL_QUESTION",
            "GUIDANCE",
            "DOCUMENT_REQUEST",
            "OTHER",
            name="classifier_intent",
            create_type=False,
        )

        op.create_table(
            "message_classifications",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "message_id",
                sa.BigInteger(),
                sa.ForeignKey("messages.id", ondelete="CASCADE"),
                nullable=False,
                unique=True,
            ),
            sa.Column("intent", classifier_intent, nullable=False),
            sa.Column("explain", sa.Text(), nullable=True),
            sa.Column("detected_language", sa.String(32), nullable=True),
            sa.Column("classifier_model", sa.String(100), nullable=True),
            sa.Column("classifier_version", sa.String(100), nullable=True),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )

        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_message_classifications_intent ON message_classifications (intent);"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_message_classifications_detected_language ON message_classifications (detected_language);"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_message_classifications_classifier_model ON message_classifications (classifier_model);"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_message_classifications_created_at ON message_classifications (created_at);"
        )

    # ------------------------------------------------------------
    # STAGE 6 — admin_actions
    # ------------------------------------------------------------
    if not _table_exists(bind, "admin_actions"):
        op.create_table(
            "admin_actions",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "admin_user_id",
                sa.Integer(),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "target_user_id",
                sa.Integer(),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("action_type", sa.String(100), nullable=False),
            sa.Column("entity_type", sa.String(50), nullable=True),
            sa.Column("entity_id", sa.BigInteger(), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )

        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_admin_actions_admin_user_id ON admin_actions (admin_user_id);"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_admin_actions_target_user_id ON admin_actions (target_user_id);"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_admin_actions_entity ON admin_actions (entity_type, entity_id);"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_admin_actions_created_at ON admin_actions (created_at);"
        )


def downgrade() -> None:
    bind = op.get_bind()

    # ------------------------------------------------------------
    # DROP TABLES (reverse order)
    # ------------------------------------------------------------
    if _table_exists(bind, "admin_actions"):
        op.drop_table("admin_actions")

    if _table_exists(bind, "message_classifications"):
        op.drop_table("message_classifications")

    if _table_exists(bind, "messages"):
        op.drop_table("messages")

    if _table_exists(bind, "usage_events"):
        op.drop_table("usage_events")

    # ------------------------------------------------------------
    # DROP ENUM TYPES
    # ------------------------------------------------------------
    op.execute("DROP TYPE IF EXISTS classifier_intent;")
    op.execute("DROP TYPE IF EXISTS usage_event_type;")

    # ------------------------------------------------------------
    # REVERT users columns
    # ------------------------------------------------------------
    users_cols = _table_columns(bind, "users")

    if "has_access" in users_cols and "access_granted" not in users_cols:
        op.alter_column("users", "has_access", new_column_name="access_granted")
    elif "has_access" in users_cols and "access_granted" in users_cols:
        op.execute(
            """
            UPDATE users
            SET access_granted = COALESCE(access_granted, has_access)
            """
        )
        op.drop_column("users", "has_access")

    for col in ["first_name", "last_name", "is_blocked", "is_admin", "access_expires_at"]:
        users_cols = _table_columns(bind, "users")
        if col in users_cols:
            op.drop_column("users", col)