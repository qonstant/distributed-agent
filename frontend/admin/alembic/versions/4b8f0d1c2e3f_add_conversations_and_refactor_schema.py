"""add conversations and refactor schema

Revision ID: 4b8f0d1c2e3f
Revises: 16facfd39709
Create Date: 2026-04-01 15:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "4b8f0d1c2e3f"
down_revision: Union[str, Sequence[str], None] = "16facfd39709"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ADMIN_ACTION_VALUES = (
    "grant_access",
    "revoke_access",
    "block_user",
    "unblock_user",
    "verify_payment",
    "extend_access",
    "classify_message",
    "other",
)

ADMIN_ENTITY_VALUES = (
    "user",
    "message",
    "conversation",
    "usage_event",
    "other",
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'admin_action_type') THEN
                CREATE TYPE admin_action_type AS ENUM (
                    'grant_access',
                    'revoke_access',
                    'block_user',
                    'unblock_user',
                    'verify_payment',
                    'extend_access',
                    'classify_message',
                    'other'
                );
            END IF;
        END
        $$;
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'admin_entity_type') THEN
                CREATE TYPE admin_entity_type AS ENUM (
                    'user',
                    'message',
                    'conversation',
                    'usage_event',
                    'other'
                );
            END IF;
        END
        $$;
        """
    )

    if "conversations" not in tables:
        op.create_table(
            "conversations",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("conversation_key", sa.String(length=255), nullable=False),
            sa.Column("summary", sa.Text(), nullable=True),
            sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        )
        op.create_index("uq_conversations_conversation_key", "conversations", ["conversation_key"], unique=True)
        op.create_index("idx_conversations_user_id", "conversations", ["user_id"], unique=False)
        op.create_index("idx_conversations_created_at", "conversations", ["created_at"], unique=False)
        op.create_index("idx_conversations_updated_at", "conversations", ["updated_at"], unique=False)

    users_cols = {col["name"] for col in inspector.get_columns("users")}
    if "telegram_username" in users_cols and "username" not in users_cols:
        op.alter_column("users", "telegram_username", new_column_name="username")
    elif "username" not in users_cols:
        op.add_column("users", sa.Column("username", sa.String(length=255), nullable=True))

    op.execute("DROP INDEX IF EXISTS idx_users_telegram_username;")
    op.execute("DROP INDEX IF EXISTS ix_users_telegram_username;")
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);")
    op.execute("DROP INDEX IF EXISTS idx_users_has_access;")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS has_access;")

    messages_cols = {col["name"] for col in inspector.get_columns("messages")}
    if "conversation_id" not in messages_cols:
        op.add_column("messages", sa.Column("conversation_id", sa.BigInteger(), nullable=True))
        op.create_foreign_key(
            "fk_messages_conversation_id",
            "messages",
            "conversations",
            ["conversation_id"],
            ["id"],
            ondelete="CASCADE",
        )

    op.execute(
        """
        INSERT INTO users (telegram_id, username, first_name, last_name, is_blocked, is_admin, access_expires_at, password)
        SELECT -1, 'legacy_orphan', 'Legacy', 'Orphan', false, false, NULL, NULL
        WHERE EXISTS (SELECT 1 FROM messages WHERE user_id IS NULL)
          AND NOT EXISTS (SELECT 1 FROM users WHERE telegram_id = -1);
        """
    )

    op.execute(
        """
        INSERT INTO conversations (user_id, conversation_key, summary)
        SELECT u.id, 'legacy-user-' || u.id::text, 'Legacy imported conversation'
        FROM users u
        WHERE NOT EXISTS (
            SELECT 1
            FROM conversations c
            WHERE c.conversation_key = 'legacy-user-' || u.id::text
        );
        """
    )

    op.execute(
        """
        UPDATE messages m
        SET user_id = (
            SELECT u.id
            FROM users u
            WHERE u.telegram_id = -1
            LIMIT 1
        )
        WHERE m.user_id IS NULL;
        """
    )

    op.execute(
        """
        UPDATE messages m
        SET conversation_id = c.id
        FROM conversations c
        WHERE m.conversation_id IS NULL
          AND c.user_id = m.user_id
          AND c.conversation_key = 'legacy-user-' || m.user_id::text;
        """
    )

    op.execute("ALTER TABLE messages ALTER COLUMN conversation_id SET NOT NULL;")
    op.execute("DROP INDEX IF EXISTS idx_messages_user_id;")
    op.execute("DROP INDEX IF EXISTS idx_messages_telegram_message_id;")
    op.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_created_at ON messages (conversation_id, created_at);")
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS language_code;")
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS telegram_message_id;")
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS user_id;")

    mc_cols = {col["name"] for col in inspector.get_columns("message_classifications")}
    if "explain" in mc_cols and "explanation" not in mc_cols:
        op.alter_column("message_classifications", "explain", new_column_name="explanation")
    op.alter_column("message_classifications", "detected_language", type_=sa.String(length=16), existing_type=sa.String(length=32))
    op.alter_column("message_classifications", "classifier_model", type_=sa.String(length=128), existing_type=sa.String(length=100))
    op.alter_column("message_classifications", "classifier_version", type_=sa.String(length=128), existing_type=sa.String(length=100))

    usage_cols = {col["name"] for col in inspector.get_columns("usage_events")}
    if "conversation_id" not in usage_cols:
        op.add_column("usage_events", sa.Column("conversation_id", sa.BigInteger(), nullable=True))
        op.create_foreign_key(
            "fk_usage_events_conversation_id",
            "usage_events",
            "conversations",
            ["conversation_id"],
            ["id"],
            ondelete="SET NULL",
        )
    if "message_id" not in usage_cols:
        op.add_column("usage_events", sa.Column("message_id", sa.BigInteger(), nullable=True))
        op.create_foreign_key(
            "fk_usage_events_message_id",
            "usage_events",
            "messages",
            ["message_id"],
            ["id"],
            ondelete="SET NULL",
        )

    op.execute(
        """
        UPDATE usage_events ue
        SET conversation_id = c.id
        FROM conversations c
        WHERE ue.conversation_id IS NULL
          AND c.user_id = ue.user_id
          AND c.conversation_key = 'legacy-user-' || ue.user_id::text;
        """
    )
    op.execute("ALTER TABLE usage_events DROP COLUMN IF EXISTS total_tokens;")
    op.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_conversation_id ON usage_events (conversation_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_message_id ON usage_events (message_id);")

    op.execute(
        f"""
        UPDATE admin_actions
        SET notes = CASE
            WHEN notes IS NULL OR notes = '' THEN 'legacy action: ' || action_type
            ELSE 'legacy action: ' || action_type || E'\\n' || notes
        END,
            action_type = 'other'
        WHERE action_type NOT IN {ADMIN_ACTION_VALUES};
        """
    )
    op.execute(
        f"""
        UPDATE admin_actions
        SET entity_type = 'other'
        WHERE entity_type IS NOT NULL
          AND entity_type NOT IN {ADMIN_ENTITY_VALUES};
        """
    )

    op.alter_column(
        "admin_actions",
        "action_type",
        existing_type=sa.String(length=100),
        type_=postgresql.ENUM(*ADMIN_ACTION_VALUES, name="admin_action_type", create_type=False),
        postgresql_using="action_type::admin_action_type",
        existing_nullable=False,
    )
    op.alter_column(
        "admin_actions",
        "entity_type",
        existing_type=sa.String(length=50),
        type_=postgresql.ENUM(*ADMIN_ENTITY_VALUES, name="admin_entity_type", create_type=False),
        postgresql_using="entity_type::admin_entity_type",
        existing_nullable=True,
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_admin_actions_action_type ON admin_actions (action_type);")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_admin_actions_action_type;")
    op.alter_column(
        "admin_actions",
        "entity_type",
        existing_type=postgresql.ENUM(*ADMIN_ENTITY_VALUES, name="admin_entity_type", create_type=False),
        type_=sa.String(length=50),
        postgresql_using="entity_type::text",
        existing_nullable=True,
    )
    op.alter_column(
        "admin_actions",
        "action_type",
        existing_type=postgresql.ENUM(*ADMIN_ACTION_VALUES, name="admin_action_type", create_type=False),
        type_=sa.String(length=100),
        postgresql_using="action_type::text",
        existing_nullable=False,
    )

    op.add_column("usage_events", sa.Column("total_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False))
    op.execute("UPDATE usage_events SET total_tokens = COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0);")
    op.execute("DROP INDEX IF EXISTS idx_usage_events_message_id;")
    op.execute("DROP INDEX IF EXISTS idx_usage_events_conversation_id;")
    op.drop_constraint("fk_usage_events_message_id", "usage_events", type_="foreignkey")
    op.drop_constraint("fk_usage_events_conversation_id", "usage_events", type_="foreignkey")
    op.drop_column("usage_events", "message_id")
    op.drop_column("usage_events", "conversation_id")

    op.alter_column("message_classifications", "classifier_version", type_=sa.String(length=100), existing_type=sa.String(length=128))
    op.alter_column("message_classifications", "classifier_model", type_=sa.String(length=100), existing_type=sa.String(length=128))
    op.alter_column("message_classifications", "detected_language", type_=sa.String(length=32), existing_type=sa.String(length=16))
    op.alter_column("message_classifications", "explanation", new_column_name="explain")

    op.add_column("messages", sa.Column("user_id", sa.BigInteger(), nullable=True))
    op.add_column("messages", sa.Column("language_code", sa.String(length=16), nullable=True))
    op.add_column("messages", sa.Column("telegram_message_id", sa.BigInteger(), nullable=True))
    op.create_foreign_key("fk_messages_user_id", "messages", "users", ["user_id"], ["id"], ondelete="CASCADE")
    op.execute(
        """
        UPDATE messages m
        SET user_id = c.user_id
        FROM conversations c
        WHERE m.conversation_id = c.id;
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages (user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_messages_telegram_message_id ON messages (telegram_message_id);")
    op.execute("DROP INDEX IF EXISTS idx_messages_conversation_created_at;")
    op.execute("DROP INDEX IF EXISTS idx_messages_conversation_id;")
    op.drop_constraint("fk_messages_conversation_id", "messages", type_="foreignkey")
    op.drop_column("messages", "conversation_id")

    op.execute("DROP INDEX IF EXISTS idx_users_username;")
    op.add_column("users", sa.Column("has_access", sa.Boolean(), server_default=sa.text("true"), nullable=False))
    op.execute(
        """
        UPDATE users
        SET has_access = CASE
            WHEN is_blocked THEN false
            WHEN access_expires_at IS NULL THEN true
            WHEN access_expires_at > now() THEN true
            ELSE false
        END;
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_has_access ON users (has_access);")
    op.alter_column("users", "username", new_column_name="telegram_username")
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_telegram_username ON users (telegram_username);")

    op.drop_index("idx_conversations_updated_at", table_name="conversations")
    op.drop_index("idx_conversations_created_at", table_name="conversations")
    op.drop_index("idx_conversations_user_id", table_name="conversations")
    op.drop_index("uq_conversations_conversation_key", table_name="conversations")
    op.drop_table("conversations")

    op.execute("DROP TYPE IF EXISTS admin_entity_type;")
    op.execute("DROP TYPE IF EXISTS admin_action_type;")
