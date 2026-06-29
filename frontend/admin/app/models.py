from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    TIMESTAMP,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import relationship

from .db import Base


usage_event_type_enum = ENUM(
    "message",
    "rag_query",
    "chat_completion",
    "embedding",
    "classification",
    "guardrail",
    "admin_action",
    "other",
    name="usage_event_type",
    create_type=True,
)

classifier_intent_enum = ENUM(
    "GREETING",
    "CHIT_CHAT",
    "CHITCHAT",
    "FACTUAL_QUESTION",
    "GUIDANCE",
    "PROCEDURE",
    "DOCUMENT_REQUEST",
    "COMPARISON",
    "OTHER",
    "OUT_OF_DOMAIN",
    name="classifier_intent",
    create_type=True,
)

admin_action_type_enum = ENUM(
    "grant_access",
    "revoke_access",
    "block_user",
    "unblock_user",
    "verify_payment",
    "extend_access",
    "classify_message",
    "other",
    name="admin_action_type",
    create_type=True,
)

admin_entity_type_enum = ENUM(
    "user",
    "message",
    "conversation",
    "usage_event",
    "other",
    name="admin_entity_type",
    create_type=True,
)


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True, index=True)
    telegram_id = Column(BigInteger, unique=True, nullable=False, index=True)
    username = Column(String(255), nullable=True, index=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    is_blocked = Column(Boolean, nullable=False, default=False, server_default=text("false"))
    is_admin = Column(Boolean, nullable=False, default=False, server_default=text("false"))
    access_expires_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    conversations = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    usage_events = relationship(
        "UsageEvent",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    admin_actions_as_admin = relationship(
        "AdminAction",
        back_populates="admin_user",
        foreign_keys="AdminAction.admin_user_id",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    admin_actions_as_target = relationship(
        "AdminAction",
        back_populates="target_user",
        foreign_keys="AdminAction.target_user_id",
        lazy="selectin",
    )

    __table_args__ = (
        Index("uq_users_telegram_id", "telegram_id", unique=True),
        Index("idx_users_username", "username"),
        Index("idx_users_is_blocked", "is_blocked"),
        Index("idx_users_is_admin", "is_admin"),
        Index("idx_users_access_expires_at", "access_expires_at"),
    )

    @property
    def has_access(self) -> bool:
        if self.is_blocked:
            return False
        if self.access_expires_at is None:
            return True

        expires_at = self.access_expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at > datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    conversation_key = Column(String(255), nullable=False, unique=True)
    summary = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    usage_events = relationship(
        "UsageEvent",
        back_populates="conversation",
        lazy="selectin",
    )

    __table_args__ = (
        Index("uq_conversations_conversation_key", "conversation_key", unique=True),
        Index("idx_conversations_user_id", "user_id"),
        Index("idx_conversations_created_at", "created_at"),
        Index("idx_conversations_updated_at", "updated_at"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(
        BigInteger,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    is_assistant = Column(Boolean, nullable=False, default=False, server_default=text("false"))
    message_text = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    conversation = relationship("Conversation", back_populates="messages")
    classification = relationship(
        "MessageClassification",
        back_populates="message",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="joined",
    )
    usage_events = relationship(
        "UsageEvent",
        back_populates="message",
        lazy="selectin",
    )

    __table_args__ = (
        Index("idx_messages_conversation_id", "conversation_id"),
        Index("idx_messages_is_assistant", "is_assistant"),
        Index("idx_messages_created_at", "created_at"),
        Index("idx_messages_conversation_created_at", "conversation_id", "created_at"),
    )


class MessageClassification(Base):
    __tablename__ = "message_classifications"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    message_id = Column(
        BigInteger,
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    intent = Column(classifier_intent_enum, nullable=False)
    explanation = Column(Text, nullable=True)
    detected_language = Column(String(16), nullable=True)
    classifier_model = Column(String(128), nullable=True)
    classifier_version = Column(String(128), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    message = relationship("Message", back_populates="classification")

    __table_args__ = (
        Index("idx_message_classifications_intent", "intent"),
        Index("idx_message_classifications_detected_language", "detected_language"),
        Index("idx_message_classifications_classifier_model", "classifier_model"),
        Index("idx_message_classifications_created_at", "created_at"),
    )


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    conversation_id = Column(
        BigInteger,
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )
    message_id = Column(
        BigInteger,
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    event_type = Column(usage_event_type_enum, nullable=False)
    input_tokens = Column(Integer, nullable=False, default=0, server_default=text("0"))
    output_tokens = Column(Integer, nullable=False, default=0, server_default=text("0"))
    estimated_cost = Column(
        Numeric(14, 6),
        nullable=False,
        default=0,
        server_default=text("0.000000"),
    )
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="usage_events")
    conversation = relationship("Conversation", back_populates="usage_events")
    message = relationship("Message", back_populates="usage_events")

    __table_args__ = (
        Index("idx_usage_events_user_created_at", "user_id", "created_at"),
        Index("idx_usage_events_conversation_id", "conversation_id"),
        Index("idx_usage_events_message_id", "message_id"),
        Index("idx_usage_events_event_type", "event_type"),
        Index("idx_usage_events_created_at", "created_at"),
    )


class AdminAction(Base):
    __tablename__ = "admin_actions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    admin_user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    target_user_id = Column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action_type = Column(admin_action_type_enum, nullable=False)
    entity_type = Column(admin_entity_type_enum, nullable=True)
    entity_id = Column(BigInteger, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    admin_user = relationship(
        "User",
        back_populates="admin_actions_as_admin",
        foreign_keys=[admin_user_id],
    )
    target_user = relationship(
        "User",
        back_populates="admin_actions_as_target",
        foreign_keys=[target_user_id],
    )

    __table_args__ = (
        Index("idx_admin_actions_admin_user_id", "admin_user_id"),
        Index("idx_admin_actions_target_user_id", "target_user_id"),
        Index("idx_admin_actions_action_type", "action_type"),
        Index("idx_admin_actions_entity", "entity_type", "entity_id"),
        Index("idx_admin_actions_created_at", "created_at"),
    )
