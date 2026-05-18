from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class UsageEventType(str, Enum):
    message = "message"
    rag_query = "rag_query"
    chat_completion = "chat_completion"
    embedding = "embedding"
    classification = "classification"
    admin_action = "admin_action"
    other = "other"


class ClassifierIntent(str, Enum):
    GREETING = "GREETING"
    CHIT_CHAT = "CHIT_CHAT"
    CHITCHAT = "CHITCHAT"
    FACTUAL_QUESTION = "FACTUAL_QUESTION"
    GUIDANCE = "GUIDANCE"
    PROCEDURE = "PROCEDURE"
    DOCUMENT_REQUEST = "DOCUMENT_REQUEST"
    COMPARISON = "COMPARISON"
    OTHER = "OTHER"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"


class AdminActionType(str, Enum):
    grant_access = "grant_access"
    revoke_access = "revoke_access"
    block_user = "block_user"
    unblock_user = "unblock_user"
    verify_payment = "verify_payment"
    extend_access = "extend_access"
    classify_message = "classify_message"
    other = "other"


class AdminEntityType(str, Enum):
    user = "user"
    message = "message"
    conversation = "conversation"
    usage_event = "usage_event"
    other = "other"


class UserBase(BaseModel):
    telegram_id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_blocked: Optional[bool] = False
    is_admin: Optional[bool] = False
    access_expires_at: Optional[datetime] = None


class UserCreate(UserBase):
    pass


class UserUpdate(BaseModel):
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_blocked: Optional[bool] = None
    is_admin: Optional[bool] = None
    access_expires_at: Optional[datetime] = None


class UserOut(UserBase):
    id: int
    has_access: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationBase(BaseModel):
    user_id: int
    conversation_key: str
    summary: Optional[str] = None


class ConversationCreate(ConversationBase):
    pass


class ConversationOut(ConversationBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UsageEventBase(BaseModel):
    user_id: int
    conversation_id: Optional[int] = None
    message_id: Optional[int] = None
    event_type: UsageEventType
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: Decimal = Decimal("0.0")


class UsageEventCreate(UsageEventBase):
    pass


class UsageEventOut(UsageEventBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageBase(BaseModel):
    conversation_id: int
    message_text: str


class MessageCreate(MessageBase):
    pass


class MessageOut(MessageBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageClassificationBase(BaseModel):
    message_id: int
    intent: ClassifierIntent
    explanation: Optional[str] = None
    detected_language: Optional[str] = None
    classifier_model: Optional[str] = None
    classifier_version: Optional[str] = None


class MessageClassificationCreate(MessageClassificationBase):
    pass


class MessageClassificationOut(MessageClassificationBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AdminActionBase(BaseModel):
    admin_user_id: int
    target_user_id: Optional[int] = None
    action_type: AdminActionType
    entity_type: Optional[AdminEntityType] = None
    entity_id: Optional[int] = None
    notes: Optional[str] = None


class AdminActionCreate(AdminActionBase):
    pass


class AdminActionOut(AdminActionBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserProfileOut(UserOut):
    conversations: List[ConversationOut] = Field(default_factory=list)
    messages: List[MessageOut] = Field(default_factory=list)
    usage_events: List[UsageEventOut] = Field(default_factory=list)
    admin_actions_as_admin: List[AdminActionOut] = Field(default_factory=list)
    admin_actions_as_target: List[AdminActionOut] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
