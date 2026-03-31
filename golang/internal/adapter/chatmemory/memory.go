package chatmemory

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type Store interface {
	Get(ctx context.Context, key string) (string, bool, error)
	Set(ctx context.Context, key, value string, ttl time.Duration) error
	Exists(ctx context.Context, key string) (bool, error)
	AppendConversationTurn(ctx context.Context, key, userPayload, assistantPayload string, maxItems int64, ttl time.Duration) error
}

type Config struct {
	TTL                         time.Duration
	MaxItems                    int64
	ConversationKeyPrefix       string
	ActiveConversationKeyPrefix string
	Now                         func() time.Time
}

type Memory struct {
	store                       Store
	ttl                         time.Duration
	maxItems                    int64
	conversationKeyPrefix       string
	activeConversationKeyPrefix string
	now                         func() time.Time
}

func New(store Store, cfg Config) *Memory {
	if cfg.TTL <= 0 {
		cfg.TTL = 2 * time.Hour
	}
	if cfg.MaxItems <= 0 {
		cfg.MaxItems = 8
	}
	if cfg.ConversationKeyPrefix == "" {
		cfg.ConversationKeyPrefix = "chat:conv:"
	}
	if cfg.ActiveConversationKeyPrefix == "" {
		cfg.ActiveConversationKeyPrefix = "chat:user:"
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}

	return &Memory{
		store:                       store,
		ttl:                         cfg.TTL,
		maxItems:                    cfg.MaxItems,
		conversationKeyPrefix:       cfg.ConversationKeyPrefix,
		activeConversationKeyPrefix: cfg.ActiveConversationKeyPrefix,
		now:                         cfg.Now,
	}
}

func (m *Memory) Context(ctx context.Context, ownerID int64) (qa.ConversationContext, error) {
	if m == nil || m.store == nil {
		return qa.ConversationContext{}, nil
	}

	conversationID, found, err := m.store.Get(ctx, m.activeConversationKey(ownerID))
	if err != nil {
		return qa.ConversationContext{}, err
	}
	if !found || strings.TrimSpace(conversationID) == "" {
		return qa.ConversationContext{ID: m.newConversationID(ownerID)}, nil
	}

	messagesKey := m.messagesKey(conversationID)
	exists, err := m.store.Exists(ctx, messagesKey)
	if err != nil {
		return qa.ConversationContext{}, err
	}
	if !exists {
		return qa.ConversationContext{ID: m.newConversationID(ownerID)}, nil
	}

	return qa.ConversationContext{ID: conversationID}, nil
}

func (m *Memory) RememberTurn(ctx context.Context, ownerID int64, conversationID, userText, assistantText string) error {
	if m == nil || m.store == nil {
		return nil
	}
	if strings.TrimSpace(conversationID) == "" {
		conversationID = m.newConversationID(ownerID)
	}

	userPayload, err := marshalMessage(qa.ConversationRoleUser, userText, m.now())
	if err != nil {
		return err
	}
	assistantPayload, err := marshalMessage(qa.ConversationRoleAssistant, assistantText, m.now())
	if err != nil {
		return err
	}

	if err := m.store.AppendConversationTurn(
		ctx,
		m.messagesKey(conversationID),
		userPayload,
		assistantPayload,
		m.maxItems,
		m.ttl,
	); err != nil {
		return err
	}

	return m.store.Set(ctx, m.activeConversationKey(ownerID), conversationID, m.ttl)
}

func marshalMessage(role, text string, now time.Time) (string, error) {
	payload, err := json.Marshal(qa.ConversationMessage{
		Role:      role,
		Text:      text,
		Timestamp: now.Unix(),
	})
	if err != nil {
		return "", err
	}
	return string(payload), nil
}

func (m *Memory) activeConversationKey(ownerID int64) string {
	return m.activeConversationKeyPrefix + strconv.FormatInt(ownerID, 10) + ":active_conversation"
}

func (m *Memory) messagesKey(conversationID string) string {
	return m.conversationKeyPrefix + conversationID + ":messages"
}

func (m *Memory) newConversationID(ownerID int64) string {
	return fmt.Sprintf("%d-%d", ownerID, m.now().UnixNano())
}
