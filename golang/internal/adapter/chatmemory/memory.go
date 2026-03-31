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
	AssistantMaxChars           int
	ConversationKeyPrefix       string
	ActiveConversationKeyPrefix string
	Now                         func() time.Time
}

type Memory struct {
	store                       Store
	ttl                         time.Duration
	maxItems                    int64
	assistantMaxChars           int
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
	if cfg.AssistantMaxChars <= 0 {
		cfg.AssistantMaxChars = 240
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
		assistantMaxChars:           cfg.AssistantMaxChars,
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

func (m *Memory) RememberTurn(
	ctx context.Context,
	ownerID int64,
	conversationID, userText, assistantText string,
	assistantAttachments []qa.ConversationAttachment,
) error {
	if m == nil || m.store == nil {
		return nil
	}
	if strings.TrimSpace(conversationID) == "" {
		conversationID = m.newConversationID(ownerID)
	}

	userPayload, err := marshalMessage(qa.ConversationRoleUser, userText, nil, m.now())
	if err != nil {
		return err
	}
	assistantPayload, err := marshalMessage(
		qa.ConversationRoleAssistant,
		truncateText(assistantText, m.assistantMaxChars),
		normalizeAttachments(assistantAttachments),
		m.now(),
	)
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

func truncateText(text string, maxChars int) string {
	if maxChars <= 0 {
		return text
	}

	runes := []rune(text)
	if len(runes) <= maxChars {
		return text
	}
	if maxChars <= 3 {
		return string(runes[:maxChars])
	}
	if maxChars <= 10 {
		return string(runes[:maxChars-3]) + "..."
	}

	const separator = "\n...\n"
	separatorRunes := []rune(separator)
	available := maxChars - len(separatorRunes)
	if available <= 1 {
		return string(runes[:maxChars-3]) + "..."
	}

	head := (available + 1) / 2
	tail := available / 2

	return string(runes[:head]) + separator + string(runes[len(runes)-tail:])
}

func normalizeAttachments(attachments []qa.ConversationAttachment) []qa.ConversationAttachment {
	if len(attachments) == 0 {
		return nil
	}

	normalized := make([]qa.ConversationAttachment, 0, len(attachments))
	for _, attachment := range attachments {
		name := strings.TrimSpace(attachment.Name)
		kind := attachment.Kind
		if name == "" {
			continue
		}
		if kind == "" {
			kind = qa.AttachmentDocument
		}
		normalized = append(normalized, qa.ConversationAttachment{
			Name: name,
			Kind: kind,
		})
	}
	if len(normalized) == 0 {
		return nil
	}
	return normalized
}

func marshalMessage(role, text string, attachments []qa.ConversationAttachment, now time.Time) (string, error) {
	payload, err := json.Marshal(qa.ConversationMessage{
		Role:        role,
		Text:        text,
		Timestamp:   now.Unix(),
		Attachments: attachments,
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
