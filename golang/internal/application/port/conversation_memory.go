package port

import (
	"context"

	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type ConversationMemory interface {
	Context(ctx context.Context, ownerID int64) (qa.ConversationContext, error)
	RememberTurn(ctx context.Context, ownerID int64, conversationID, userText, assistantText string, assistantAttachments []qa.ConversationAttachment) error
}

type ConversationAwareAnswerSource interface {
	AskWithConversation(ctx context.Context, question qa.Question, conversationID, preferredName string) (qa.DraftResponse, error)
}
