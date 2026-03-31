package port

import (
	"context"

	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type ConversationMemory interface {
	Context(ctx context.Context, ownerID int64) (qa.ConversationContext, error)
	RememberTurn(ctx context.Context, ownerID int64, conversationID, userText, assistantText string) error
}

type ContextualAnswerSource interface {
	AskWithHistory(ctx context.Context, question qa.Question, history []qa.ConversationMessage) (qa.DraftResponse, error)
}
