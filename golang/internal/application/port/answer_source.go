package port

import (
	"context"

	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type AnswerSource interface {
	Ask(ctx context.Context, question qa.Question) (qa.DraftResponse, error)
}
