package port

import (
	"context"

	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type AttachmentResolver interface {
	Resolve(ctx context.Context, refs []qa.AttachmentRef) ([]qa.Attachment, error)
}
