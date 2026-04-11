package port

import (
	"context"

	"github.com/qonstant/distributed-agent/internal/domain/persistence"
)

type TurnEventPublisher interface {
	PublishTurn(ctx context.Context, event persistence.TurnEvent) error
}
