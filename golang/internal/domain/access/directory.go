package access

import "context"

type Directory interface {
	FindByTelegramID(ctx context.Context, telegramID int64) (Record, bool, error)
}
