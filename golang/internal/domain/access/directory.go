package access

import "context"

type Directory interface {
	FindByTelegramID(ctx context.Context, telegramID int64) (Record, bool, error)
}

type RecordRefresher interface {
	StoreRecord(ctx context.Context, record Record) error
}
