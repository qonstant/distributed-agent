package access

import (
	"context"
	"strings"
	"time"
)

type Policy struct {
	directory Directory
	now       func() time.Time
}

func NewPolicy(directory Directory) Policy {
	return Policy{
		directory: directory,
		now:       time.Now,
	}
}

func (p Policy) Authorize(ctx context.Context, user User) error {
	_, err := p.AuthorizeAndLoad(ctx, user)
	return err
}

func (p Policy) AuthorizeAndLoad(ctx context.Context, user User) (Record, error) {
	if user.TelegramID == 0 {
		return Record{}, ErrUnauthorized
	}

	record, found, err := p.directory.FindByTelegramID(ctx, user.TelegramID)
	if err != nil {
		return Record{}, err
	}
	if !found || !record.Allows(p.now()) {
		return Record{}, ErrUnauthorized
	}
	return record, nil
}

func (p Policy) AccessMode() string {
	return "users table"
}

func (p Policy) RememberRecord(ctx context.Context, record Record) error {
	refresher, ok := p.directory.(RecordRefresher)
	if !ok {
		return nil
	}
	return refresher.StoreRecord(ctx, record)
}

func normalizeUsername(value string) string {
	return strings.TrimPrefix(strings.TrimSpace(value), "@")
}
