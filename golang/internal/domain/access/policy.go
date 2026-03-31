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
	if user.TelegramID == 0 {
		return ErrUnauthorized
	}

	record, found, err := p.directory.FindByTelegramID(ctx, user.TelegramID)
	if err != nil {
		return err
	}
	if !found || !record.Allows(p.now()) {
		return ErrUnauthorized
	}
	return nil
}

func (p Policy) AccessMode() string {
	return "users table"
}

func normalizeUsername(value string) string {
	return strings.TrimPrefix(strings.TrimSpace(value), "@")
}
