package access

import "time"

type Record struct {
	TelegramID      int64
	Username        string
	IsBlocked       bool
	AccessExpiresAt *time.Time
}

func (r Record) Allows(now time.Time) bool {
	if r.TelegramID == 0 {
		return false
	}
	if r.IsBlocked {
		return false
	}
	if r.AccessExpiresAt != nil && !r.AccessExpiresAt.After(now) {
		return false
	}
	return true
}
