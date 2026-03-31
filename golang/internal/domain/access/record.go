package access

import "time"

type Record struct {
	TelegramID       int64
	IsBlocked        bool
	HasAccess        bool
	AccessExpiresAt  *time.Time
}

func (r Record) Allows(now time.Time) bool {
	if r.TelegramID == 0 {
		return false
	}
	if r.IsBlocked || !r.HasAccess {
		return false
	}
	if r.AccessExpiresAt != nil && !r.AccessExpiresAt.After(now) {
		return false
	}
	return true
}
