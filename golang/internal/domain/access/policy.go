package access

import "strings"

type Policy struct {
	allowedUsername string
}

func NewPolicy(allowedUsername string) Policy {
	return Policy{allowedUsername: normalizeUsername(allowedUsername)}
}

func (p Policy) Authorize(user User) error {
	if normalizeUsername(user.Username) == "" {
		return ErrUnauthorized
	}
	if !strings.EqualFold(normalizeUsername(user.Username), p.allowedUsername) {
		return ErrUnauthorized
	}
	return nil
}

func (p Policy) AllowedUsername() string {
	return p.allowedUsername
}

func normalizeUsername(value string) string {
	return strings.TrimPrefix(strings.TrimSpace(value), "@")
}
