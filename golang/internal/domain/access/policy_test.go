package access

import (
	"errors"
	"testing"
)

func TestPolicyAuthorize(t *testing.T) {
	t.Parallel()

	policy := NewPolicy("@AllowedUser")

	tests := []struct {
		name    string
		user    User
		wantErr error
	}{
		{
			name: "accepts matching username ignoring case and at sign",
			user: User{Username: "alloweduser"},
		},
		{
			name:    "rejects empty username",
			user:    User{},
			wantErr: ErrUnauthorized,
		},
		{
			name:    "rejects different username",
			user:    User{Username: "someone_else"},
			wantErr: ErrUnauthorized,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := policy.Authorize(tt.user)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("Authorize() error = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestPolicyAllowedUsername(t *testing.T) {
	t.Parallel()

	policy := NewPolicy(" @AllowedUser ")
	if got, want := policy.AllowedUsername(), "AllowedUser"; got != want {
		t.Fatalf("AllowedUsername() = %q, want %q", got, want)
	}
}
