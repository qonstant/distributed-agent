package access

import (
	"context"
	"errors"
	"testing"
	"time"
)

type fakeDirectory struct {
	findFn func(context.Context, int64) (Record, bool, error)
}

func (f fakeDirectory) FindByTelegramID(ctx context.Context, telegramID int64) (Record, bool, error) {
	return f.findFn(ctx, telegramID)
}

func TestPolicyAuthorize(t *testing.T) {
	t.Parallel()

	now := time.Date(2026, 3, 27, 12, 0, 0, 0, time.UTC)
	errDBDown := errors.New("db down")

	tests := []struct {
		name    string
		user    User
		record  Record
		found   bool
		repoErr error
		wantErr error
	}{
		{
			name: "accepts user with active access",
			user: User{TelegramID: 42, Username: "alloweduser"},
			record: Record{
				TelegramID: 42,
				HasAccess: true,
			},
			found: true,
		},
		{
			name:    "rejects empty telegram id",
			user:    User{},
			wantErr: ErrUnauthorized,
		},
		{
			name: "rejects missing user record",
			user: User{TelegramID: 42},
			found: false,
			wantErr: ErrUnauthorized,
		},
		{
			name: "rejects blocked user",
			user: User{TelegramID: 42},
			record: Record{
				TelegramID: 42,
				HasAccess: true,
				IsBlocked: true,
			},
			found: true,
			wantErr: ErrUnauthorized,
		},
		{
			name: "rejects user without access grant",
			user: User{TelegramID: 42},
			record: Record{
				TelegramID: 42,
				HasAccess: false,
			},
			found: true,
			wantErr: ErrUnauthorized,
		},
		{
			name: "rejects expired access",
			user: User{TelegramID: 42},
			record: Record{
				TelegramID:      42,
				HasAccess:       true,
				AccessExpiresAt: ptrTime(now.Add(-time.Minute)),
			},
			found: true,
			wantErr: ErrUnauthorized,
		},
		{
			name:    "returns repository errors",
			user:    User{TelegramID: 42},
			repoErr: errDBDown,
			wantErr: errDBDown,
		},
		{
			name: "accepts future expiration",
			user: User{TelegramID: 42},
			record: Record{
				TelegramID:      42,
				HasAccess:       true,
				AccessExpiresAt: ptrTime(now.Add(time.Minute)),
			},
			found: true,
		},
		{
			name: "accepts nil expiration",
			user: User{TelegramID: 42},
			record: Record{
				TelegramID:      42,
				HasAccess:       true,
				AccessExpiresAt: nil,
			},
			found: true,
		},
		{
			name: "rejects record with mismatched zero telegram id",
			user: User{TelegramID: 42},
			record: Record{
				TelegramID: 0,
				HasAccess: true,
			},
			found: true,
			wantErr: ErrUnauthorized,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			policy := NewPolicy(fakeDirectory{
				findFn: func(_ context.Context, telegramID int64) (Record, bool, error) {
					if telegramID != tt.user.TelegramID {
						t.Fatalf("FindByTelegramID() telegramID = %d, want %d", telegramID, tt.user.TelegramID)
					}
					return tt.record, tt.found, tt.repoErr
				},
			})
			policy.now = func() time.Time { return now }

			err := policy.Authorize(context.Background(), tt.user)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("Authorize() error = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestPolicyAccessMode(t *testing.T) {
	t.Parallel()

	policy := NewPolicy(fakeDirectory{
		findFn: func(context.Context, int64) (Record, bool, error) {
			return Record{}, false, nil
		},
	})
	if got, want := policy.AccessMode(), "users table"; got != want {
		t.Fatalf("AccessMode() = %q, want %q", got, want)
	}
}

func ptrTime(value time.Time) *time.Time {
	return &value
}
