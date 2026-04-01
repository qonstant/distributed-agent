package accesscache

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/qonstant/distributed-agent/internal/domain/access"
)

type fakeDirectory struct {
	findFn func(context.Context, int64) (access.Record, bool, error)
}

func (f fakeDirectory) FindByTelegramID(ctx context.Context, telegramID int64) (access.Record, bool, error) {
	return f.findFn(ctx, telegramID)
}

type fakeStore struct {
	getFn func(context.Context, string) (string, bool, error)
	setFn func(context.Context, string, string, time.Duration) error
}

func (f fakeStore) Get(ctx context.Context, key string) (string, bool, error) {
	return f.getFn(ctx, key)
}

func (f fakeStore) Set(ctx context.Context, key, value string, ttl time.Duration) error {
	return f.setFn(ctx, key, value, ttl)
}

func TestCachedAccessDirectoryFindByTelegramID(t *testing.T) {
	t.Parallel()

	now := time.Date(2026, 3, 29, 12, 0, 0, 0, time.UTC)
	activeRecord := access.Record{
		TelegramID:      42,
		AccessExpiresAt: ptrTime(now.Add(time.Minute)),
	}

	t.Run("returns cached positive hit without querying delegate", func(t *testing.T) {
		t.Parallel()

		delegateCalled := false
		var refreshedKey string
		var refreshedValue string
		var refreshedTTL time.Duration
		store := fakeStore{
			getFn: func(_ context.Context, key string) (string, bool, error) {
				if got, want := key, "access:v2:telegram:42"; got != want {
					t.Fatalf("Get() key = %q, want %q", got, want)
				}
				return `{"found":true,"record":{"TelegramID":42,"IsBlocked":false}}`, true, nil
			},
			setFn: func(_ context.Context, key, value string, ttl time.Duration) error {
				refreshedKey = key
				refreshedValue = value
				refreshedTTL = ttl
				return nil
			},
		}
		directory := NewCachedAccessDirectory(fakeDirectory{
			findFn: func(context.Context, int64) (access.Record, bool, error) {
				delegateCalled = true
				return access.Record{}, false, nil
			},
		}, store, CachedAccessDirectoryConfig{
			KeyPrefix:   "access:v2:telegram:",
			TTL:         time.Minute,
			NegativeTTL: 15 * time.Second,
		})

		record, found, err := directory.FindByTelegramID(context.Background(), 42)
		if err != nil {
			t.Fatalf("FindByTelegramID() error = %v", err)
		}
		if !found {
			t.Fatal("FindByTelegramID() found = false, want true")
		}
		if record.TelegramID != 42 {
			t.Fatalf("FindByTelegramID() record = %+v", record)
		}
		if delegateCalled {
			t.Fatal("delegate was called on cache hit")
		}
		if got, want := refreshedKey, "access:v2:telegram:42"; got != want {
			t.Fatalf("Set() key = %q, want %q", got, want)
		}
		if got, want := refreshedTTL, time.Minute; got != want {
			t.Fatalf("Set() ttl = %v, want %v", got, want)
		}
		if refreshedValue == "" {
			t.Fatal("Set() value was empty")
		}
	})

	t.Run("returns cached negative hit without querying delegate", func(t *testing.T) {
		t.Parallel()

		delegateCalled := false
		var refreshedKey string
		var refreshedValue string
		var refreshedTTL time.Duration
		store := fakeStore{
			getFn: func(context.Context, string) (string, bool, error) {
				return `{"found":false,"record":{"TelegramID":0,"IsBlocked":false}}`, true, nil
			},
			setFn: func(_ context.Context, key, value string, ttl time.Duration) error {
				refreshedKey = key
				refreshedValue = value
				refreshedTTL = ttl
				return nil
			},
		}
		directory := NewCachedAccessDirectory(fakeDirectory{
			findFn: func(context.Context, int64) (access.Record, bool, error) {
				delegateCalled = true
				return access.Record{}, false, nil
			},
		}, store, CachedAccessDirectoryConfig{
			KeyPrefix:   "access:v2:telegram:",
			TTL:         time.Minute,
			NegativeTTL: 15 * time.Second,
		})

		record, found, err := directory.FindByTelegramID(context.Background(), 42)
		if err != nil {
			t.Fatalf("FindByTelegramID() error = %v", err)
		}
		if found {
			t.Fatal("FindByTelegramID() found = true, want false")
		}
		if record != (access.Record{}) {
			t.Fatalf("FindByTelegramID() record = %+v, want zero", record)
		}
		if delegateCalled {
			t.Fatal("delegate was called on negative cache hit")
		}
		if got, want := refreshedKey, "access:v2:telegram:42"; got != want {
			t.Fatalf("Set() key = %q, want %q", got, want)
		}
		if got, want := refreshedTTL, 15*time.Second; got != want {
			t.Fatalf("Set() ttl = %v, want %v", got, want)
		}
		if refreshedValue == "" {
			t.Fatal("Set() value was empty")
		}
	})

	t.Run("loads from delegate on miss and caches positive result", func(t *testing.T) {
		t.Parallel()

		var cachedKey string
		var cachedValue string
		var cachedTTL time.Duration

		store := fakeStore{
			getFn: func(context.Context, string) (string, bool, error) {
				return "", false, nil
			},
			setFn: func(_ context.Context, key, value string, ttl time.Duration) error {
				cachedKey = key
				cachedValue = value
				cachedTTL = ttl
				return nil
			},
		}
		directory := NewCachedAccessDirectory(fakeDirectory{
			findFn: func(_ context.Context, telegramID int64) (access.Record, bool, error) {
				if got, want := telegramID, int64(42); got != want {
					t.Fatalf("FindByTelegramID() telegramID = %d, want %d", got, want)
				}
				return activeRecord, true, nil
			},
		}, store, CachedAccessDirectoryConfig{
			KeyPrefix:   "access:v2:telegram:",
			TTL:         time.Minute,
			NegativeTTL: 15 * time.Second,
		})

		record, found, err := directory.FindByTelegramID(context.Background(), 42)
		if err != nil {
			t.Fatalf("FindByTelegramID() error = %v", err)
		}
		if !found {
			t.Fatal("FindByTelegramID() found = false, want true")
		}
		if got, want := record, activeRecord; got != want {
			t.Fatalf("FindByTelegramID() record = %+v, want %+v", got, want)
		}
		if got, want := cachedKey, "access:v2:telegram:42"; got != want {
			t.Fatalf("Set() key = %q, want %q", got, want)
		}
		if got, want := cachedTTL, time.Minute; got != want {
			t.Fatalf("Set() ttl = %v, want %v", got, want)
		}
		if cachedValue == "" {
			t.Fatal("Set() value was empty")
		}
	})

	t.Run("loads from delegate on miss and caches negative result with negative ttl", func(t *testing.T) {
		t.Parallel()

		var cachedTTL time.Duration
		store := fakeStore{
			getFn: func(context.Context, string) (string, bool, error) {
				return "", false, nil
			},
			setFn: func(_ context.Context, _ string, _ string, ttl time.Duration) error {
				cachedTTL = ttl
				return nil
			},
		}
		directory := NewCachedAccessDirectory(fakeDirectory{
			findFn: func(context.Context, int64) (access.Record, bool, error) {
				return access.Record{}, false, nil
			},
		}, store, CachedAccessDirectoryConfig{
			KeyPrefix:   "access:v2:telegram:",
			TTL:         time.Minute,
			NegativeTTL: 15 * time.Second,
		})

		_, found, err := directory.FindByTelegramID(context.Background(), 42)
		if err != nil {
			t.Fatalf("FindByTelegramID() error = %v", err)
		}
		if found {
			t.Fatal("FindByTelegramID() found = true, want false")
		}
		if got, want := cachedTTL, 15*time.Second; got != want {
			t.Fatalf("Set() ttl = %v, want %v", got, want)
		}
	})

	t.Run("falls back to delegate when cache get errors", func(t *testing.T) {
		t.Parallel()

		store := fakeStore{
			getFn: func(context.Context, string) (string, bool, error) {
				return "", false, errors.New("redis unavailable")
			},
			setFn: func(context.Context, string, string, time.Duration) error {
				return nil
			},
		}
		directory := NewCachedAccessDirectory(fakeDirectory{
			findFn: func(context.Context, int64) (access.Record, bool, error) {
				return activeRecord, true, nil
			},
		}, store, CachedAccessDirectoryConfig{})

		record, found, err := directory.FindByTelegramID(context.Background(), 42)
		if err != nil {
			t.Fatalf("FindByTelegramID() error = %v", err)
		}
		if !found || record.TelegramID != activeRecord.TelegramID {
			t.Fatalf("FindByTelegramID() = (%+v, %t), want delegate result", record, found)
		}
	})

	t.Run("falls back to delegate when cached payload is invalid", func(t *testing.T) {
		t.Parallel()

		store := fakeStore{
			getFn: func(context.Context, string) (string, bool, error) {
				return "{not-json", true, nil
			},
			setFn: func(context.Context, string, string, time.Duration) error {
				return nil
			},
		}
		directory := NewCachedAccessDirectory(fakeDirectory{
			findFn: func(context.Context, int64) (access.Record, bool, error) {
				return activeRecord, true, nil
			},
		}, store, CachedAccessDirectoryConfig{})

		record, found, err := directory.FindByTelegramID(context.Background(), 42)
		if err != nil {
			t.Fatalf("FindByTelegramID() error = %v", err)
		}
		if !found || record.TelegramID != activeRecord.TelegramID {
			t.Fatalf("FindByTelegramID() = (%+v, %t), want delegate result", record, found)
		}
	})

	t.Run("returns delegate error without caching", func(t *testing.T) {
		t.Parallel()

		delegateErr := errors.New("postgres unavailable")
		store := fakeStore{
			getFn: func(context.Context, string) (string, bool, error) {
				return "", false, nil
			},
			setFn: func(context.Context, string, string, time.Duration) error {
				t.Fatal("Set should not be called when delegate returns error")
				return nil
			},
		}
		directory := NewCachedAccessDirectory(fakeDirectory{
			findFn: func(context.Context, int64) (access.Record, bool, error) {
				return access.Record{}, false, delegateErr
			},
		}, store, CachedAccessDirectoryConfig{})

		_, _, err := directory.FindByTelegramID(context.Background(), 42)
		if !errors.Is(err, delegateErr) {
			t.Fatalf("FindByTelegramID() error = %v, want %v", err, delegateErr)
		}
	})

	t.Run("ignores cache set errors after loading delegate result", func(t *testing.T) {
		t.Parallel()

		store := fakeStore{
			getFn: func(context.Context, string) (string, bool, error) {
				return "", false, nil
			},
			setFn: func(context.Context, string, string, time.Duration) error {
				return errors.New("redis write failed")
			},
		}
		directory := NewCachedAccessDirectory(fakeDirectory{
			findFn: func(context.Context, int64) (access.Record, bool, error) {
				return activeRecord, true, nil
			},
		}, store, CachedAccessDirectoryConfig{
			TTL: time.Minute,
		})

		record, found, err := directory.FindByTelegramID(context.Background(), 42)
		if err != nil {
			t.Fatalf("FindByTelegramID() error = %v", err)
		}
		if !found || record.TelegramID != activeRecord.TelegramID {
			t.Fatalf("FindByTelegramID() = (%+v, %t), want delegate result", record, found)
		}
	})
}

func ptrTime(value time.Time) *time.Time {
	return &value
}
