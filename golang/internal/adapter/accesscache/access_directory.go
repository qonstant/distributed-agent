package accesscache

import (
	"context"
	"encoding/json"
	"strconv"
	"time"

	"github.com/qonstant/distributed-agent/internal/domain/access"
)

type Store interface {
	Get(ctx context.Context, key string) (string, bool, error)
	Set(ctx context.Context, key, value string, ttl time.Duration) error
}

type CachedAccessDirectoryConfig struct {
	KeyPrefix   string
	TTL         time.Duration
	NegativeTTL time.Duration
}

type CachedAccessDirectory struct {
	delegate    access.Directory
	store       Store
	keyPrefix   string
	ttl         time.Duration
	negativeTTL time.Duration
}

type cacheEntry struct {
	Found  bool          `json:"found"`
	Record access.Record `json:"record"`
}

func NewCachedAccessDirectory(delegate access.Directory, store Store, cfg CachedAccessDirectoryConfig) *CachedAccessDirectory {
	keyPrefix := cfg.KeyPrefix
	if keyPrefix == "" {
		keyPrefix = "access:telegram:"
	}
	if cfg.TTL == 0 {
		cfg.TTL = time.Minute
	}
	if cfg.NegativeTTL == 0 {
		cfg.NegativeTTL = 15 * time.Second
	}

	return &CachedAccessDirectory{
		delegate:    delegate,
		store:       store,
		keyPrefix:   keyPrefix,
		ttl:         cfg.TTL,
		negativeTTL: cfg.NegativeTTL,
	}
}

func (d *CachedAccessDirectory) FindByTelegramID(ctx context.Context, telegramID int64) (access.Record, bool, error) {
	key := d.keyPrefix + strconv.FormatInt(telegramID, 10)
	if d.store != nil {
		value, found, err := d.store.Get(ctx, key)
		if err == nil && found {
			var entry cacheEntry
			if json.Unmarshal([]byte(value), &entry) == nil {
				d.writeCacheEntry(ctx, key, entry)
				if !entry.Found {
					return access.Record{}, false, nil
				}
				return entry.Record, true, nil
			}
		}
	}

	record, found, err := d.delegate.FindByTelegramID(ctx, telegramID)
	if err != nil {
		return access.Record{}, false, err
	}

	d.writeCacheEntry(ctx, key, cacheEntry{Found: found, Record: record})

	return record, found, nil
}

func (d *CachedAccessDirectory) writeCacheEntry(ctx context.Context, key string, entry cacheEntry) {
	if d.store == nil {
		return
	}

	ttl := d.negativeTTL
	if entry.Found {
		ttl = d.ttl
	}
	if ttl <= 0 {
		return
	}

	payload, err := json.Marshal(entry)
	if err != nil {
		return
	}

	_ = d.store.Set(ctx, key, string(payload), ttl)
}
