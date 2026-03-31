package chatmemory

import (
	"context"
	"fmt"
	"time"

	redislib "github.com/redis/go-redis/v9"
)

type RedisStore struct {
	client *redislib.Client
}

func NewRedisStore(redisURL string) (*RedisStore, error) {
	options, err := redislib.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("parse redis url: %w", err)
	}

	client := redislib.NewClient(options)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		client.Close()
		return nil, fmt.Errorf("ping redis: %w", err)
	}

	return &RedisStore{client: client}, nil
}

func (s *RedisStore) Close() error {
	if s == nil || s.client == nil {
		return nil
	}
	return s.client.Close()
}

func (s *RedisStore) Get(ctx context.Context, key string) (string, bool, error) {
	value, err := s.client.Get(ctx, key).Result()
	if err == redislib.Nil {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return value, true, nil
}

func (s *RedisStore) Set(ctx context.Context, key, value string, ttl time.Duration) error {
	return s.client.Set(ctx, key, value, ttl).Err()
}

func (s *RedisStore) Exists(ctx context.Context, key string) (bool, error) {
	count, err := s.client.Exists(ctx, key).Result()
	if err != nil {
		return false, err
	}
	return count > 0, nil
}

func (s *RedisStore) AppendConversationTurn(
	ctx context.Context,
	key, userPayload, assistantPayload string,
	maxItems int64,
	ttl time.Duration,
) error {
	pipe := s.client.TxPipeline()
	pipe.LPush(ctx, key, userPayload)
	pipe.LPush(ctx, key, assistantPayload)
	pipe.LTrim(ctx, key, 0, maxItems-1)
	pipe.Expire(ctx, key, ttl)
	_, err := pipe.Exec(ctx)
	return err
}
