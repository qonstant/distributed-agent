package accesscache

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
