package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

type Config struct {
	TelegramToken        string
	DBURL                string
	LocalAPIURL          string
	RabbitMQURL          string
	DocRoot              string
	SampleAlbumTitle     string
	SampleAttachmentKeys []string
	Redis                RedisConfig
	S3                   S3Config
}

type RedisConfig struct {
	URL                                 string
	AccessCacheTTL                      time.Duration
	NegativeCacheTTL                    time.Duration
	ConversationMemoryTTL               time.Duration
	ConversationMemoryMaxItems          int
	ConversationMemoryAssistantMaxChars int
}

type S3Config struct {
	Endpoint        string
	AccessKeyID     string
	SecretAccessKey string
	Bucket          string
	UseSSL          bool
	PresignExpiry   time.Duration
}

func Load() (Config, error) {
	_ = godotenv.Load(".env", "golang/.env")

	cfg := Config{
		TelegramToken:        strings.TrimSpace(os.Getenv("TELEGRAM_BOT_TOKEN")),
		DBURL:                strings.TrimSpace(os.Getenv("DB_URL")),
		LocalAPIURL:          strings.TrimSpace(os.Getenv("LOCAL_API_URL")),
		RabbitMQURL:          strings.TrimSpace(os.Getenv("RABBITMQ_URL")),
		DocRoot:              strings.TrimSpace(os.Getenv("DOC_ROOT")),
		SampleAlbumTitle:     defaultString(os.Getenv("SAMPLE_ALBUM_TITLE"), "📄 Residence permit documents"),
		SampleAttachmentKeys: parseCSV(os.Getenv("SAMPLE_ATTACHMENT_KEYS")),
		Redis: RedisConfig{
			URL:                                 strings.TrimSpace(os.Getenv("REDIS_URL")),
			AccessCacheTTL:                      parseDurationEnv("ACCESS_CACHE_TTL", 5*time.Minute),
			NegativeCacheTTL:                    parseDurationEnv("ACCESS_CACHE_NEGATIVE_TTL", time.Minute),
			ConversationMemoryTTL:               parseDurationEnv("CONVERSATION_MEMORY_TTL", 2*time.Hour),
			ConversationMemoryMaxItems:          parseIntEnv("CONVERSATION_MEMORY_MAX_ITEMS", 8),
			ConversationMemoryAssistantMaxChars: parseIntEnv("CONVERSATION_MEMORY_ASSISTANT_MAX_CHARS", 240),
		},
		S3: S3Config{
			Endpoint:        strings.TrimSpace(os.Getenv("S3_ENDPOINT")),
			AccessKeyID:     strings.TrimSpace(os.Getenv("S3_ACCESS_KEY_ID")),
			SecretAccessKey: strings.TrimSpace(os.Getenv("S3_SECRET_ACCESS_KEY")),
			Bucket:          strings.TrimSpace(os.Getenv("S3_BUCKET")),
			UseSSL:          strings.EqualFold(strings.TrimSpace(os.Getenv("S3_USE_SSL")), "true"),
			PresignExpiry:   15 * time.Minute,
		},
	}

	if v := strings.TrimSpace(os.Getenv("PRESIGN_EXPIRY_MIN")); v != "" {
		if minutes, err := strconv.Atoi(v); err == nil && minutes > 0 {
			cfg.S3.PresignExpiry = time.Duration(minutes) * time.Minute
		}
	}

	return cfg, nil
}

func (cfg Config) ValidateBot() error {
	if cfg.TelegramToken == "" {
		return fmt.Errorf("TELEGRAM_BOT_TOKEN is missing")
	}
	if cfg.DBURL == "" {
		return fmt.Errorf("DB_URL is missing")
	}
	if cfg.LocalAPIURL == "" {
		return fmt.Errorf("LOCAL_API_URL is missing")
	}
	return nil
}

func (cfg Config) ValidateWorker() error {
	if cfg.DBURL == "" {
		return fmt.Errorf("DB_URL is missing")
	}
	if cfg.RabbitMQURL == "" {
		return fmt.Errorf("RABBITMQ_URL is missing")
	}
	return nil
}

func defaultString(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}

func parseDurationEnv(name string, fallback time.Duration) time.Duration {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	duration, err := time.ParseDuration(value)
	if err != nil {
		return fallback
	}
	return duration
}

func parseIntEnv(name string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}

	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return fallback
	}

	return parsed
}

func parseCSV(value string) []string {
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		out = append(out, part)
	}
	return out
}
