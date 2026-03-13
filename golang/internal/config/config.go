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
	AllowedUsername      string
	LocalAPIURL          string
	DocRoot              string
	SampleAlbumTitle     string
	SampleAttachmentKeys []string
	S3                   S3Config
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
		TelegramToken:    strings.TrimSpace(os.Getenv("TELEGRAM_BOT_TOKEN")),
		AllowedUsername:  normalizeUsername(defaultString(os.Getenv("ALLOWED_USERNAME"), "chupapimunyao")),
		LocalAPIURL:      defaultString(os.Getenv("LOCAL_API_URL"), "http://127.0.0.1:8080/query"),
		DocRoot:          strings.TrimSpace(os.Getenv("DOC_ROOT")),
		SampleAlbumTitle: defaultString(os.Getenv("SAMPLE_ALBUM_TITLE"), "📄 Residence permit documents"),
		SampleAttachmentKeys: parseCSV(
			defaultString(
				os.Getenv("SAMPLE_ATTACHMENT_KEYS"),
				"residence_permit/residence_permit_page11_img9.png,residence_permit/residence_permit_page2_img1.png,residence_permit/residence_permit_page2_img2.png",
			),
		),
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

	if cfg.TelegramToken == "" {
		return Config{}, fmt.Errorf("TELEGRAM_BOT_TOKEN is missing")
	}

	return cfg, nil
}

func defaultString(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}

func normalizeUsername(value string) string {
	return strings.TrimPrefix(strings.TrimSpace(value), "@")
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
