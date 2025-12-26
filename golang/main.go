package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
	"github.com/joho/godotenv"

	minio "github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

/* ============================
   GLOBALS
============================ */

var (
	s3Client      *minio.Client
	s3Bucket      string
	presignExpiry = 15 * time.Minute
)

/* ============================
   MAIN
============================ */

func main() {
	_ = godotenv.Load()
	if err := initS3(); err != nil {
		log.Printf("S3 init warning (continuing): %v", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	token := os.Getenv("TELEGRAM_BOT_TOKEN")
	if token == "" {
		log.Fatal("TELEGRAM_BOT_TOKEN is missing")
	}

	b, err := bot.New(token, bot.WithDefaultHandler(defaultHandler))
	if err != nil {
		log.Fatalf("failed to create bot: %v", err)
	}

	// Register handlers
	b.RegisterHandler(bot.HandlerTypeMessageText, "/start", bot.MatchTypeCommand, startHandler)
	b.RegisterHandler(bot.HandlerTypeMessageText, "/help", bot.MatchTypeCommand, helpHandler)
	b.RegisterHandler(bot.HandlerTypeMessageText, "/randompic", bot.MatchTypeCommand, randomPicHandler)

	// regex dispatcher
	cmdRe := regexp.MustCompile(`^/(start|help|randompic)(@\w+)?`)
	b.RegisterHandlerRegexp(bot.HandlerTypeMessageText, cmdRe, func(ctx context.Context, b *bot.Bot, u *models.Update) {
		if u == nil || u.Message == nil {
			return
		}
		switch {
		case strings.HasPrefix(u.Message.Text, "/start"):
			startHandler(ctx, b, u)
		case strings.HasPrefix(u.Message.Text, "/help"):
			helpHandler(ctx, b, u)
		case strings.HasPrefix(u.Message.Text, "/randompic"):
			randomPicHandler(ctx, b, u)
		}
	})

	log.Println("🤖 Bot started")
	go b.Start(ctx)
	<-ctx.Done()
	log.Println("🛑 Bot stopped")
}

/* ============================
   S3 INIT
============================ */

func initS3() error {
	endpoint := os.Getenv("S3_ENDPOINT")
	access := os.Getenv("S3_ACCESS_KEY_ID")
	secret := os.Getenv("S3_SECRET_ACCESS_KEY")
	bucket := os.Getenv("S3_BUCKET")

	if endpoint == "" || access == "" || secret == "" || bucket == "" {
		return fmt.Errorf("missing S3 env vars (S3_ENDPOINT,S3_ACCESS_KEY_ID,S3_SECRET_ACCESS_KEY,S3_BUCKET)")
	}

	useSSL := strings.ToLower(os.Getenv("S3_USE_SSL")) == "true"

	if v := os.Getenv("PRESIGN_EXPIRY_MIN"); v != "" {
		if m, err := strconv.Atoi(v); err == nil && m > 0 {
			presignExpiry = time.Duration(m) * time.Minute
		}
	}

	client, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(access, secret, ""),
		Secure: useSSL,
	})
	if err != nil {
		return fmt.Errorf("minio.New: %w", err)
	}

	ok, err := client.BucketExists(context.Background(), bucket)
	if err != nil || !ok {
		return fmt.Errorf("bucket %q not accessible: %w", bucket, err)
	}

	s3Client = client
	s3Bucket = bucket
	log.Printf("✅ S3 connected: endpoint=%s bucket=%s secure=%v", endpoint, bucket, useSSL)
	return nil
}

/* ============================
   HELPERS
============================ */

func username(u *models.Update) string {
	if u == nil || u.Message == nil || u.Message.From == nil {
		return ""
	}
	if u.Message.From.Username != "" {
		return u.Message.From.Username
	}
	return fmt.Sprintf("%s %s", u.Message.From.FirstName, u.Message.From.LastName)
}

func chatID(u *models.Update) int64 {
	if u == nil || u.Message == nil {
		return 0
	}
	return u.Message.Chat.ID
}

/* ============================
   HANDLERS
============================ */

func defaultHandler(ctx context.Context, b *bot.Bot, u *models.Update) {
	if u == nil || u.Message == nil {
		return
	}
	_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID: u.Message.Chat.ID,
		Text:   "Try /randompic to get two images in one message.",
	})
}

func startHandler(ctx context.Context, b *bot.Bot, u *models.Update) {
	if u == nil || u.Message == nil {
		return
	}
	_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID: u.Message.Chat.ID,
		Text:   "Hello! Use /randompic to receive two images in a single album message.",
	})
}

func helpHandler(ctx context.Context, b *bot.Bot, u *models.Update) {
	if u == nil || u.Message == nil {
		return
	}
	_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID: u.Message.Chat.ID,
		Text:   "/randompic - send two residence-permit images as a single album (one message, one caption).",
	})
}

/* ============================
   RANDOMPIC Handler
============================ */

func randomPicHandler(ctx context.Context, b *bot.Bot, u *models.Update) {
	if u == nil || u.Message == nil {
		return
	}
	chat := u.Message.Chat.ID

	keys := []string{
		"residence_permit/residence_permit_page11_img9.png",
		"residence_permit/residence_permit_page2_img1.png",
		"residence_permit/residence_permit_page2_img2.png",
	}

	// Attempt to send as a single album by uploading both files in one request
	if err := uploadMediaGroupFromS3(ctx, chat, keys); err != nil {
		log.Printf("[randomPicHandler] uploadMediaGroupFromS3 failed: %v", err)
		_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: chat,
			Text:   fmt.Sprintf("Failed to send images as album: %v", err),
		})
		return
	}
	// success, nothing else to send
}

/* ============================
   Upload media group from S3 (single HTTP request)
   Builds multipart form with:
   - chat_id
   - media: JSON array referring to attach://file0, attach://file1
   - file parts named file0, file1
   This ensures a single Telegram message (album) with one caption.
============================ */

func uploadMediaGroupFromS3(ctx context.Context, chatID int64, keys []string) error {
	if s3Client == nil {
		return fmt.Errorf("s3 client not configured")
	}
	if len(keys) == 0 {
		return fmt.Errorf("no keys provided")
	}
	// Download files from S3 into memory
	type fileBuf struct {
		FieldName string // form field name e.g. file0
		FileName  string // original file name
		Data      []byte
	}
	files := make([]fileBuf, 0, len(keys))

	for i, key := range keys {
		obj, err := s3Client.GetObject(ctx, s3Bucket, key, minio.GetObjectOptions{})
		if err != nil {
			return fmt.Errorf("GetObject(%s): %w", key, err)
		}
		data, err := io.ReadAll(obj)
		_ = obj.Close()
		if err != nil {
			return fmt.Errorf("read object %s: %w", key, err)
		}
		files = append(files, fileBuf{
			FieldName: fmt.Sprintf("file%d", i),
			FileName:  filepath.Base(key),
			Data:      data,
		})
	}

	// Build media JSON referencing attach://file0 etc.
	type mediaItem struct {
		Type    string `json:"type"`
		Media   string `json:"media"`
		Caption string `json:"caption,omitempty"`
	}
	mediaArr := make([]mediaItem, 0, len(files))
	for i, f := range files {
		item := mediaItem{
			Type:  "photo",
			Media: "attach://" + f.FieldName,
		}
		if i == 0 {
			item.Caption = "📄 Residence permit documents"
		}
		mediaArr = append(mediaArr, item)
	}
	mediaJSON, err := json.Marshal(mediaArr)
	if err != nil {
		return fmt.Errorf("marshal media json: %w", err)
	}

	// Prepare multipart form
	var body bytes.Buffer
	w := multipart.NewWriter(&body)

	// chat_id field
	if err := w.WriteField("chat_id", strconv.FormatInt(chatID, 10)); err != nil {
		return fmt.Errorf("write chat_id: %w", err)
	}
	// media field (JSON)
	if err := w.WriteField("media", string(mediaJSON)); err != nil {
		return fmt.Errorf("write media json: %w", err)
	}

	// Attach files
	for _, f := range files {
		part, err := w.CreateFormFile(f.FieldName, f.FileName)
		if err != nil {
			return fmt.Errorf("create form file %s: %w", f.FieldName, err)
		}
		if _, err := part.Write(f.Data); err != nil {
			return fmt.Errorf("write file %s: %w", f.FieldName, err)
		}
	}

	// Close writer
	if err := w.Close(); err != nil {
		return fmt.Errorf("close multipart writer: %w", err)
	}

	// Send request to Telegram Bot API directly
	token := os.Getenv("TELEGRAM_BOT_TOKEN")
	if token == "" {
		return fmt.Errorf("telegram token missing")
	}
	apiURL := fmt.Sprintf("https://api.telegram.org/bot%s/sendMediaGroup", token)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, &body)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)

	// Parse response to tell if Telegram accepted it
	var tr struct {
		OK          bool            `json:"ok"`
		Description string          `json:"description,omitempty"`
		Result      json.RawMessage `json:"result,omitempty"`
	}
	if err := json.Unmarshal(respBody, &tr); err != nil {
		// include raw body for debugging
		return fmt.Errorf("telegram response parse error: %w — body: %s", err, string(respBody))
	}
	if !tr.OK {
		return fmt.Errorf("telegram sendMediaGroup failed: %s", tr.Description)
	}

	log.Printf("[uploadMediaGroupFromS3] sent album (chat=%d)", chatID)
	return nil
}
