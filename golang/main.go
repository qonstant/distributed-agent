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
	s3Client        *minio.Client
	s3Bucket        string
	presignExpiry   = 15 * time.Minute
	localAPIURL     = "http://127.0.0.1:8080/query" // local query API
	allowedUsername string                          // accepted username (without @), set in main()
)

/* ============================
   MAIN
============================ */

func main() {
	_ = godotenv.Load()
	// allowed username to accept queries from (without @). Default to "chupapimunyao"
	allowedUsername = os.Getenv("ALLOWED_USERNAME")
	if allowedUsername == "" {
		allowedUsername = "chupapimunyao"
	}
	// strip leading @ if present and normalize
	allowedUsername = strings.TrimPrefix(allowedUsername, "@")

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

	// regex dispatcher for those commands (keeps previous behavior)
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

	log.Printf("🤖 Bot started (only answering username: @%s)", allowedUsername)
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

func accessDeniedText() string {
	// instruct user to message the allowed username to request access
	return fmt.Sprintf("This bot accepts requests only with subscription.\n\nTo request access, please message @%s.", allowedUsername)
}

/* ============================
   HANDLERS
   - defaultHandler: now treats message text as a query to local API,
     returns text+document where available
   - replies with an access instruction when sender is not allowed
============================ */

func defaultHandler(ctx context.Context, b *bot.Bot, u *models.Update) {
	if u == nil || u.Message == nil || u.Message.From == nil {
		return
	}

	// get raw sender username (does not include '@')
	senderUsername := u.Message.From.Username
	displayName := username(u)

	// If sender has no username or doesn't match allowed -> send short instruction
	if senderUsername == "" || !strings.EqualFold(senderUsername, allowedUsername) {
		log.Printf("[defaultHandler] access denied for %s: %s", displayName, u.Message.Text)
		_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: u.Message.Chat.ID,
			Text:   accessDeniedText(),
		})
		return
	}

	// ignore commands (they are handled separately)
	if strings.HasPrefix(u.Message.Text, "/") {
		_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: u.Message.Chat.ID,
			Text:   "Unknown command. Try /help.",
		})
		return
	}

	query := u.Message.Text
	log.Printf("[defaultHandler] query from %s: %s", senderUsername, query)

	// Query local API
	resp, err := queryLocalAPI(ctx, query)
	if err != nil {
		log.Printf("[defaultHandler] local API error: %v", err)
		// fallback: send error as message
		_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: u.Message.Chat.ID,
			Text:   fmt.Sprintf("Ошибка обращения к локальному API: %v", err),
		})
		return
	}

	// If no file path was provided, just send the textual answer
	if strings.TrimSpace(resp.File) == "" {
		_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: u.Message.Chat.ID,
			Text:   resp.Answer,
		})
		return
	}

	// Attempt to fetch file bytes (local file, HTTP URL, or S3)
	data, fname, ferr := fetchFileData(ctx, resp.File)
	if ferr != nil {
		// Couldn't fetch file -> send textual answer and a note
		log.Printf("[defaultHandler] fetchFileData failed: %v", ferr)
		_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: u.Message.Chat.ID,
			Text:   fmt.Sprintf("%s\n\n(Файл %s недоступен: %v)", resp.Answer, resp.File, ferr),
		})
		return
	}

	// Send document with caption = answer (Telegram caption limit ~1024)
	if len(resp.Answer) > 1020 {
		// trim slightly to be safe
		resp.Answer = resp.Answer[:1020] + "…"
	}
	if err := sendDocumentFromBytes(ctx, u.Message.Chat.ID, fname, data, resp.Answer); err != nil {
		log.Printf("[defaultHandler] sendDocumentFromBytes failed: %v", err)
		_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: u.Message.Chat.ID,
			Text:   fmt.Sprintf("%s\n\n(Не удалось отправить файл: %v)", resp.Answer, err),
		})
		return
	}
	// success
}

/* ============================
   start/help/randompic Handlers
   - now reply with access instruction for unauthorized users
============================ */

func startHandler(ctx context.Context, b *bot.Bot, u *models.Update) {
	if u == nil || u.Message == nil || u.Message.From == nil {
		return
	}
	if u.Message.From.Username == "" || !strings.EqualFold(u.Message.From.Username, allowedUsername) {
		log.Printf("[startHandler] access denied for %s", username(u))
		_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: u.Message.Chat.ID,
			Text:   accessDeniedText(),
		})
		return
	}

	_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID: u.Message.Chat.ID,
		Text:   "Привет! Отправь любой вопрос (на любом языке) — бот перешлёт запрос локальному API и пришлёт ответ + файл справки, если он есть.",
	})
}

func helpHandler(ctx context.Context, b *bot.Bot, u *models.Update) {
	if u == nil || u.Message == nil || u.Message.From == nil {
		return
	}
	if u.Message.From.Username == "" || !strings.EqualFold(u.Message.From.Username, allowedUsername) {
		log.Printf("[helpHandler] access denied for %s", username(u))
		_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: u.Message.Chat.ID,
			Text:   accessDeniedText(),
		})
		return
	}
	_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID: u.Message.Chat.ID,
		Text:   "Просто отправь текст (например: \"Как податься внж?\") — бот запросит локальный API и вернёт ответ + документ (если API вернул файл).",
	})
}

func randomPicHandler(ctx context.Context, b *bot.Bot, u *models.Update) {
	if u == nil || u.Message == nil || u.Message.From == nil {
		return
	}
	if u.Message.From.Username == "" || !strings.EqualFold(u.Message.From.Username, allowedUsername) {
		log.Printf("[randomPicHandler] access denied for %s", username(u))
		_, _ = b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: u.Message.Chat.ID,
			Text:   accessDeniedText(),
		})
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
   Local API query
============================ */

type QueryResp struct {
	Answer string `json:"answer"`
	File   string `json:"file"`
}

func queryLocalAPI(ctx context.Context, q string) (*QueryResp, error) {
	client := &http.Client{Timeout: 15 * time.Second}
	bodyObj := map[string]string{"query": q}
	js, err := json.Marshal(bodyObj)
	if err != nil {
		return nil, fmt.Errorf("marshal query: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, localAPIURL, bytes.NewReader(js))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("post to local API: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		bb, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("local API returned %d: %s", resp.StatusCode, string(bb))
	}
	var qr QueryResp
	if err := json.NewDecoder(resp.Body).Decode(&qr); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &qr, nil
}

/* ============================
   File fetching logic
   Tries (in order):
   - if file is an absolute http(s) URL -> fetch via HTTP
   - if DOC_ROOT env var is set -> try local file join(DOC_ROOT, file)
   - if s3Client configured -> try to GetObject from S3 bucket with key = file
============================ */

func fetchFileData(ctx context.Context, path string) ([]byte, string, error) {
	trim := strings.TrimSpace(path)
	if trim == "" {
		return nil, "", fmt.Errorf("empty path")
	}
	// If URL
	if strings.HasPrefix(trim, "http://") || strings.HasPrefix(trim, "https://") {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, trim, nil)
		if err != nil {
			return nil, "", err
		}
		client := &http.Client{Timeout: 30 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			return nil, "", err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, "", fmt.Errorf("HTTP %d", resp.StatusCode)
		}
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, "", err
		}
		_, fname := filepath.Split(trim)
		if fname == "" {
			fname = "file"
		}
		return data, fname, nil
	}

	// Try DOC_ROOT local filesystem
	if docRoot := os.Getenv("DOC_ROOT"); docRoot != "" {
		fp := filepath.Join(docRoot, filepath.FromSlash(trim))
		if fi, err := os.Stat(fp); err == nil && !fi.IsDir() {
			data, err := os.ReadFile(fp)
			if err != nil {
				return nil, "", err
			}
			return data, filepath.Base(fp), nil
		}
	}

	// If S3 available, try to get from S3 bucket with key = path
	if s3Client != nil && s3Bucket != "" {
		obj, err := s3Client.GetObject(ctx, s3Bucket, trim, minio.GetObjectOptions{})
		if err != nil {
			return nil, "", fmt.Errorf("s3 GetObject: %w", err)
		}
		defer obj.Close()
		data, err := io.ReadAll(obj)
		if err != nil {
			return nil, "", fmt.Errorf("read s3 object: %w", err)
		}
		return data, filepath.Base(trim), nil
	}

	return nil, "", fmt.Errorf("file not found locally (DOC_ROOT unset or file missing) and S3 not configured")
}

/* ============================
   Sending single document (attach://)
============================ */

func sendDocumentFromBytes(ctx context.Context, chatID int64, filename string, data []byte, caption string) error {
	var body bytes.Buffer
	w := multipart.NewWriter(&body)

	// chat_id
	if err := w.WriteField("chat_id", strconv.FormatInt(chatID, 10)); err != nil {
		return fmt.Errorf("write chat_id: %w", err)
	}
	// caption (optional)
	if caption != "" {
		if err := w.WriteField("caption", caption); err != nil {
			return fmt.Errorf("write caption: %w", err)
		}
	}

	// document field should be named "document" and referenced in form directly as file
	part, err := w.CreateFormFile("document", filename)
	if err != nil {
		return fmt.Errorf("create form file: %w", err)
	}
	if _, err := part.Write(data); err != nil {
		return fmt.Errorf("write file data: %w", err)
	}

	if err := w.Close(); err != nil {
		return fmt.Errorf("close writer: %w", err)
	}

	token := os.Getenv("TELEGRAM_BOT_TOKEN")
	if token == "" {
		return fmt.Errorf("telegram token missing")
	}
	apiURL := fmt.Sprintf("https://api.telegram.org/bot%s/sendDocument", token)
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

	var tr struct {
		OK          bool   `json:"ok"`
		Description string `json:"description,omitempty"`
	}
	if err := json.Unmarshal(respBody, &tr); err != nil {
		return fmt.Errorf("telegram response parse error: %w — body: %s", err, string(respBody))
	}
	if !tr.OK {
		return fmt.Errorf("telegram sendDocument failed: %s", tr.Description)
	}
	return nil
}

/* ============================
   Existing helper you had: uploadMediaGroupFromS3
   (left unchanged except minor comment)
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
