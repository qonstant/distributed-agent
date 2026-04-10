package telegram

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"time"
	"unicode/utf8"

	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type Sender struct {
	token      string
	httpClient *http.Client
}

func NewSender(token string) *Sender {
	return &Sender{
		token: token,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

func (s *Sender) SendDocument(ctx context.Context, chatID int64, attachment qa.Attachment, caption string) error {
	caption = sanitizeUTF8(caption)
	if utf8.RuneCountInString(caption) > 1020 {
		caption = trimRunes(caption, 1020) + "..."
	}

	filename := safeFilename(attachment.Name)

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	if err := writer.WriteField("chat_id", strconv.FormatInt(chatID, 10)); err != nil {
		return fmt.Errorf("write chat_id: %w", err)
	}
	if caption != "" {
		if err := writer.WriteField("caption", caption); err != nil {
			return fmt.Errorf("write caption: %w", err)
		}
	}

	part, err := writer.CreateFormFile("document", filename)
	if err != nil {
		return fmt.Errorf("create form file: %w", err)
	}
	if _, err := part.Write(attachment.Content); err != nil {
		return fmt.Errorf("write file data: %w", err)
	}

	if err := writer.Close(); err != nil {
		return fmt.Errorf("close writer: %w", err)
	}

	return s.doTelegramRequest(ctx, "sendDocument", &body, writer.FormDataContentType())
}

func (s *Sender) SendMediaGroup(ctx context.Context, chatID int64, attachments []qa.Attachment, caption string) error {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	if err := writer.WriteField("chat_id", strconv.FormatInt(chatID, 10)); err != nil {
		return fmt.Errorf("write chat_id: %w", err)
	}

	type mediaItem struct {
		Type    string `json:"type"`
		Media   string `json:"media"`
		Caption string `json:"caption,omitempty"`
	}

	media := make([]mediaItem, 0, len(attachments))
	for index, attachment := range attachments {
		fieldName := fmt.Sprintf("file%d", index)
		item := mediaItem{
			Type:  "photo",
			Media: "attach://" + fieldName,
		}
		if index == 0 && caption != "" {
			item.Caption = sanitizeUTF8(caption)
			if utf8.RuneCountInString(item.Caption) > 1020 {
				item.Caption = trimRunes(item.Caption, 1020) + "..."
			}
		}
		media = append(media, item)

		part, err := writer.CreateFormFile(fieldName, safeFilename(attachment.Name))
		if err != nil {
			return fmt.Errorf("create form file %s: %w", fieldName, err)
		}
		if _, err := part.Write(attachment.Content); err != nil {
			return fmt.Errorf("write file %s: %w", fieldName, err)
		}
	}

	mediaJSON, err := json.Marshal(media)
	if err != nil {
		return fmt.Errorf("marshal media json: %w", err)
	}

	if err := writer.WriteField("media", string(mediaJSON)); err != nil {
		return fmt.Errorf("write media json: %w", err)
	}

	if err := writer.Close(); err != nil {
		return fmt.Errorf("close multipart writer: %w", err)
	}

	return s.doTelegramRequest(ctx, "sendMediaGroup", &body, writer.FormDataContentType())
}

func (s *Sender) doTelegramRequest(ctx context.Context, method string, body io.Reader, contentType string) error {
	apiURL := fmt.Sprintf("https://api.telegram.org/bot%s/%s", s.token, method)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, body)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", contentType)

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	var telegramResp struct {
		OK          bool   `json:"ok"`
		Description string `json:"description,omitempty"`
	}

	if err := json.Unmarshal(respBody, &telegramResp); err != nil {
		return fmt.Errorf("telegram response parse error: %w - body: %s", err, string(respBody))
	}
	if !telegramResp.OK {
		return fmt.Errorf("telegram %s failed: %s", method, telegramResp.Description)
	}

	return nil
}
