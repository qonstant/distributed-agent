package localapi

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type Client struct {
	apiURL       string
	httpClient   *http.Client
	maxAttempts  int
	retryBackoff time.Duration
}

type queryRequest struct {
	Query          string `json:"query"`
	ConversationID string `json:"conversation_id,omitempty"`
	PreferredName  string `json:"preferred_name,omitempty"`
}

type ClientConfig struct {
	Timeout      time.Duration
	MaxAttempts  int
	RetryBackoff time.Duration
}

func NewClient(apiURL string) *Client {
	return NewClientWithConfig(apiURL, ClientConfig{})
}

func NewClientWithConfig(apiURL string, cfg ClientConfig) *Client {
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	maxAttempts := cfg.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 2
	}
	retryBackoff := cfg.RetryBackoff
	if retryBackoff <= 0 {
		retryBackoff = 750 * time.Millisecond
	}

	return &Client{
		apiURL:       apiURL,
		maxAttempts:  maxAttempts,
		retryBackoff: retryBackoff,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func (c *Client) Ask(ctx context.Context, question qa.Question) (qa.DraftResponse, error) {
	return c.ask(ctx, question, "", "")
}

func (c *Client) AskWithConversation(ctx context.Context, question qa.Question, conversationID, preferredName string) (qa.DraftResponse, error) {
	return c.ask(ctx, question, conversationID, preferredName)
}

func (c *Client) ask(ctx context.Context, question qa.Question, conversationID, preferredName string) (qa.DraftResponse, error) {
	body := queryRequest{
		Query:          question.Text,
		ConversationID: strings.TrimSpace(conversationID),
		PreferredName:  strings.TrimSpace(preferredName),
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return qa.DraftResponse{}, fmt.Errorf("marshal query: %w", err)
	}

	attempts := c.maxAttempts
	if attempts <= 0 {
		attempts = 1
	}

	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.apiURL, bytes.NewReader(payload))
		if err != nil {
			return qa.DraftResponse{}, fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.httpClient.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return qa.DraftResponse{}, fmt.Errorf("post to local API: %w", err)
			}
			lastErr = fmt.Errorf("post to local API: %w", err)
			if attempt < attempts && c.waitBeforeRetry(ctx) {
				continue
			}
			return qa.DraftResponse{}, lastErr
		}

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("local API returned %d: %s", resp.StatusCode, string(body))
			if retryableStatus(resp.StatusCode) && attempt < attempts && c.waitBeforeRetry(ctx) {
				continue
			}
			return qa.DraftResponse{}, lastErr
		}

		var result struct {
			Answer         string `json:"answer"`
			File           string `json:"file"`
			Classification *struct {
				Intent        string `json:"intent"`
				Explain       string `json:"explain"`
				Language      string `json:"language"`
				Model         string `json:"model"`
				Version       string `json:"version"`
				ProfileAction string `json:"profile_action"`
				PreferredName string `json:"preferred_name"`
			} `json:"classification"`
			UsageEvents []struct {
				EventType     string  `json:"event_type"`
				InputTokens   int     `json:"input_tokens"`
				OutputTokens  int     `json:"output_tokens"`
				TotalTokens   int     `json:"total_tokens"`
				EstimatedCost float64 `json:"estimated_cost"`
			} `json:"usage_events"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			_ = resp.Body.Close()
			return qa.DraftResponse{}, fmt.Errorf("decode response: %w", err)
		}
		_ = resp.Body.Close()

		draft := qa.DraftResponse{Text: result.Answer}
		if file := strings.TrimSpace(result.File); file != "" {
			draft.AttachmentRefs = append(draft.AttachmentRefs, qa.AttachmentRef{
				Source: file,
				Kind:   qa.AttachmentDocument,
			})
		}
		if result.Classification != nil {
			draft.Classification = &qa.MessageClassification{
				Intent:            strings.TrimSpace(result.Classification.Intent),
				Explain:           strings.TrimSpace(result.Classification.Explain),
				DetectedLanguage:  strings.TrimSpace(result.Classification.Language),
				ClassifierModel:   strings.TrimSpace(result.Classification.Model),
				ClassifierVersion: strings.TrimSpace(result.Classification.Version),
				ProfileAction:     strings.TrimSpace(result.Classification.ProfileAction),
				PreferredName:     strings.TrimSpace(result.Classification.PreferredName),
			}
		}
		for _, item := range result.UsageEvents {
			draft.UsageEvents = append(draft.UsageEvents, qa.UsageEvent{
				EventType:     strings.TrimSpace(item.EventType),
				InputTokens:   item.InputTokens,
				OutputTokens:  item.OutputTokens,
				TotalTokens:   item.TotalTokens,
				EstimatedCost: item.EstimatedCost,
			})
		}

		return draft, nil
	}

	if lastErr != nil {
		return qa.DraftResponse{}, lastErr
	}
	return qa.DraftResponse{}, fmt.Errorf("local API request failed")
}

func (c *Client) waitBeforeRetry(ctx context.Context) bool {
	if c.retryBackoff <= 0 {
		return true
	}

	timer := time.NewTimer(c.retryBackoff)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

func retryableStatus(statusCode int) bool {
	return statusCode == http.StatusInternalServerError ||
		statusCode == http.StatusBadGateway ||
		statusCode == http.StatusServiceUnavailable ||
		statusCode == http.StatusGatewayTimeout
}
