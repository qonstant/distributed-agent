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
	apiURL     string
	httpClient *http.Client
}

type queryRequest struct {
	Query          string `json:"query"`
	ConversationID string `json:"conversation_id,omitempty"`
}

func NewClient(apiURL string) *Client {
	return &Client{
		apiURL: apiURL,
		httpClient: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
}

func (c *Client) Ask(ctx context.Context, question qa.Question) (qa.DraftResponse, error) {
	return c.ask(ctx, question, "")
}

func (c *Client) AskWithConversation(ctx context.Context, question qa.Question, conversationID string) (qa.DraftResponse, error) {
	return c.ask(ctx, question, conversationID)
}

func (c *Client) ask(ctx context.Context, question qa.Question, conversationID string) (qa.DraftResponse, error) {
	body := queryRequest{
		Query:          question.Text,
		ConversationID: strings.TrimSpace(conversationID),
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return qa.DraftResponse{}, fmt.Errorf("marshal query: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.apiURL, bytes.NewReader(payload))
	if err != nil {
		return qa.DraftResponse{}, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return qa.DraftResponse{}, fmt.Errorf("post to local API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return qa.DraftResponse{}, fmt.Errorf("local API returned %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Answer         string `json:"answer"`
		File           string `json:"file"`
		Classification *struct {
			Intent   string `json:"intent"`
			Explain  string `json:"explain"`
			Language string `json:"language"`
			Model    string `json:"model"`
			Version  string `json:"version"`
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
		return qa.DraftResponse{}, fmt.Errorf("decode response: %w", err)
	}

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
