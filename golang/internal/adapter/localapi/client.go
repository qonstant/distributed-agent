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

func NewClient(apiURL string) *Client {
	return &Client{
		apiURL: apiURL,
		httpClient: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
}

func (c *Client) Ask(ctx context.Context, question qa.Question) (qa.DraftResponse, error) {
	body := map[string]string{"query": question.Text}
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
		Answer string `json:"answer"`
		File   string `json:"file"`
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

	return draft, nil
}
