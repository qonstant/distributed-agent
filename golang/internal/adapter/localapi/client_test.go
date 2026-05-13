package localapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestClientAsk(t *testing.T) {
	t.Parallel()

	t.Run("maps successful response with attachment", func(t *testing.T) {
		t.Parallel()

		client := NewClient("http://local-api.test/query")
		client.httpClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if got, want := r.Method, http.MethodPost; got != want {
				t.Fatalf("method = %s, want %s", got, want)
			}
			if got, want := r.Header.Get("Content-Type"), "application/json"; got != want {
				t.Fatalf("Content-Type = %q, want %q", got, want)
			}

			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("Decode() error = %v", err)
			}
			if got, want := body["query"], "hello"; got != want {
				t.Fatalf("query = %q, want %q", got, want)
			}
			if _, ok := body["conversation_id"]; ok {
				t.Fatal("conversation_id should be omitted for Ask()")
			}
			if _, ok := body["preferred_name"]; ok {
				t.Fatal("preferred_name should be omitted for Ask()")
			}

			payload := `{"answer":"world","file":"docs/file.pdf","classification":{"intent":"DOCUMENT_REQUEST","explain":"user asked for a file","language":"en","model":"gpt-4o-mini","version":"v1","profile_action":"","preferred_name":""},"usage_events":[{"event_type":"classification","input_tokens":0,"output_tokens":0,"total_tokens":0,"estimated_cost":0},{"event_type":"chat_completion","input_tokens":0,"output_tokens":0,"total_tokens":0,"estimated_cost":0}]}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(payload)),
				Header:     make(http.Header),
			}, nil
		})}
		response, err := client.Ask(context.Background(), qa.Question{Text: "hello"})
		if err != nil {
			t.Fatalf("Ask() error = %v", err)
		}
		if got, want := response.Text, "world"; got != want {
			t.Fatalf("response.Text = %q, want %q", got, want)
		}
		if len(response.AttachmentRefs) != 1 {
			t.Fatalf("len(response.AttachmentRefs) = %d, want 1", len(response.AttachmentRefs))
		}
		if got, want := response.AttachmentRefs[0].Source, "docs/file.pdf"; got != want {
			t.Fatalf("response.AttachmentRefs[0].Source = %q, want %q", got, want)
		}
		if got, want := response.AttachmentRefs[0].Kind, qa.AttachmentDocument; got != want {
			t.Fatalf("response.AttachmentRefs[0].Kind = %q, want %q", got, want)
		}
		if response.Classification == nil {
			t.Fatal("response.Classification = nil, want non-nil")
		}
		if got, want := response.Classification.Intent, "DOCUMENT_REQUEST"; got != want {
			t.Fatalf("response.Classification.Intent = %q, want %q", got, want)
		}
		if got, want := response.Classification.ProfileAction, ""; got != want {
			t.Fatalf("response.Classification.ProfileAction = %q, want %q", got, want)
		}
		if len(response.UsageEvents) != 2 {
			t.Fatalf("len(response.UsageEvents) = %d, want 2", len(response.UsageEvents))
		}
		if got, want := response.UsageEvents[1].EventType, "chat_completion"; got != want {
			t.Fatalf("response.UsageEvents[1].EventType = %q, want %q", got, want)
		}
	})

	t.Run("sends conversation id when available", func(t *testing.T) {
		t.Parallel()

		client := NewClient("http://local-api.test/query")
		client.httpClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("Decode() error = %v", err)
			}
			if got, want := body["query"], "hello"; got != want {
				t.Fatalf("query = %q, want %q", got, want)
			}
			if got, want := body["conversation_id"], "conv-1"; got != want {
				t.Fatalf("conversation_id = %q, want %q", got, want)
			}
			if got, want := body["preferred_name"], "Stored Name"; got != want {
				t.Fatalf("preferred_name = %q, want %q", got, want)
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(`{"answer":"world","file":""}`)),
				Header:     make(http.Header),
			}, nil
		})}

		response, err := client.AskWithConversation(context.Background(), qa.Question{Text: "hello"}, "conv-1", "Stored Name")
		if err != nil {
			t.Fatalf("AskWithConversation() error = %v", err)
		}
		if got, want := response.Text, "world"; got != want {
			t.Fatalf("response.Text = %q, want %q", got, want)
		}
	})

	t.Run("omits empty attachment", func(t *testing.T) {
		t.Parallel()

		client := NewClient("http://local-api.test/query")
		client.httpClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(`{"answer":"world","file":"   "}`)),
				Header:     make(http.Header),
			}, nil
		})}
		response, err := client.Ask(context.Background(), qa.Question{Text: "hello"})
		if err != nil {
			t.Fatalf("Ask() error = %v", err)
		}
		if len(response.AttachmentRefs) != 0 {
			t.Fatalf("len(response.AttachmentRefs) = %d, want 0", len(response.AttachmentRefs))
		}
	})

	t.Run("returns non-200 error", func(t *testing.T) {
		t.Parallel()

		attempts := 0
		client := NewClient("http://local-api.test/query")
		client.httpClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			attempts++
			return &http.Response{
				StatusCode: http.StatusBadRequest,
				Body:       io.NopCloser(strings.NewReader("bad request\n")),
				Header:     make(http.Header),
			}, nil
		})}
		_, err := client.Ask(context.Background(), qa.Question{Text: "hello"})
		if err == nil {
			t.Fatal("Ask() error = nil, want non-nil")
		}
		if got, want := attempts, 1; got != want {
			t.Fatalf("attempts = %d, want %d", got, want)
		}
	})

	t.Run("retries transient server errors", func(t *testing.T) {
		t.Parallel()

		attempts := 0
		client := NewClientWithConfig("http://local-api.test/query", ClientConfig{
			MaxAttempts:  2,
			RetryBackoff: time.Nanosecond,
		})
		client.httpClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			attempts++
			if attempts == 1 {
				return &http.Response{
					StatusCode: http.StatusServiceUnavailable,
					Body:       io.NopCloser(strings.NewReader("not ready\n")),
					Header:     make(http.Header),
				}, nil
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(`{"answer":"world","file":""}`)),
				Header:     make(http.Header),
			}, nil
		})}

		response, err := client.Ask(context.Background(), qa.Question{Text: "hello"})
		if err != nil {
			t.Fatalf("Ask() error = %v", err)
		}
		if got, want := response.Text, "world"; got != want {
			t.Fatalf("response.Text = %q, want %q", got, want)
		}
		if got, want := attempts, 2; got != want {
			t.Fatalf("attempts = %d, want %d", got, want)
		}
	})

	t.Run("returns decode error for invalid json", func(t *testing.T) {
		t.Parallel()

		client := NewClient("http://local-api.test/query")
		client.httpClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader("{invalid")),
				Header:     make(http.Header),
			}, nil
		})}
		_, err := client.Ask(context.Background(), qa.Question{Text: "hello"})
		if err == nil {
			t.Fatal("Ask() error = nil, want non-nil")
		}
	})
}

func TestNewClientWithConfig(t *testing.T) {
	t.Parallel()

	client := NewClientWithConfig("http://local-api.test/query", ClientConfig{
		Timeout:      42 * time.Second,
		MaxAttempts:  3,
		RetryBackoff: 2 * time.Second,
	})

	if got, want := client.httpClient.Timeout, 42*time.Second; got != want {
		t.Fatalf("httpClient.Timeout = %v, want %v", got, want)
	}
	if got, want := client.maxAttempts, 3; got != want {
		t.Fatalf("maxAttempts = %d, want %d", got, want)
	}
	if got, want := client.retryBackoff, 2*time.Second; got != want {
		t.Fatalf("retryBackoff = %v, want %v", got, want)
	}
}

func TestClientAskWithCancelledContext(t *testing.T) {
	t.Parallel()

	client := NewClient("http://local-api.test/query")
	client.httpClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		<-r.Context().Done()
		return nil, r.Context().Err()
	})}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := client.Ask(ctx, qa.Question{Text: "hello"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Ask() error = %v, want wrapped context.Canceled", err)
	}
}
