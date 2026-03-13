package localapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

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

			var body map[string]string
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("Decode() error = %v", err)
			}
			if got, want := body["query"], "hello"; got != want {
				t.Fatalf("query = %q, want %q", got, want)
			}

			payload := `{"answer":"world","file":"docs/file.pdf"}`
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

		client := NewClient("http://local-api.test/query")
		client.httpClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
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
