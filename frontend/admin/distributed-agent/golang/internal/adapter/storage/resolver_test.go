package storage

import (
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestResolverResolveFromLocalFile(t *testing.T) {
	t.Parallel()

	docRoot := t.TempDir()
	filePath := filepath.Join(docRoot, "docs", "guide.pdf")
	if err := os.MkdirAll(filepath.Dir(filePath), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if err := os.WriteFile(filePath, []byte("pdf-data"), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	resolver := NewResolver(docRoot, nil)
	attachments, err := resolver.Resolve(context.Background(), []qa.AttachmentRef{
		{Source: "docs/guide.pdf", Kind: qa.AttachmentDocument},
	})
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if len(attachments) != 1 {
		t.Fatalf("len(attachments) = %d, want 1", len(attachments))
	}
	if got, want := attachments[0].Name, "guide.pdf"; got != want {
		t.Fatalf("attachments[0].Name = %q, want %q", got, want)
	}
	if got, want := string(attachments[0].Content), "pdf-data"; got != want {
		t.Fatalf("attachments[0].Content = %q, want %q", got, want)
	}
}

func TestResolverResolveFromHTTP(t *testing.T) {
	t.Parallel()

	resolver := NewResolver("", nil)
	resolver.httpClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if got, want := r.Method, http.MethodGet; got != want {
			t.Fatalf("method = %s, want %s", got, want)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("image-bytes")),
			Header:     make(http.Header),
		}, nil
	})}
	attachments, err := resolver.Resolve(context.Background(), []qa.AttachmentRef{
		{Source: "https://example.test/photo.png", Kind: qa.AttachmentPhoto},
	})
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if len(attachments) != 1 {
		t.Fatalf("len(attachments) = %d, want 1", len(attachments))
	}
	if got, want := attachments[0].Kind, qa.AttachmentPhoto; got != want {
		t.Fatalf("attachments[0].Kind = %q, want %q", got, want)
	}
}

func TestResolverDefaultsKindToDocument(t *testing.T) {
	t.Parallel()

	docRoot := t.TempDir()
	filePath := filepath.Join(docRoot, "file.txt")
	if err := os.WriteFile(filePath, []byte("hello"), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	resolver := NewResolver(docRoot, nil)
	attachments, err := resolver.Resolve(context.Background(), []qa.AttachmentRef{{Source: "file.txt"}})
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if got, want := attachments[0].Kind, qa.AttachmentDocument; got != want {
		t.Fatalf("attachments[0].Kind = %q, want %q", got, want)
	}
}

func TestResolverReturnsErrorForEmptySource(t *testing.T) {
	t.Parallel()

	resolver := NewResolver("", nil)
	_, err := resolver.Resolve(context.Background(), []qa.AttachmentRef{{Source: "   "}})
	if err == nil {
		t.Fatal("Resolve() error = nil, want non-nil")
	}
}

func TestResolverReturnsErrorWhenSourceMissing(t *testing.T) {
	t.Parallel()

	resolver := NewResolver("", nil)
	_, err := resolver.Resolve(context.Background(), []qa.AttachmentRef{{Source: "missing.pdf"}})
	if err == nil {
		t.Fatal("Resolve() error = nil, want non-nil")
	}
}
