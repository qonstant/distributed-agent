package storage

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type Resolver struct {
	docRoot    string
	s3         *S3Store
	httpClient *http.Client
}

func NewResolver(docRoot string, s3 *S3Store) *Resolver {
	return &Resolver{
		docRoot: strings.TrimSpace(docRoot),
		s3:      s3,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (r *Resolver) Resolve(ctx context.Context, refs []qa.AttachmentRef) ([]qa.Attachment, error) {
	attachments := make([]qa.Attachment, 0, len(refs))
	for _, ref := range refs {
		attachment, err := r.resolveOne(ctx, ref)
		if err != nil {
			return nil, err
		}
		attachments = append(attachments, attachment)
	}
	return attachments, nil
}

func (r *Resolver) resolveOne(ctx context.Context, ref qa.AttachmentRef) (qa.Attachment, error) {
	source := strings.TrimSpace(ref.Source)
	if source == "" {
		return qa.Attachment{}, fmt.Errorf("empty attachment source")
	}

	data, name, err := r.fetchBytes(ctx, source)
	if err != nil {
		return qa.Attachment{}, err
	}

	kind := ref.Kind
	if kind == "" {
		kind = qa.AttachmentDocument
	}

	return qa.Attachment{
		Name:    name,
		Kind:    kind,
		Content: data,
	}, nil
}

func (r *Resolver) fetchBytes(ctx context.Context, source string) ([]byte, string, error) {
	if strings.HasPrefix(source, "http://") || strings.HasPrefix(source, "https://") {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, source, nil)
		if err != nil {
			return nil, "", err
		}

		resp, err := r.httpClient.Do(req)
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

		name := filepath.Base(source)
		if name == "" {
			name = "file"
		}
		return data, name, nil
	}

	if r.docRoot != "" {
		localPath := filepath.Join(r.docRoot, filepath.FromSlash(source))
		if fileInfo, err := os.Stat(localPath); err == nil && !fileInfo.IsDir() {
			data, err := os.ReadFile(localPath)
			if err != nil {
				return nil, "", err
			}
			return data, filepath.Base(localPath), nil
		}
	}

	if r.s3 != nil {
		return r.s3.GetObject(ctx, source)
	}

	return nil, "", fmt.Errorf("file not found locally (DOC_ROOT unset or file missing) and S3 not configured")
}
