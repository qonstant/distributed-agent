package storage

import (
	"context"
	"fmt"
	"io"
	"path/filepath"

	minio "github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/qonstant/distributed-agent/internal/config"
)

type S3Store struct {
	client *minio.Client
	bucket string
}

func NewS3Store(ctx context.Context, cfg config.S3Config) (*S3Store, error) {
	if cfg.Endpoint == "" || cfg.AccessKeyID == "" || cfg.SecretAccessKey == "" || cfg.Bucket == "" {
		return nil, nil
	}

	client, err := minio.New(cfg.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.AccessKeyID, cfg.SecretAccessKey, ""),
		Secure: cfg.UseSSL,
	})
	if err != nil {
		return nil, fmt.Errorf("minio.New: %w", err)
	}

	ok, err := client.BucketExists(ctx, cfg.Bucket)
	if err != nil || !ok {
		return nil, fmt.Errorf("bucket %q not accessible: %w", cfg.Bucket, err)
	}

	return &S3Store{
		client: client,
		bucket: cfg.Bucket,
	}, nil
}

func (s *S3Store) GetObject(ctx context.Context, key string) ([]byte, string, error) {
	object, err := s.client.GetObject(ctx, s.bucket, key, minio.GetObjectOptions{})
	if err != nil {
		return nil, "", fmt.Errorf("s3 GetObject: %w", err)
	}
	defer object.Close()

	data, err := io.ReadAll(object)
	if err != nil {
		return nil, "", fmt.Errorf("read s3 object: %w", err)
	}

	name := filepath.Base(key)
	if name == "" {
		name = "file"
	}
	return data, name, nil
}
