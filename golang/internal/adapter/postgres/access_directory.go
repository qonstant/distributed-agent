package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	_ "github.com/lib/pq"
	"github.com/qonstant/distributed-agent/internal/domain/access"
)

type AccessDirectory struct {
	db *sql.DB
}

func NewAccessDirectory(dbURL string) (*AccessDirectory, error) {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("open postgres connection: %w", err)
	}
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}
	return &AccessDirectory{db: db}, nil
}

func (d *AccessDirectory) Close() error {
	if d == nil || d.db == nil {
		return nil
	}
	return d.db.Close()
}

func (d *AccessDirectory) FindByTelegramID(ctx context.Context, telegramID int64) (access.Record, bool, error) {
	const query = `
		SELECT telegram_id, username, is_blocked, access_expires_at
		FROM users
		WHERE telegram_id = $1
		LIMIT 1
	`

	var record access.Record
	var username sql.NullString
	var accessExpiresAt sql.NullTime
	err := d.db.QueryRowContext(ctx, query, telegramID).Scan(
		&record.TelegramID,
		&username,
		&record.IsBlocked,
		&accessExpiresAt,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return access.Record{}, false, nil
	}
	if err != nil {
		return access.Record{}, false, fmt.Errorf("query users access by telegram_id: %w", err)
	}
	if username.Valid {
		record.Username = username.String
	}
	if accessExpiresAt.Valid {
		expiresAt := accessExpiresAt.Time
		record.AccessExpiresAt = &expiresAt
	}

	return record, true, nil
}
