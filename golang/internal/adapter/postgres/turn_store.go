package postgres

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	_ "github.com/lib/pq"

	"github.com/qonstant/distributed-agent/internal/domain/persistence"
)

type TurnStore struct {
	db *sql.DB
}

func NewTurnStore(dbURL string) (*TurnStore, error) {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("open postgres connection: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}

	return &TurnStore{db: db}, nil
}

func (s *TurnStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *TurnStore) SaveTurn(ctx context.Context, event persistence.TurnEvent) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("postgres turn store is not initialized")
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	userID, err := upsertUser(ctx, tx, event.User)
	if err != nil {
		return err
	}

	conversationID, err := ensureConversation(ctx, tx, userID, event.Conversation)
	if err != nil {
		return err
	}

	userMessageID, err := insertMessage(ctx, tx, conversationID, event.UserMessage)
	if err != nil {
		return err
	}

	if event.Classification != nil {
		if err := insertClassification(ctx, tx, userMessageID, *event.Classification); err != nil {
			return err
		}
	}

	if event.AssistantMessage != nil {
		if _, err := insertMessage(ctx, tx, conversationID, *event.AssistantMessage); err != nil {
			return err
		}
	}

	for _, usage := range event.UsageEvents {
		if err := insertUsageEvent(ctx, tx, userID, conversationID, userMessageID, usage); err != nil {
			return err
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit transaction: %w", err)
	}
	return nil
}

func upsertUser(ctx context.Context, tx *sql.Tx, user persistence.User) (int64, error) {
	if user.TelegramID == 0 {
		return 0, fmt.Errorf("missing telegram_id in turn event")
	}

	var id int64
	if err := tx.QueryRowContext(
		ctx,
		`
		INSERT INTO "users" ("telegram_id", "username", "created_at", "updated_at")
		VALUES ($1, NULLIF($2, ''), now(), now())
		ON CONFLICT ("telegram_id") DO UPDATE
		SET "username" = COALESCE(NULLIF(EXCLUDED."username", ''), "users"."username"),
		    "updated_at" = now()
		RETURNING "id"
		`,
		user.TelegramID,
		strings.TrimSpace(user.Username),
	).Scan(&id); err != nil {
		return 0, fmt.Errorf("upsert user %d: %w", user.TelegramID, err)
	}

	return id, nil
}

func ensureConversation(ctx context.Context, tx *sql.Tx, userID int64, conversation persistence.Conversation) (int64, error) {
	key := strings.TrimSpace(conversation.Key)
	if key == "" {
		return 0, fmt.Errorf("missing conversation key in turn event")
	}

	createdAt := conversation.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}
	updatedAt := conversation.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}

	var id int64
	if err := tx.QueryRowContext(
		ctx,
		`
		INSERT INTO "conversations" ("user_id", "conversation_key", "summary", "created_at", "updated_at")
		VALUES ($1, $2, NULL, $3, $4)
		ON CONFLICT ("conversation_key") DO UPDATE
		SET "updated_at" = GREATEST("conversations"."updated_at", EXCLUDED."updated_at")
		RETURNING "id"
		`,
		userID,
		key,
		createdAt,
		updatedAt,
	).Scan(&id); err != nil {
		return 0, fmt.Errorf("ensure conversation %q: %w", key, err)
	}

	return id, nil
}

func insertMessage(
	ctx context.Context,
	tx *sql.Tx,
	conversationID int64,
	message persistence.Message,
) (int64, error) {
	createdAt := message.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}

	var id int64
	if err := tx.QueryRowContext(
		ctx,
		`
		INSERT INTO "messages" ("conversation_id", "is_assistant", "message_text", "created_at")
		VALUES ($1, $2, $3, $4)
		RETURNING "id"
		`,
		conversationID,
		message.IsAssistant,
		message.Text,
		createdAt,
	).Scan(&id); err != nil {
		return 0, fmt.Errorf("insert message: %w", err)
	}

	return id, nil
}

func insertClassification(ctx context.Context, tx *sql.Tx, messageID int64, classification persistence.MessageClassification) error {
	intent := strings.TrimSpace(classification.Intent)
	if intent == "" {
		intent = "OTHER"
	}
	createdAt := classification.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}

	if _, err := tx.ExecContext(
		ctx,
		`
		INSERT INTO "message_classifications" (
			"message_id",
			"intent",
			"explanation",
			"detected_language",
			"classifier_model",
			"classifier_version",
			"created_at"
		)
		VALUES ($1, $2, NULLIF($3, ''), NULLIF($4, ''), NULLIF($5, ''), NULLIF($6, ''), $7)
		ON CONFLICT ("message_id") DO UPDATE
		SET "intent" = EXCLUDED."intent",
		    "explanation" = EXCLUDED."explanation",
		    "detected_language" = EXCLUDED."detected_language",
		    "classifier_model" = EXCLUDED."classifier_model",
		    "classifier_version" = EXCLUDED."classifier_version"
		`,
		messageID,
		intent,
		strings.TrimSpace(classification.Explanation),
		strings.TrimSpace(classification.DetectedLanguage),
		strings.TrimSpace(classification.ClassifierModel),
		strings.TrimSpace(classification.ClassifierVersion),
		createdAt,
	); err != nil {
		return fmt.Errorf("insert classification: %w", err)
	}

	return nil
}

func insertUsageEvent(
	ctx context.Context,
	tx *sql.Tx,
	userID int64,
	conversationID int64,
	messageID int64,
	usage persistence.UsageEvent,
) error {
	eventType := strings.TrimSpace(usage.EventType)
	if eventType == "" {
		eventType = "other"
	}
	createdAt := usage.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}

	if _, err := tx.ExecContext(
		ctx,
		`
		INSERT INTO "usage_events" (
			"user_id",
			"conversation_id",
			"message_id",
			"event_type",
			"input_tokens",
			"output_tokens",
			"estimated_cost",
			"created_at"
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		`,
		userID,
		conversationID,
		messageID,
		eventType,
		usage.InputTokens,
		usage.OutputTokens,
		usage.EstimatedCost,
		createdAt,
	); err != nil {
		return fmt.Errorf("insert usage event %q: %w", eventType, err)
	}

	return nil
}
