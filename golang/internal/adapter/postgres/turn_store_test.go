package postgres

import (
	"context"
	"errors"
	"regexp"
	"testing"
	"time"

	sqlmock "github.com/DATA-DOG/go-sqlmock"

	"github.com/qonstant/distributed-agent/internal/domain/persistence"
)

func TestTurnStoreSaveTurn(t *testing.T) {
	t.Parallel()

	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock.New() error = %v", err)
	}
	defer db.Close()

	store := &TurnStore{db: db}
	now := time.Unix(1774920000, 0).UTC()
	event := persistence.TurnEvent{
		Version: 1,
		User: persistence.User{
			TelegramID:  1111111111,
			Username:    "test_user",
			DisplayName: "Test User",
		},
		Conversation: persistence.Conversation{
			Key:       "conv-test-1774920000000000000",
			CreatedAt: now,
			UpdatedAt: now,
		},
		UserMessage: persistence.Message{
			IsAssistant: false,
			Text:        "What is the test code?",
			CreatedAt:   now,
		},
		AssistantMessage: &persistence.Message{
			IsAssistant: true,
			Text:        "The test code is 1234.",
			CreatedAt:   now,
		},
		Classification: &persistence.MessageClassification{
			Intent:            "FACTUAL_QUESTION",
			Explanation:       "user asks about prior context",
			DetectedLanguage:  "en",
			ClassifierModel:   "gpt-4o-mini",
			ClassifierVersion: "v1",
			CreatedAt:         now,
		},
		UsageEvents: []persistence.UsageEvent{
			{EventType: "classification", CreatedAt: now},
			{EventType: "chat_completion", CreatedAt: now},
		},
	}

	mock.ExpectBegin()
	mock.ExpectQuery(regexp.QuoteMeta(`
		INSERT INTO "users" ("telegram_id", "username", "created_at", "updated_at")
		VALUES ($1, NULLIF($2, ''), now(), now())
		ON CONFLICT ("telegram_id") DO UPDATE
		SET "username" = COALESCE(NULLIF(EXCLUDED."username", ''), "users"."username"),
		    "updated_at" = now()
		RETURNING "id"
	`)).
		WithArgs(int64(1111111111), "test_user").
		WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(int64(7)))

	mock.ExpectQuery(regexp.QuoteMeta(`
		INSERT INTO "conversations" ("user_id", "conversation_key", "summary", "created_at", "updated_at")
		VALUES ($1, $2, NULL, $3, $4)
		ON CONFLICT ("conversation_key") DO UPDATE
		SET "updated_at" = GREATEST("conversations"."updated_at", EXCLUDED."updated_at")
		RETURNING "id"
	`)).
		WithArgs(int64(7), event.Conversation.Key, now, now).
		WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(int64(11)))

	mock.ExpectQuery(regexp.QuoteMeta(`
		INSERT INTO "messages" ("conversation_id", "is_assistant", "message_text", "created_at")
		VALUES ($1, $2, $3, $4)
		RETURNING "id"
	`)).
		WithArgs(int64(11), false, "What is the test code?", now).
		WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(int64(101)))

	mock.ExpectExec(regexp.QuoteMeta(`
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
	`)).
		WithArgs(int64(101), "FACTUAL_QUESTION", "user asks about prior context", "en", "gpt-4o-mini", "v1", now).
		WillReturnResult(sqlmock.NewResult(1, 1))

	mock.ExpectQuery(regexp.QuoteMeta(`
		INSERT INTO "messages" ("conversation_id", "is_assistant", "message_text", "created_at")
		VALUES ($1, $2, $3, $4)
		RETURNING "id"
	`)).
		WithArgs(int64(11), true, "The test code is 1234.", now).
		WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(int64(102)))

	mock.ExpectExec(regexp.QuoteMeta(`
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
	`)).
		WithArgs(int64(7), int64(11), int64(101), "classification", 0, 0, 0.0, now).
		WillReturnResult(sqlmock.NewResult(1, 1))

	mock.ExpectExec(regexp.QuoteMeta(`
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
	`)).
		WithArgs(int64(7), int64(11), int64(101), "chat_completion", 0, 0, 0.0, now).
		WillReturnResult(sqlmock.NewResult(1, 1))

	mock.ExpectCommit()

	if err := store.SaveTurn(context.Background(), event); err != nil {
		t.Fatalf("SaveTurn() error = %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("ExpectationsWereMet() error = %v", err)
	}
}

func TestTurnStoreSaveTurnRollsBackOnUserUpsertError(t *testing.T) {
	t.Parallel()

	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock.New() error = %v", err)
	}
	defer db.Close()

	store := &TurnStore{db: db}
	wantErr := errors.New("insert failed")

	mock.ExpectBegin()
	mock.ExpectQuery(regexp.QuoteMeta(`
		INSERT INTO "users" ("telegram_id", "username", "created_at", "updated_at")
		VALUES ($1, NULLIF($2, ''), now(), now())
		ON CONFLICT ("telegram_id") DO UPDATE
		SET "username" = COALESCE(NULLIF(EXCLUDED."username", ''), "users"."username"),
		    "updated_at" = now()
		RETURNING "id"
	`)).
		WithArgs(int64(1111111111), "test_user").
		WillReturnError(wantErr)
	mock.ExpectRollback()

	err = store.SaveTurn(context.Background(), persistence.TurnEvent{
		User: persistence.User{TelegramID: 1111111111, Username: "test_user"},
		Conversation: persistence.Conversation{
			Key:       "conv-test-1",
			CreatedAt: time.Unix(1774920000, 0).UTC(),
			UpdatedAt: time.Unix(1774920000, 0).UTC(),
		},
		UserMessage: persistence.Message{Text: "hello"},
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("SaveTurn() error = %v, want wrapped %v", err, wantErr)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("ExpectationsWereMet() error = %v", err)
	}
}
