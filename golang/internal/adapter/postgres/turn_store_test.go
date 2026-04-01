package postgres

import (
	"context"
	"errors"
	"regexp"
	"testing"
	"time"

	sqlmock "github.com/DATA-DOG/go-sqlmock"

	"github.com/qonstant/distributed-agent/internal/domain/persistence"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
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
			Role:         qa.ConversationRoleUser,
			Text:         "What is the test code?",
			LanguageCode: "en",
			CreatedAt:    now,
		},
		AssistantMessage: persistence.Message{
			Role:         qa.ConversationRoleAssistant,
			Text:         "The test code is ALPHA-123.",
			LanguageCode: "en",
			CreatedAt:    now,
		},
		Classification: &persistence.MessageClassification{
			Intent:            "FACTUAL_QUESTION",
			Explain:           "user asks about prior context",
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
		INSERT INTO "users" ("telegram_id", "telegram_username", "has_access", "created_at", "updated_at")
		VALUES ($1, NULLIF($2, ''), false, now(), now())
		ON CONFLICT ("telegram_id") DO UPDATE
		SET "telegram_username" = COALESCE(NULLIF(EXCLUDED."telegram_username", ''), "users"."telegram_username"),
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
		INSERT INTO "messages" ("conversation_id", "user_id", "message_text", "language_code", "telegram_message_id", "created_at")
		VALUES ($1, $2, $3, NULLIF($4, ''), $5, $6)
		RETURNING "id"
	`)).
		WithArgs(int64(11), int64(7), "What is the test code?", "en", nil, now).
		WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(int64(100)))

	mock.ExpectQuery(regexp.QuoteMeta(`
		INSERT INTO "messages" ("conversation_id", "user_id", "message_text", "language_code", "telegram_message_id", "created_at")
		VALUES ($1, $2, $3, NULLIF($4, ''), $5, $6)
		RETURNING "id"
	`)).
		WithArgs(int64(11), nil, "The test code is ALPHA-123.", "en", nil, now).
		WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(int64(101)))

	mock.ExpectExec(regexp.QuoteMeta(`
		INSERT INTO "message_classifications" (
			"message_id",
			"intent",
			"explain",
			"detected_language",
			"classifier_model",
			"classifier_version",
			"created_at"
		)
		VALUES ($1, $2, NULLIF($3, ''), NULLIF($4, ''), NULLIF($5, ''), NULLIF($6, ''), $7)
		ON CONFLICT ("message_id") DO UPDATE
		SET "intent" = EXCLUDED."intent",
		    "explain" = EXCLUDED."explain",
		    "detected_language" = EXCLUDED."detected_language",
		    "classifier_model" = EXCLUDED."classifier_model",
		    "classifier_version" = EXCLUDED."classifier_version"
	`)).
		WithArgs(int64(100), "FACTUAL_QUESTION", "user asks about prior context", "en", "gpt-4o-mini", "v1", now).
		WillReturnResult(sqlmock.NewResult(1, 1))

	mock.ExpectExec(regexp.QuoteMeta(`
		INSERT INTO "usage_events" (
			"user_id",
			"event_type",
			"input_tokens",
			"output_tokens",
			"total_tokens",
			"estimated_cost",
			"created_at"
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`)).
		WithArgs(int64(7), "classification", 0, 0, 0, 0.0, now).
		WillReturnResult(sqlmock.NewResult(1, 1))

	mock.ExpectExec(regexp.QuoteMeta(`
		INSERT INTO "usage_events" (
			"user_id",
			"event_type",
			"input_tokens",
			"output_tokens",
			"total_tokens",
			"estimated_cost",
			"created_at"
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`)).
		WithArgs(int64(7), "chat_completion", 0, 0, 0, 0.0, now).
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
		INSERT INTO "users" ("telegram_id", "telegram_username", "has_access", "created_at", "updated_at")
		VALUES ($1, NULLIF($2, ''), false, now(), now())
		ON CONFLICT ("telegram_id") DO UPDATE
		SET "telegram_username" = COALESCE(NULLIF(EXCLUDED."telegram_username", ''), "users"."telegram_username"),
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
		UserMessage:      persistence.Message{Role: qa.ConversationRoleUser, Text: "hello"},
		AssistantMessage: persistence.Message{Role: qa.ConversationRoleAssistant, Text: "world"},
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("SaveTurn() error = %v, want wrapped %v", err, wantErr)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("ExpectationsWereMet() error = %v", err)
	}
}
