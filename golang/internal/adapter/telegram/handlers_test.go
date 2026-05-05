package telegram

import (
	"strings"
	"testing"

	"github.com/go-telegram/bot/models"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

func TestAskFailureUserText(t *testing.T) {
	t.Parallel()

	t.Run("hides internal attachment resolution error when partial answer exists", func(t *testing.T) {
		t.Parallel()

		text := askFailureUserText(qa.Response{Text: "answer"}, "ru")

		if got, want := text, "answer"; got != want {
			t.Fatalf("text = %q, want %q", got, want)
		}
		if strings.Contains(text, "read s3 object") {
			t.Fatalf("text exposes internal error: %q", text)
		}
	})

	t.Run("uses generic message for complete ask failure", func(t *testing.T) {
		t.Parallel()

		text := askFailureUserText(qa.Response{}, "ru")

		if got, want := text, "Сейчас не удалось подготовить ответ. Попробуйте позже."; got != want {
			t.Fatalf("text = %q, want %q", got, want)
		}
		if strings.Contains(text, "local api") || strings.Contains(text, "deadline") {
			t.Fatalf("text exposes internal API details: %q", text)
		}
	})
}

func TestFileDeliveryFailureUserText(t *testing.T) {
	t.Parallel()

	t.Run("keeps answer without exposing send error details", func(t *testing.T) {
		t.Parallel()

		text := fileDeliveryFailureUserText(qa.Response{Text: "answer"}, "ru")

		if got, want := text, "answer"; got != want {
			t.Fatalf("text = %q, want %q", got, want)
		}
		if strings.Contains(text, "send") || strings.Contains(text, "error") {
			t.Fatalf("text exposes internal delivery details: %q", text)
		}
	})

	t.Run("uses generic message when only file delivery failed", func(t *testing.T) {
		t.Parallel()

		text := fileDeliveryFailureUserText(qa.Response{}, "ru")

		if got, want := text, "Не удалось отправить файл. Попробуйте позже."; got != want {
			t.Fatalf("text = %q, want %q", got, want)
		}
	})
}

func TestUnsupportedInputText(t *testing.T) {
	t.Parallel()

	t.Run("text message is supported", func(t *testing.T) {
		t.Parallel()

		update := &models.Update{
			Message: &models.Message{
				Text: "hello",
				From: &models.User{LanguageCode: "en"},
			},
		}

		text, unsupported := unsupportedInputText(update)
		if unsupported {
			t.Fatalf("unsupported = true, text = %q", text)
		}
	})

	t.Run("photo message uses russian image notice", func(t *testing.T) {
		t.Parallel()

		update := &models.Update{
			Message: &models.Message{
				Photo: []models.PhotoSize{{FileID: "photo-id"}},
				From:  &models.User{LanguageCode: "ru"},
			},
		}

		text, unsupported := unsupportedInputText(update)
		if !unsupported {
			t.Fatal("unsupported = false, want true")
		}
		if got, want := text, "Я пока не умею обрабатывать изображения. Пожалуйста, отправьте вопрос текстом."; got != want {
			t.Fatalf("text = %q, want %q", got, want)
		}
	})

	t.Run("photo message uses kazakh image notice", func(t *testing.T) {
		t.Parallel()

		update := &models.Update{
			Message: &models.Message{
				Photo: []models.PhotoSize{{FileID: "photo-id"}},
				From:  &models.User{LanguageCode: "kk"},
			},
		}

		text, unsupported := unsupportedInputText(update)
		if !unsupported {
			t.Fatal("unsupported = false, want true")
		}
		if got, want := text, "Мен әзірге суреттерді өңдей алмаймын. Сұрағыңызды мәтінмен жіберіңіз."; got != want {
			t.Fatalf("text = %q, want %q", got, want)
		}
	})

	t.Run("photo message falls back to cyrillic caption detection", func(t *testing.T) {
		t.Parallel()

		update := &models.Update{
			Message: &models.Message{
				Caption: "Дарова",
				Photo:   []models.PhotoSize{{FileID: "photo-id"}},
			},
		}

		text, unsupported := unsupportedInputText(update)
		if !unsupported {
			t.Fatal("unsupported = false, want true")
		}
		if got, want := text, "Я пока не умею обрабатывать изображения. Пожалуйста, отправьте вопрос текстом."; got != want {
			t.Fatalf("text = %q, want %q", got, want)
		}
	})

	t.Run("sticker message uses generic english notice", func(t *testing.T) {
		t.Parallel()

		update := &models.Update{
			Message: &models.Message{
				Sticker: &models.Sticker{FileID: "sticker-id"},
				From:    &models.User{LanguageCode: "en"},
			},
		}

		text, unsupported := unsupportedInputText(update)
		if !unsupported {
			t.Fatal("unsupported = false, want true")
		}
		if got, want := text, "I currently accept text messages only. Please send your question as text."; got != want {
			t.Fatalf("text = %q, want %q", got, want)
		}
	})
}
