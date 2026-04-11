package telegram

import (
	"testing"

	"github.com/go-telegram/bot/models"
)

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
