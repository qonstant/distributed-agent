package qa

import (
	"errors"
	"testing"
)

func TestNewQuestion(t *testing.T) {
	t.Parallel()

	t.Run("trims surrounding whitespace", func(t *testing.T) {
		t.Parallel()

		question, err := NewQuestion("  hello world  ")
		if err != nil {
			t.Fatalf("NewQuestion() error = %v", err)
		}
		if got, want := question.Text, "hello world"; got != want {
			t.Fatalf("Question.Text = %q, want %q", got, want)
		}
	})

	t.Run("rejects empty text", func(t *testing.T) {
		t.Parallel()

		_, err := NewQuestion("   ")
		if !errors.Is(err, ErrEmptyQuestion) {
			t.Fatalf("NewQuestion() error = %v, want %v", err, ErrEmptyQuestion)
		}
	})
}
