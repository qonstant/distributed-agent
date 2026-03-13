package qa

import (
	"errors"
	"strings"
)

var ErrEmptyQuestion = errors.New("question is empty")

type Question struct {
	Text string
}

func NewQuestion(text string) (Question, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return Question{}, ErrEmptyQuestion
	}
	return Question{Text: text}, nil
}
