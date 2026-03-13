package usecase

import (
	"context"

	"github.com/qonstant/distributed-agent/internal/application/port"
	"github.com/qonstant/distributed-agent/internal/domain/access"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type AskQuestion struct {
	Policy      access.Policy
	Answers     port.AnswerSource
	Attachments port.AttachmentResolver
}

func (uc AskQuestion) Execute(ctx context.Context, user access.User, text string) (qa.Response, error) {
	if err := uc.Policy.Authorize(user); err != nil {
		return qa.Response{}, err
	}

	question, err := qa.NewQuestion(text)
	if err != nil {
		return qa.Response{}, err
	}

	draft, err := uc.Answers.Ask(ctx, question)
	if err != nil {
		return qa.Response{}, err
	}

	response := qa.Response{Text: draft.Text}
	if len(draft.AttachmentRefs) == 0 {
		return response, nil
	}

	attachments, err := uc.Attachments.Resolve(ctx, draft.AttachmentRefs)
	if err != nil {
		return response, err
	}

	response.Attachments = attachments
	return response, nil
}
