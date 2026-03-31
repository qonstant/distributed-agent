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
	Memory      port.ConversationMemory
}

func (uc AskQuestion) Execute(ctx context.Context, user access.User, text string) (qa.Response, error) {
	if err := uc.Policy.Authorize(ctx, user); err != nil {
		return qa.Response{}, err
	}

	question, err := qa.NewQuestion(text)
	if err != nil {
		return qa.Response{}, err
	}

	conversation := qa.ConversationContext{}
	if uc.Memory != nil {
		if loaded, err := uc.Memory.Context(ctx, user.TelegramID); err == nil {
			conversation = loaded
		}
	}

	draft, err := askDraft(ctx, uc.Answers, question, conversation.Messages)
	if err != nil {
		return qa.Response{}, err
	}

	if uc.Memory != nil {
		_ = uc.Memory.RememberTurn(ctx, user.TelegramID, conversation.ID, question.Text, draft.Text)
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

func askDraft(
	ctx context.Context,
	answers port.AnswerSource,
	question qa.Question,
	history []qa.ConversationMessage,
) (qa.DraftResponse, error) {
	if contextual, ok := answers.(port.ContextualAnswerSource); ok {
		return contextual.AskWithHistory(ctx, question, history)
	}

	return answers.Ask(ctx, question)
}
