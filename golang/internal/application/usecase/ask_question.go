package usecase

import (
	"context"
	"path"
	"strings"

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

	draft, err := askDraft(ctx, uc.Answers, question, conversation.ID)
	if err != nil {
		return qa.Response{}, err
	}

	response := qa.Response{Text: draft.Text}
	var memoryAttachments []qa.ConversationAttachment
	if len(draft.AttachmentRefs) == 0 {
		if uc.Memory != nil {
			_ = uc.Memory.RememberTurn(ctx, user.TelegramID, conversation.ID, question.Text, draft.Text, nil)
		}
		return response, nil
	}

	attachments, err := uc.Attachments.Resolve(ctx, draft.AttachmentRefs)
	if err != nil {
		return response, err
	}

	response.Attachments = attachments
	memoryAttachments = toConversationAttachments(draft.AttachmentRefs, attachments)
	if uc.Memory != nil {
		_ = uc.Memory.RememberTurn(ctx, user.TelegramID, conversation.ID, question.Text, draft.Text, memoryAttachments)
	}
	return response, nil
}

func askDraft(
	ctx context.Context,
	answers port.AnswerSource,
	question qa.Question,
	conversationID string,
) (qa.DraftResponse, error) {
	if contextual, ok := answers.(port.ConversationAwareAnswerSource); ok {
		return contextual.AskWithConversation(ctx, question, conversationID)
	}

	return answers.Ask(ctx, question)
}

func toConversationAttachments(refs []qa.AttachmentRef, attachments []qa.Attachment) []qa.ConversationAttachment {
	if len(refs) == 0 {
		return nil
	}

	out := make([]qa.ConversationAttachment, 0, len(refs))
	for idx, ref := range refs {
		source := strings.TrimSpace(ref.Source)
		if source == "" {
			continue
		}

		name := path.Base(source)
		if idx < len(attachments) && strings.TrimSpace(attachments[idx].Name) != "" {
			name = strings.TrimSpace(attachments[idx].Name)
		}
		if name == "." || name == "/" {
			name = ""
		}

		kind := ref.Kind
		if idx < len(attachments) && attachments[idx].Kind != "" {
			kind = attachments[idx].Kind
		}
		if kind == "" {
			kind = qa.AttachmentDocument
		}
		if name == "" {
			continue
		}

		out = append(out, qa.ConversationAttachment{
			Source: source,
			Name:   name,
			Kind:   kind,
		})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
