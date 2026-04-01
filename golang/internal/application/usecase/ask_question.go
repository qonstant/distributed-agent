package usecase

import (
	"context"
	"fmt"
	"path"
	"strings"
	"time"

	"github.com/qonstant/distributed-agent/internal/application/port"
	"github.com/qonstant/distributed-agent/internal/domain/access"
	"github.com/qonstant/distributed-agent/internal/domain/persistence"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type AskQuestion struct {
	Policy      access.Policy
	Answers     port.AnswerSource
	Attachments port.AttachmentResolver
	Memory      port.ConversationMemory
	TurnEvents  port.TurnEventPublisher
	Now         func() time.Time
}

func (uc AskQuestion) Execute(ctx context.Context, user access.User, text string) (qa.Response, error) {
	if err := uc.Policy.Authorize(ctx, user); err != nil {
		return qa.Response{}, err
	}

	question, err := qa.NewQuestion(text)
	if err != nil {
		return qa.Response{}, err
	}
	now := time.Now().UTC()
	if uc.Now != nil {
		now = uc.Now().UTC()
	}

	conversation := qa.ConversationContext{}
	if uc.Memory != nil {
		if loaded, err := uc.Memory.Context(ctx, user.TelegramID); err == nil {
			conversation = loaded
		}
	}
	if strings.TrimSpace(conversation.ID) == "" {
		conversation.ID = fallbackConversationKey(user.TelegramID, now)
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
		if uc.TurnEvents != nil {
			_ = uc.TurnEvents.PublishTurn(ctx, buildTurnEvent(user, conversation.ID, question, draft, now))
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
	if uc.TurnEvents != nil {
		_ = uc.TurnEvents.PublishTurn(ctx, buildTurnEvent(user, conversation.ID, question, draft, now))
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

func buildTurnEvent(
	user access.User,
	conversationID string,
	question qa.Question,
	draft qa.DraftResponse,
	now time.Time,
) persistence.TurnEvent {
	event := persistence.TurnEvent{
		Version: 1,
		User: persistence.User{
			TelegramID:  user.TelegramID,
			Username:    strings.TrimSpace(user.Username),
			DisplayName: strings.TrimSpace(user.DisplayName),
		},
		Conversation: persistence.Conversation{
			Key:       strings.TrimSpace(conversationID),
			CreatedAt: now,
			UpdatedAt: now,
		},
		UserMessage: persistence.Message{
			Text:      question.Text,
			CreatedAt: now,
		},
	}

	if draft.Classification != nil {
		event.Classification = &persistence.MessageClassification{
			Intent:            strings.TrimSpace(draft.Classification.Intent),
			Explanation:       strings.TrimSpace(draft.Classification.Explain),
			DetectedLanguage:  strings.TrimSpace(draft.Classification.DetectedLanguage),
			ClassifierModel:   strings.TrimSpace(draft.Classification.ClassifierModel),
			ClassifierVersion: strings.TrimSpace(draft.Classification.ClassifierVersion),
			CreatedAt:         now,
		}
	}
	if len(draft.UsageEvents) > 0 {
		event.UsageEvents = make([]persistence.UsageEvent, 0, len(draft.UsageEvents))
		for _, usage := range draft.UsageEvents {
			event.UsageEvents = append(event.UsageEvents, persistence.UsageEvent{
				EventType:     strings.TrimSpace(usage.EventType),
				InputTokens:   usage.InputTokens,
				OutputTokens:  usage.OutputTokens,
				EstimatedCost: usage.EstimatedCost,
				CreatedAt:     now,
			})
		}
	}

	return event
}

func fallbackConversationKey(ownerID int64, now time.Time) string {
	return fmt.Sprintf("%d-%d", ownerID, now.UnixNano())
}
