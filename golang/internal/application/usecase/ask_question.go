package usecase

import (
	"context"
	"fmt"
	"path"
	"strings"
	"time"
	"unicode"

	"github.com/qonstant/distributed-agent/internal/application/port"
	"github.com/qonstant/distributed-agent/internal/domain/access"
	"github.com/qonstant/distributed-agent/internal/domain/persistence"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

const preferredNamePromptEnglish = "By the way, how should I call you? You can say: \"Call me Alex\"."

type AskQuestion struct {
	Policy      access.Policy
	Answers     port.AnswerSource
	Attachments port.AttachmentResolver
	Memory      port.ConversationMemory
	TurnEvents  port.TurnEventPublisher
	Now         func() time.Time
}

func (uc AskQuestion) Execute(ctx context.Context, user access.User, text string) (qa.Response, error) {
	record, err := uc.Policy.AuthorizeAndLoad(ctx, user)
	if err != nil {
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

	preferredName := strings.TrimSpace(record.Username)

	conversation := qa.ConversationContext{}
	if uc.Memory != nil {
		if loaded, err := uc.Memory.Context(ctx, user.TelegramID); err == nil {
			conversation = loaded
		}
	}
	if strings.TrimSpace(conversation.ID) == "" {
		conversation.ID = fallbackConversationKey(user.TelegramID, now)
	}

	draft, err := askDraft(ctx, uc.Answers, question, conversation.ID, preferredName)
	if err != nil {
		return qa.Response{}, err
	}

	if updatedName, updated := classifiedPreferredName(draft.Classification); updated {
		preferredName = updatedName
		draft.Text = preferredNameAcknowledgement(preferredName, detectedLanguage(draft.Classification))
	} else if preferredName == "" {
		draft.Text = mergePreferredNamePrompt(draft.Text, detectedLanguage(draft.Classification))
	}

	response := qa.Response{Text: draft.Text}
	var memoryAttachments []qa.ConversationAttachment
	if len(draft.AttachmentRefs) == 0 {
		uc.rememberUserProfile(ctx, record, preferredName)
		if uc.Memory != nil {
			_ = uc.Memory.RememberTurn(ctx, user.TelegramID, conversation.ID, question.Text, draft.Text, nil)
		}
		if uc.TurnEvents != nil {
			_ = uc.TurnEvents.PublishTurn(ctx, buildTurnEvent(user, preferredName, conversation.ID, question, draft, now))
		}
		return response, nil
	}

	attachments, err := uc.Attachments.Resolve(ctx, draft.AttachmentRefs)
	if err != nil {
		return response, err
	}

	response.Attachments = attachments
	memoryAttachments = toConversationAttachments(draft.AttachmentRefs, attachments)
	uc.rememberUserProfile(ctx, record, preferredName)
	if uc.Memory != nil {
		_ = uc.Memory.RememberTurn(ctx, user.TelegramID, conversation.ID, question.Text, draft.Text, memoryAttachments)
	}
	if uc.TurnEvents != nil {
		_ = uc.TurnEvents.PublishTurn(ctx, buildTurnEvent(user, preferredName, conversation.ID, question, draft, now))
	}
	return response, nil
}

func askDraft(
	ctx context.Context,
	answers port.AnswerSource,
	question qa.Question,
	conversationID string,
	preferredName string,
) (qa.DraftResponse, error) {
	if contextual, ok := answers.(port.ConversationAwareAnswerSource); ok {
		return contextual.AskWithConversation(ctx, question, conversationID, preferredName)
	}

	return answers.Ask(ctx, question)
}

func (uc AskQuestion) rememberUserProfile(ctx context.Context, record access.Record, preferredName string) {
	if record.TelegramID == 0 {
		return
	}

	name := strings.TrimSpace(preferredName)
	if name == "" {
		name = strings.TrimSpace(record.Username)
	}

	_ = uc.Policy.RememberRecord(ctx, access.Record{
		TelegramID:      record.TelegramID,
		Username:        name,
		IsBlocked:       record.IsBlocked,
		AccessExpiresAt: record.AccessExpiresAt,
	})
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
	preferredName string,
	conversationID string,
	question qa.Question,
	draft qa.DraftResponse,
	now time.Time,
) persistence.TurnEvent {
	event := persistence.TurnEvent{
		Version: 1,
		User: persistence.User{
			TelegramID:  user.TelegramID,
			Username:    strings.TrimSpace(preferredName),
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

func classifiedPreferredName(classification *qa.MessageClassification) (string, bool) {
	if classification == nil {
		return "", false
	}
	if strings.TrimSpace(classification.ProfileAction) != "set_preferred_name" {
		return "", false
	}

	name := normalizePreferredName(classification.PreferredName)
	if name == "" {
		return "", false
	}
	return name, true
}

func detectedLanguage(classification *qa.MessageClassification) string {
	if classification == nil {
		return ""
	}
	return strings.TrimSpace(classification.DetectedLanguage)
}

func normalizePreferredName(value string) string {
	value = strings.TrimSpace(value)
	value = strings.Trim(value, " \t\n\r.,!?;:\"'()[]{}")
	value = strings.Join(strings.Fields(value), " ")
	if value == "" {
		return ""
	}

	words := strings.Fields(value)
	if len(words) > 4 {
		return ""
	}

	runes := []rune(value)
	if len(runes) > 64 {
		return ""
	}

	hasLetter := false
	for _, r := range runes {
		switch {
		case unicode.IsLetter(r):
			hasLetter = true
		case unicode.IsSpace(r):
		case r == '-' || r == '\'' || r == '`':
		default:
			return ""
		}
	}
	if !hasLetter {
		return ""
	}

	return value
}

func preferredNameAcknowledgement(name, language string) string {
	switch strings.ToLower(strings.TrimSpace(language)) {
	case "ru", "russian", "русский":
		if strings.TrimSpace(name) == "" {
			return "Приятно познакомиться! Я это запомню."
		}
		return fmt.Sprintf("Приятно познакомиться, %s! Буду звать тебя так.", name)
	case "kk", "kazakh", "қазақ", "қазақша":
		if strings.TrimSpace(name) == "" {
			return "Танысқаныма қуаныштымын! Мұны есте сақтаймын."
		}
		return fmt.Sprintf("Танысқаныма қуаныштымын, %s! Сені осылай атаймын.", name)
	}
	if strings.TrimSpace(name) == "" {
		return "Nice to meet you! I'll remember that."
	}
	return fmt.Sprintf("Nice to meet you, %s! I'll call you that.", name)
}

func preferredNamePrompt(language string) string {
	switch strings.ToLower(strings.TrimSpace(language)) {
	case "ru", "russian", "русский":
		return "Кстати, как мне тебя называть? Можешь написать: \"Зови меня Алекс\"."
	case "kk", "kazakh", "қазақ", "қазақша":
		return "Айтпақшы, сені қалай атайын? Мысалы: \"Мені Алекс деп ата\" деп жаза аласың."
	default:
		return preferredNamePromptEnglish
	}
}

func mergePreferredNamePrompt(base, language string) string {
	prompt := preferredNamePrompt(language)
	base = strings.TrimSpace(base)
	if base == "" {
		return prompt
	}
	if strings.Contains(base, prompt) {
		return base
	}
	return base + "\n\n" + prompt
}
