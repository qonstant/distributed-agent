package telegram

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"unicode"

	"github.com/go-telegram/bot"
	"github.com/go-telegram/bot/models"
	"github.com/qonstant/distributed-agent/internal/application/usecase"
	"github.com/qonstant/distributed-agent/internal/domain/access"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type Handlers struct {
	start             usecase.GetStartMessage
	help              usecase.GetHelpMessage
	ask               usecase.AskQuestion
	sampleAttachments usecase.GetSampleAttachments
	presenter         Presenter
	accessDeniedText  string
}

func NewHandlers(
	start usecase.GetStartMessage,
	help usecase.GetHelpMessage,
	ask usecase.AskQuestion,
	sampleAttachments usecase.GetSampleAttachments,
	presenter Presenter,
	accessDeniedText string,
) *Handlers {
	return &Handlers{
		start:             start,
		help:              help,
		ask:               ask,
		sampleAttachments: sampleAttachments,
		presenter:         presenter,
		accessDeniedText:  accessDeniedText,
	}
}

func (h *Handlers) Register(b *bot.Bot) {
	b.RegisterHandler(bot.HandlerTypeMessageText, "/start", bot.MatchTypeCommand, h.HandleStart)
	b.RegisterHandler(bot.HandlerTypeMessageText, "/help", bot.MatchTypeCommand, h.HandleHelp)
	b.RegisterHandler(bot.HandlerTypeMessageText, "/randompic", bot.MatchTypeCommand, h.HandleRandomPic)

	b.RegisterHandlerRegexp(bot.HandlerTypeMessageText, commandRegexp, func(ctx context.Context, b *bot.Bot, update *models.Update) {
		if update == nil || update.Message == nil {
			return
		}

		switch {
		case strings.HasPrefix(update.Message.Text, "/start"):
			h.HandleStart(ctx, b, update)
		case strings.HasPrefix(update.Message.Text, "/help"):
			h.HandleHelp(ctx, b, update)
		case strings.HasPrefix(update.Message.Text, "/randompic"):
			h.HandleRandomPic(ctx, b, update)
		}
	})
}

func (h *Handlers) HandleDefault(ctx context.Context, b *bot.Bot, update *models.Update) {
	if update == nil || update.Message == nil || update.Message.From == nil {
		return
	}

	if text, unsupported := unsupportedInputText(update); unsupported {
		h.sendText(ctx, b, update.Message.Chat.ID, text)
		return
	}

	if strings.HasPrefix(update.Message.Text, "/") {
		h.sendText(ctx, b, update.Message.Chat.ID, "Unknown command. Try /help.")
		return
	}

	user := userFromUpdate(update)
	progress := StartProgressMessage(ctx, b, update.Message.Chat.ID, update.Message.Text)
	response, err := h.ask.Execute(ctx, user, update.Message.Text)
	if err != nil {
		if errors.Is(err, access.ErrUnauthorized) {
			if progress != nil {
				_ = progress.Replace(ctx, h.accessDeniedText)
			} else {
				h.sendText(ctx, b, update.Message.Chat.ID, h.accessDeniedText)
			}
			return
		}

		log.Printf("[defaultHandler] ask question failed for %s: %v", user.DisplayName, err)
		if strings.TrimSpace(response.Text) != "" {
			text := fmt.Sprintf("%s\n\n(Не удалось подготовить вложения: %v)", response.Text, err)
			if progress != nil {
				_ = progress.Replace(ctx, text)
			} else {
				h.sendText(ctx, b, update.Message.Chat.ID, text)
			}
			return
		}

		text := fmt.Sprintf("Ошибка обращения к локальному API: %v", err)
		if progress != nil {
			_ = progress.Replace(ctx, text)
		} else {
			h.sendText(ctx, b, update.Message.Chat.ID, text)
		}
		return
	}

	if err := h.presenter.PresentWithProgress(ctx, b, update.Message.Chat.ID, response, progress); err != nil {
		log.Printf("[defaultHandler] present failed for %s: %v", user.DisplayName, err)
		if progress != nil {
			_ = progress.Replace(
				ctx,
				fmt.Sprintf("%s\n\n(Не удалось отправить файл: %v)", response.Text, err),
			)
		} else {
			h.sendText(
				ctx,
				b,
				update.Message.Chat.ID,
				fmt.Sprintf("%s\n\n(Не удалось отправить файл: %v)", response.Text, err),
			)
		}
	}
}

func (h *Handlers) HandleStart(ctx context.Context, b *bot.Bot, update *models.Update) {
	if update == nil || update.Message == nil || update.Message.From == nil {
		return
	}

	text, err := h.start.Execute(ctx, userFromUpdate(update))
	if err != nil {
		if h.handleAccessError(ctx, b, update.Message.Chat.ID, err) {
			return
		}
		log.Printf("[startHandler] failed: %v", err)
		return
	}

	h.sendText(ctx, b, update.Message.Chat.ID, text)
}

func (h *Handlers) HandleHelp(ctx context.Context, b *bot.Bot, update *models.Update) {
	if update == nil || update.Message == nil || update.Message.From == nil {
		return
	}

	text, err := h.help.Execute(ctx, userFromUpdate(update))
	if err != nil {
		if h.handleAccessError(ctx, b, update.Message.Chat.ID, err) {
			return
		}
		log.Printf("[helpHandler] failed: %v", err)
		return
	}

	h.sendText(ctx, b, update.Message.Chat.ID, text)
}

func (h *Handlers) HandleRandomPic(ctx context.Context, b *bot.Bot, update *models.Update) {
	if update == nil || update.Message == nil || update.Message.From == nil {
		return
	}

	response, err := h.sampleAttachments.Execute(ctx, userFromUpdate(update))
	if err != nil {
		if h.handleAccessError(ctx, b, update.Message.Chat.ID, err) {
			return
		}
		log.Printf("[randomPicHandler] failed: %v", err)
		h.sendText(ctx, b, update.Message.Chat.ID, fmt.Sprintf("Failed to send images as album: %v", err))
		return
	}

	if err := h.presenter.Present(ctx, b, update.Message.Chat.ID, response); err != nil {
		log.Printf("[randomPicHandler] present failed: %v", err)
		h.sendText(ctx, b, update.Message.Chat.ID, fmt.Sprintf("Failed to send images as album: %v", err))
	}
}

func (h *Handlers) handleAccessError(ctx context.Context, b *bot.Bot, chatID int64, err error) bool {
	if !errors.Is(err, access.ErrUnauthorized) {
		return false
	}

	h.sendText(ctx, b, chatID, h.accessDeniedText)
	return true
}

func (h *Handlers) sendText(ctx context.Context, b *bot.Bot, chatID int64, text string) {
	if _, err := b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID: chatID,
		Text:   text,
	}); err != nil {
		log.Printf("[telegram] send message failed: %v", err)
	}
}

func userFromUpdate(update *models.Update) access.User {
	if update == nil || update.Message == nil || update.Message.From == nil {
		return access.User{}
	}

	return access.User{
		TelegramID:  int64(update.Message.From.ID),
		Username:    "",
		DisplayName: displayName(update),
	}
}

func displayName(update *models.Update) string {
	if update == nil || update.Message == nil || update.Message.From == nil {
		return ""
	}

	if update.Message.From.Username != "" {
		return update.Message.From.Username
	}

	return strings.TrimSpace(
		fmt.Sprintf("%s %s", update.Message.From.FirstName, update.Message.From.LastName),
	)
}

func isPhotoAlbum(response qa.Response) bool {
	if len(response.Attachments) < 2 {
		return false
	}

	for _, attachment := range response.Attachments {
		if attachment.Kind != qa.AttachmentPhoto {
			return false
		}
	}

	return true
}

func unsupportedInputText(update *models.Update) (string, bool) {
	if update == nil || update.Message == nil {
		return "", false
	}

	if strings.TrimSpace(update.Message.Text) != "" {
		return "", false
	}

	language := detectReplyLanguage(update)
	if len(update.Message.Photo) > 0 {
		return unsupportedImageMessage(language), true
	}

	if update.Message.Document != nil ||
		update.Message.Video != nil ||
		update.Message.VideoNote != nil ||
		update.Message.Voice != nil ||
		update.Message.Audio != nil ||
		update.Message.Animation != nil ||
		update.Message.Sticker != nil ||
		update.Message.Contact != nil ||
		update.Message.Location != nil ||
		update.Message.Venue != nil ||
		update.Message.Poll != nil {
		return unsupportedGenericMessage(language), true
	}

	return unsupportedGenericMessage(language), true
}

func detectReplyLanguage(update *models.Update) string {
	if update == nil || update.Message == nil {
		return "en"
	}

	if update.Message.From != nil {
		code := strings.ToLower(strings.TrimSpace(update.Message.From.LanguageCode))
		switch {
		case strings.HasPrefix(code, "kk"):
			return "kk"
		case strings.HasPrefix(code, "ru"):
			return "ru"
		case strings.HasPrefix(code, "en"):
			return "en"
		}
	}

	sample := strings.TrimSpace(update.Message.Text)
	if sample == "" {
		sample = strings.TrimSpace(update.Message.Caption)
	}
	if sample == "" {
		return "en"
	}

	for _, r := range sample {
		if !unicode.IsLetter(r) {
			continue
		}
		if unicode.In(r, unicode.Cyrillic) {
			return "ru"
		}
		if unicode.In(r, unicode.Latin) {
			return "en"
		}
	}
	return "en"
}

func unsupportedImageMessage(language string) string {
	switch language {
	case "ru":
		return "Я пока не умею обрабатывать изображения. Пожалуйста, отправьте вопрос текстом."
	case "kk":
		return "Мен әзірге суреттерді өңдей алмаймын. Сұрағыңызды мәтінмен жіберіңіз."
	default:
		return "I can't process images yet. Please send your question as text."
	}
}

func unsupportedGenericMessage(language string) string {
	switch language {
	case "ru":
		return "Сейчас я принимаю только текстовые сообщения. Пожалуйста, отправьте вопрос текстом."
	case "kk":
		return "Қазір мен тек мәтіндік хабарламаларды қабылдаймын. Сұрағыңызды мәтінмен жіберіңіз."
	default:
		return "I currently accept text messages only. Please send your question as text."
	}
}
