package telegram

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"

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

	if strings.HasPrefix(update.Message.Text, "/") {
		h.sendText(ctx, b, update.Message.Chat.ID, "Unknown command. Try /help.")
		return
	}

	user := userFromUpdate(update)
	response, err := h.ask.Execute(ctx, user, update.Message.Text)
	if err != nil {
		if h.handleAccessError(ctx, b, update.Message.Chat.ID, err) {
			return
		}

		log.Printf("[defaultHandler] ask question failed for %s: %v", user.DisplayName, err)
		if strings.TrimSpace(response.Text) != "" {
			h.sendText(
				ctx,
				b,
				update.Message.Chat.ID,
				fmt.Sprintf("%s\n\n(Не удалось подготовить вложения: %v)", response.Text, err),
			)
			return
		}

		h.sendText(ctx, b, update.Message.Chat.ID, fmt.Sprintf("Ошибка обращения к локальному API: %v", err))
		return
	}

	if err := h.presenter.Present(ctx, b, update.Message.Chat.ID, response); err != nil {
		log.Printf("[defaultHandler] present failed for %s: %v", user.DisplayName, err)
		h.sendText(
			ctx,
			b,
			update.Message.Chat.ID,
			fmt.Sprintf("%s\n\n(Не удалось отправить файл: %v)", response.Text, err),
		)
	}
}

func (h *Handlers) HandleStart(ctx context.Context, b *bot.Bot, update *models.Update) {
	if update == nil || update.Message == nil || update.Message.From == nil {
		return
	}

	text, err := h.start.Execute(userFromUpdate(update))
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

	text, err := h.help.Execute(userFromUpdate(update))
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
		Username:    update.Message.From.Username,
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
