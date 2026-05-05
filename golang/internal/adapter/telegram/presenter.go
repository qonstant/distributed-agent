package telegram

import (
	"context"
	"strings"

	"github.com/go-telegram/bot"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type Presenter struct {
	sender *Sender
}

func NewPresenter(sender *Sender) Presenter {
	return Presenter{sender: sender}
}

func (p Presenter) Present(ctx context.Context, b *bot.Bot, chatID int64, response qa.Response) error {
	return p.present(ctx, b, chatID, response, nil)
}

func (p Presenter) PresentWithProgress(ctx context.Context, b *bot.Bot, chatID int64, response qa.Response, progress *ProgressMessage) error {
	return p.present(ctx, b, chatID, response, progress)
}

func (p Presenter) present(
	ctx context.Context,
	b *bot.Bot,
	chatID int64,
	response qa.Response,
	progress *ProgressMessage,
) error {
	textDelivered := false
	if progress != nil {
		if strings.TrimSpace(response.Text) != "" {
			if err := progress.Replace(ctx, response.Text); err == nil {
				textDelivered = true
			} else {
				_ = progress.Delete(ctx)
			}
		} else {
			_ = progress.Delete(ctx)
		}
	}

	if len(response.Attachments) == 0 {
		if textDelivered {
			return nil
		}
		if strings.TrimSpace(response.Text) == "" {
			return nil
		}
		_, err := b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: chatID,
			Text:   response.Text,
		})
		return err
	}

	if len(response.Attachments) == 1 {
		caption := response.Text
		if textDelivered {
			caption = ""
		}
		return p.sender.SendDocument(ctx, chatID, response.Attachments[0], caption)
	}

	if isPhotoAlbum(response) {
		caption := response.Text
		if textDelivered {
			caption = ""
		}
		return p.sender.SendMediaGroup(ctx, chatID, response.Attachments, caption)
	}

	if strings.TrimSpace(response.Text) != "" && !textDelivered {
		if _, err := b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: chatID,
			Text:   response.Text,
		}); err != nil {
			return err
		}
	}

	for _, attachment := range response.Attachments {
		if err := p.sender.SendDocument(ctx, chatID, attachment, ""); err != nil {
			return err
		}
	}

	return nil
}
