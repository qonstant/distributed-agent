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
	if len(response.Attachments) == 0 {
		_, err := b.SendMessage(ctx, &bot.SendMessageParams{
			ChatID: chatID,
			Text:   response.Text,
		})
		return err
	}

	if len(response.Attachments) == 1 {
		return p.sender.SendDocument(ctx, chatID, response.Attachments[0], response.Text)
	}

	if isPhotoAlbum(response) {
		return p.sender.SendMediaGroup(ctx, chatID, response.Attachments, response.Text)
	}

	if strings.TrimSpace(response.Text) != "" {
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
