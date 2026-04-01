package telegram

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/go-telegram/bot"
)

var thinkingFrames = []string{
	"Thinking.",
	"Thinking..",
	"Thinking...",
}

const thinkingUpdateInterval = 1200 * time.Millisecond

type ProgressMessage struct {
	bot       *bot.Bot
	chatID    int64
	messageID int
	cancel    context.CancelFunc
	done      chan struct{}
	stopOnce  sync.Once
}

func StartProgressMessage(ctx context.Context, b *bot.Bot, chatID int64) *ProgressMessage {
	if b == nil {
		return nil
	}

	msg, err := b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID: chatID,
		Text:   thinkingFrames[0],
	})
	if err != nil {
		return nil
	}

	runCtx, cancel := context.WithCancel(ctx)
	progress := &ProgressMessage{
		bot:       b,
		chatID:    chatID,
		messageID: msg.ID,
		cancel:    cancel,
		done:      make(chan struct{}),
	}

	go progress.animate(runCtx)
	return progress
}

func (p *ProgressMessage) animate(ctx context.Context) {
	defer close(p.done)

	ticker := time.NewTicker(thinkingUpdateInterval)
	defer ticker.Stop()

	frameIdx := 1
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			text := thinkingFrames[frameIdx%len(thinkingFrames)]
			frameIdx++
			_, _ = p.bot.EditMessageText(ctx, &bot.EditMessageTextParams{
				ChatID:    p.chatID,
				MessageID: p.messageID,
				Text:      text,
			})
		}
	}
}

func (p *ProgressMessage) stop() {
	if p == nil {
		return
	}

	p.stopOnce.Do(func() {
		if p.cancel != nil {
			p.cancel()
		}
		if p.done != nil {
			<-p.done
		}
	})
}

func (p *ProgressMessage) Replace(ctx context.Context, text string) error {
	if p == nil {
		return nil
	}

	p.stop()
	text = sanitizeUTF8(strings.TrimSpace(text))
	if text == "" {
		return p.Delete(ctx)
	}

	_, err := p.bot.EditMessageText(ctx, &bot.EditMessageTextParams{
		ChatID:    p.chatID,
		MessageID: p.messageID,
		Text:      text,
	})
	return err
}

func (p *ProgressMessage) Delete(ctx context.Context) error {
	if p == nil {
		return nil
	}

	p.stop()
	_, err := p.bot.DeleteMessage(ctx, &bot.DeleteMessageParams{
		ChatID:    p.chatID,
		MessageID: p.messageID,
	})
	return err
}
