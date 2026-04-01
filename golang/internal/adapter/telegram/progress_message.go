package telegram

import (
	"context"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/go-telegram/bot"
)

var thinkingFramesLatin = []string{
	"Thinking.",
	"Thinking..",
	"Thinking...",
}

var thinkingFramesCyrillic = []string{
	"Думаю.",
	"Думаю..",
	"Думаю...",
}

const thinkingUpdateInterval = 1200 * time.Millisecond

type ProgressMessage struct {
	bot       *bot.Bot
	chatID    int64
	messageID int
	cancel    context.CancelFunc
	done      chan struct{}
	stopOnce  sync.Once
	frames    []string
}

func StartProgressMessage(ctx context.Context, b *bot.Bot, chatID int64, text string) *ProgressMessage {
	if b == nil {
		return nil
	}
	frames := thinkingFramesForText(text)

	msg, err := b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID: chatID,
		Text:   frames[0],
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
		frames:    frames,
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
			text := p.frames[frameIdx%len(p.frames)]
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

func thinkingFramesForText(text string) []string {
	hasLatin := false
	for _, r := range text {
		if !unicode.IsLetter(r) {
			continue
		}
		if unicode.In(r, unicode.Cyrillic) {
			return thinkingFramesCyrillic
		}
		if unicode.In(r, unicode.Latin) {
			hasLatin = true
		}
	}
	if hasLatin {
		return thinkingFramesLatin
	}
	return thinkingFramesLatin
}
