package telegram

import (
	"context"
	"fmt"
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

type progressFact struct {
	Text   string
	Source string
}

var europeFactsLatin = []progressFact{
	{
		Text:   "Many European universities use ECTS credits so workloads can be compared across countries.",
		Source: "ECTS / European Higher Education Area",
	},
	{
		Text:   "The Bologna Process helped align degree structures across much of Europe into bachelor-master-doctorate cycles.",
		Source: "Bologna Process",
	},
	{
		Text:   "Erasmus+ supports study and exchange mobility across many European countries.",
		Source: "Erasmus+",
	},
}

var europeFactsCyrillic = []progressFact{
	{
		Text:   "Во многих европейских университетах используют кредиты ECTS, чтобы сравнивать учебную нагрузку между странами.",
		Source: "ECTS / European Higher Education Area",
	},
	{
		Text:   "Болонский процесс помог выстроить во многих странах Европы общую структуру: бакалавриат, магистратура, докторантура.",
		Source: "Bologna Process",
	},
	{
		Text:   "Программа Erasmus+ поддерживает учебную мобильность и обмены между многими европейскими странами.",
		Source: "Erasmus+",
	},
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
	facts     []progressFact
}

func StartProgressMessage(ctx context.Context, b *bot.Bot, chatID int64, text string) *ProgressMessage {
	if b == nil {
		return nil
	}
	frames := thinkingFramesForText(text)
	facts := progressFactsForText(text)

	msg, err := b.SendMessage(ctx, &bot.SendMessageParams{
		ChatID: chatID,
		Text:   renderProgressText(frames, facts, 0),
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
		facts:     facts,
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
			text := renderProgressText(p.frames, p.facts, frameIdx)
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

func progressFactsForText(text string) []progressFact {
	hasLatin := false
	for _, r := range text {
		if !unicode.IsLetter(r) {
			continue
		}
		if unicode.In(r, unicode.Cyrillic) {
			return europeFactsCyrillic
		}
		if unicode.In(r, unicode.Latin) {
			hasLatin = true
		}
	}
	if hasLatin {
		return europeFactsLatin
	}
	return europeFactsLatin
}

func renderProgressText(frames []string, facts []progressFact, step int) string {
	header := thinkingFramesLatin[0]
	if len(frames) > 0 {
		header = frames[step%len(frames)]
	}
	if len(facts) == 0 {
		return header
	}

	fact := facts[step%len(facts)]
	sourceLabel := "Source"
	if len(frames) > 0 && frames[0] == thinkingFramesCyrillic[0] {
		sourceLabel = "Источник"
	}
	return fmt.Sprintf("%s\n\n%s\n%s: %s", header, fact.Text, sourceLabel, fact.Source)
}
