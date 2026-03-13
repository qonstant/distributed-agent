package app

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/go-telegram/bot"
	"github.com/qonstant/distributed-agent/internal/adapter/localapi"
	"github.com/qonstant/distributed-agent/internal/adapter/storage"
	telegramadapter "github.com/qonstant/distributed-agent/internal/adapter/telegram"
	"github.com/qonstant/distributed-agent/internal/application/usecase"
	"github.com/qonstant/distributed-agent/internal/config"
	"github.com/qonstant/distributed-agent/internal/domain/access"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

func Run() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	policy := access.NewPolicy(cfg.AllowedUsername)

	s3Store, err := storage.NewS3Store(context.Background(), cfg.S3)
	if err != nil {
		log.Printf("S3 init warning (continuing): %v", err)
	}

	resolver := storage.NewResolver(cfg.DocRoot, s3Store)
	answerSource := localapi.NewClient(cfg.LocalAPIURL)
	presenter := telegramadapter.NewPresenter(telegramadapter.NewSender(cfg.TelegramToken))

	sampleRefs := make([]qa.AttachmentRef, 0, len(cfg.SampleAttachmentKeys))
	for _, key := range cfg.SampleAttachmentKeys {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		sampleRefs = append(sampleRefs, qa.AttachmentRef{
			Source: key,
			Kind:   qa.AttachmentPhoto,
		})
	}

	handlers := telegramadapter.NewHandlers(
		usecase.GetStartMessage{Policy: policy},
		usecase.GetHelpMessage{Policy: policy},
		usecase.AskQuestion{
			Policy:      policy,
			Answers:     answerSource,
			Attachments: resolver,
		},
		usecase.GetSampleAttachments{
			Policy:         policy,
			AttachmentRefs: sampleRefs,
			Attachments:    resolver,
			Title:          cfg.SampleAlbumTitle,
		},
		presenter,
		fmt.Sprintf(
			"This bot accepts requests only with subscription.\n\nTo request access, please message @%s.",
			policy.AllowedUsername(),
		),
	)

	b, err := bot.New(cfg.TelegramToken, bot.WithDefaultHandler(handlers.HandleDefault))
	if err != nil {
		return fmt.Errorf("failed to create bot: %w", err)
	}

	handlers.Register(b)

	log.Printf("Bot started (only answering username: @%s)", policy.AllowedUsername())
	go b.Start(ctx)

	<-ctx.Done()
	log.Println("Bot stopped")
	return nil
}
