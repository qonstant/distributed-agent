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
	"github.com/qonstant/distributed-agent/internal/adapter/accesscache"
	"github.com/qonstant/distributed-agent/internal/adapter/chatmemory"
	"github.com/qonstant/distributed-agent/internal/adapter/localapi"
	"github.com/qonstant/distributed-agent/internal/adapter/postgres"
	"github.com/qonstant/distributed-agent/internal/adapter/rabbitmq"
	"github.com/qonstant/distributed-agent/internal/adapter/storage"
	telegramadapter "github.com/qonstant/distributed-agent/internal/adapter/telegram"
	"github.com/qonstant/distributed-agent/internal/application/port"
	"github.com/qonstant/distributed-agent/internal/application/usecase"
	"github.com/qonstant/distributed-agent/internal/config"
	"github.com/qonstant/distributed-agent/internal/domain/access"
	"github.com/qonstant/distributed-agent/internal/domain/persistence"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

func Run() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	if err := cfg.ValidateBot(); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	accessDirectory, err := postgres.NewAccessDirectory(cfg.DBURL)
	if err != nil {
		return fmt.Errorf("failed to initialize access directory: %w", err)
	}
	defer accessDirectory.Close()

	var directory access.Directory = accessDirectory
	var memory port.ConversationMemory
	var turnEvents port.TurnEventPublisher
	if cfg.Redis.URL != "" {
		redisStore, err := accesscache.NewRedisStore(cfg.Redis.URL)
		if err != nil {
			log.Printf("Redis access cache warning (continuing without cache): %v", err)
		} else {
			defer redisStore.Close()
			directory = accesscache.NewCachedAccessDirectory(directory, redisStore, accesscache.CachedAccessDirectoryConfig{
				TTL:         cfg.Redis.AccessCacheTTL,
				NegativeTTL: cfg.Redis.NegativeCacheTTL,
			})
			log.Printf(
				"Access cache enabled (redis, ttl=%s negative_ttl=%s)",
				cfg.Redis.AccessCacheTTL,
				cfg.Redis.NegativeCacheTTL,
			)
		}

		memoryStore, err := chatmemory.NewRedisStore(cfg.Redis.URL)
		if err != nil {
			log.Printf("Redis conversation memory warning (continuing without memory): %v", err)
		} else {
			defer memoryStore.Close()
			memory = chatmemory.New(memoryStore, chatmemory.Config{
				TTL:               cfg.Redis.ConversationMemoryTTL,
				MaxItems:          int64(cfg.Redis.ConversationMemoryMaxItems),
				AssistantMaxChars: cfg.Redis.ConversationMemoryAssistantMaxChars,
			})
			log.Printf(
				"Conversation memory enabled (redis, ttl=%s max_items=%d assistant_max_chars=%d)",
				cfg.Redis.ConversationMemoryTTL,
				cfg.Redis.ConversationMemoryMaxItems,
				cfg.Redis.ConversationMemoryAssistantMaxChars,
			)
		}
	}
	if cfg.RabbitMQURL != "" {
		publisher, err := rabbitmq.NewTurnPublisher(cfg.RabbitMQURL, persistence.TurnEventsQueueName)
		if err != nil {
			log.Printf("RabbitMQ turn publisher warning (continuing without turn events): %v", err)
		} else {
			defer publisher.Close()
			turnEvents = publisher
			log.Printf("Turn persistence events enabled (rabbitmq, queue=%s)", persistence.TurnEventsQueueName)
		}
	}

	policy := access.NewPolicy(directory)

	s3Store, err := storage.NewS3Store(context.Background(), cfg.S3)
	if err != nil {
		log.Printf("S3 init warning (continuing): %v", err)
	}

	resolver := storage.NewResolver(cfg.DocRoot, s3Store)
	answerSource := localapi.NewClientWithConfig(cfg.LocalAPIURL, localapi.ClientConfig{
		Timeout:      cfg.LocalAPITimeout,
		MaxAttempts:  cfg.LocalAPIMaxAttempts,
		RetryBackoff: cfg.LocalAPIRetryBackoff,
	})
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
			Memory:      memory,
			TurnEvents:  turnEvents,
		},
		usecase.GetSampleAttachments{
			Policy:         policy,
			AttachmentRefs: sampleRefs,
			Attachments:    resolver,
			Title:          cfg.SampleAlbumTitle,
		},
		presenter,
		"This bot accepts requests only for users with active access.\n\nPlease contact an administrator to request access.",
	)

	b, err := bot.New(cfg.TelegramToken, bot.WithDefaultHandler(handlers.HandleDefault))
	if err != nil {
		return fmt.Errorf("failed to create bot: %w", err)
	}

	handlers.Register(b)

	log.Printf("Bot started (access control: %s)", policy.AccessMode())
	go b.Start(ctx)

	<-ctx.Done()
	log.Println("Bot stopped")
	return nil
}
