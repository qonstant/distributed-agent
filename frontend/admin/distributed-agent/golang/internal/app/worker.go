package app

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/qonstant/distributed-agent/internal/adapter/postgres"
	"github.com/qonstant/distributed-agent/internal/adapter/rabbitmq"
	"github.com/qonstant/distributed-agent/internal/config"
	"github.com/qonstant/distributed-agent/internal/domain/persistence"
)

func RunTurnWorker() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	if err := cfg.ValidateWorker(); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	store, err := postgres.NewTurnStore(cfg.DBURL)
	if err != nil {
		return fmt.Errorf("initialize turn store: %w", err)
	}
	defer store.Close()

	consumer, err := rabbitmq.NewTurnConsumer(cfg.RabbitMQURL, persistence.TurnEventsQueueName)
	if err != nil {
		return fmt.Errorf("initialize rabbitmq turn consumer: %w", err)
	}
	defer consumer.Close()

	log.Printf("Turn worker started (queue=%s)", persistence.TurnEventsQueueName)
	err = consumer.Consume(ctx, func(ctx context.Context, event persistence.TurnEvent) error {
		if err := store.SaveTurn(ctx, event); err != nil {
			log.Printf("turn worker save failed: %v", err)
			return err
		}
		return nil
	})
	if err != nil && err != context.Canceled {
		return err
	}

	log.Println("Turn worker stopped")
	return nil
}
