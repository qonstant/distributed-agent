package rabbitmq

import (
	"context"
	"encoding/json"
	"fmt"

	amqp "github.com/rabbitmq/amqp091-go"

	"github.com/qonstant/distributed-agent/internal/domain/persistence"
)

type TurnConsumer struct {
	conn      *amqp.Connection
	channel   *amqp.Channel
	queueName string
}

func NewTurnConsumer(rabbitURL, queueName string) (*TurnConsumer, error) {
	conn, err := amqp.Dial(rabbitURL)
	if err != nil {
		return nil, fmt.Errorf("dial rabbitmq: %w", err)
	}

	channel, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("open rabbitmq channel: %w", err)
	}

	if _, err := channel.QueueDeclare(
		queueName,
		true,
		false,
		false,
		false,
		nil,
	); err != nil {
		channel.Close()
		conn.Close()
		return nil, fmt.Errorf("declare queue %q: %w", queueName, err)
	}

	return &TurnConsumer{
		conn:      conn,
		channel:   channel,
		queueName: queueName,
	}, nil
}

func (c *TurnConsumer) Close() error {
	if c == nil {
		return nil
	}

	var firstErr error
	if c.channel != nil {
		if err := c.channel.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if c.conn != nil {
		if err := c.conn.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (c *TurnConsumer) Consume(ctx context.Context, handler func(context.Context, persistence.TurnEvent) error) error {
	if c == nil || c.channel == nil {
		return fmt.Errorf("rabbitmq consumer is not initialized")
	}

	if err := c.channel.Qos(1, 0, false); err != nil {
		return fmt.Errorf("set rabbitmq qos: %w", err)
	}

	deliveries, err := c.channel.Consume(
		c.queueName,
		"",
		false,
		false,
		false,
		false,
		nil,
	)
	if err != nil {
		return fmt.Errorf("consume queue %q: %w", c.queueName, err)
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case delivery, ok := <-deliveries:
			if !ok {
				return fmt.Errorf("rabbitmq deliveries closed")
			}

			var event persistence.TurnEvent
			if err := json.Unmarshal(delivery.Body, &event); err != nil {
				_ = delivery.Ack(false)
				continue
			}

			if err := handler(ctx, event); err != nil {
				_ = delivery.Nack(false, true)
				continue
			}

			if err := delivery.Ack(false); err != nil {
				return fmt.Errorf("ack delivery: %w", err)
			}
		}
	}
}
