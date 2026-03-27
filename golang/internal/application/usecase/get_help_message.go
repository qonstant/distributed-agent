package usecase

import (
	"context"

	"github.com/qonstant/distributed-agent/internal/domain/access"
)

type GetHelpMessage struct {
	Policy access.Policy
}

func (uc GetHelpMessage) Execute(ctx context.Context, user access.User) (string, error) {
	if err := uc.Policy.Authorize(ctx, user); err != nil {
		return "", err
	}

	return "Просто отправь текст (например: \"Как податься внж?\") — бот запросит локальный API и вернёт ответ + документ (если API вернул файл).", nil
}
