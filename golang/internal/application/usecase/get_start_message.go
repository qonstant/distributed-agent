package usecase

import "github.com/qonstant/distributed-agent/internal/domain/access"

type GetStartMessage struct {
	Policy access.Policy
}

func (uc GetStartMessage) Execute(user access.User) (string, error) {
	if err := uc.Policy.Authorize(user); err != nil {
		return "", err
	}

	return "Привет! Отправь любой вопрос (на любом языке) — бот перешлёт запрос локальному API и пришлёт ответ + файл справки, если он есть.", nil
}
