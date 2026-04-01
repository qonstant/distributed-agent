package persistence

import "time"

const TurnEventsQueueName = "agent.turn_persistence"

type TurnEvent struct {
	Version          int                    `json:"version"`
	User             User                   `json:"user"`
	Conversation     Conversation           `json:"conversation"`
	UserMessage      Message                `json:"user_message"`
	AssistantMessage Message                `json:"assistant_message"`
	Classification   *MessageClassification `json:"classification,omitempty"`
	UsageEvents      []UsageEvent           `json:"usage_events,omitempty"`
}

type User struct {
	TelegramID  int64  `json:"telegram_id"`
	Username    string `json:"username,omitempty"`
	DisplayName string `json:"display_name,omitempty"`
}

type Conversation struct {
	Key       string    `json:"key"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type Message struct {
	Role              string    `json:"role"`
	Text              string    `json:"text"`
	LanguageCode      string    `json:"language_code,omitempty"`
	TelegramMessageID *int64    `json:"telegram_message_id,omitempty"`
	CreatedAt         time.Time `json:"created_at"`
}

type MessageClassification struct {
	Intent            string    `json:"intent"`
	Explain           string    `json:"explain,omitempty"`
	DetectedLanguage  string    `json:"detected_language,omitempty"`
	ClassifierModel   string    `json:"classifier_model,omitempty"`
	ClassifierVersion string    `json:"classifier_version,omitempty"`
	CreatedAt         time.Time `json:"created_at"`
}

type UsageEvent struct {
	EventType     string    `json:"event_type"`
	InputTokens   int       `json:"input_tokens"`
	OutputTokens  int       `json:"output_tokens"`
	TotalTokens   int       `json:"total_tokens"`
	EstimatedCost float64   `json:"estimated_cost"`
	CreatedAt     time.Time `json:"created_at"`
}
