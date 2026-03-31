package qa

const (
	ConversationRoleUser      = "user"
	ConversationRoleAssistant = "assistant"
)

type ConversationMessage struct {
	Role      string `json:"role"`
	Text      string `json:"text"`
	Timestamp int64  `json:"ts"`
}

type ConversationContext struct {
	ID string
}
