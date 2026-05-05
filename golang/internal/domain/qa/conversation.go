package qa

const (
	ConversationRoleUser      = "user"
	ConversationRoleAssistant = "assistant"
)

type ConversationMessage struct {
	Role        string                   `json:"role"`
	Text        string                   `json:"text"`
	Timestamp   int64                    `json:"ts"`
	Attachments []ConversationAttachment `json:"attachments,omitempty"`
}

type ConversationAttachment struct {
	Source string         `json:"source,omitempty"`
	Name   string         `json:"name"`
	Kind   AttachmentKind `json:"kind"`
}

type ConversationContext struct {
	ID string
}
