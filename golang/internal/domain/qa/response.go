package qa

type DraftResponse struct {
	Text           string
	AttachmentRefs []AttachmentRef
	Classification *MessageClassification
	UsageEvents    []UsageEvent
}

type Response struct {
	Text        string
	Attachments []Attachment
}

type MessageClassification struct {
	Intent            string
	Explain           string
	DetectedLanguage  string
	ClassifierModel   string
	ClassifierVersion string
}

type UsageEvent struct {
	EventType     string
	InputTokens   int
	OutputTokens  int
	TotalTokens   int
	EstimatedCost float64
}
