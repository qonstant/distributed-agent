package qa

type DraftResponse struct {
	Text           string
	AttachmentRefs []AttachmentRef
}

type Response struct {
	Text        string
	Attachments []Attachment
}
