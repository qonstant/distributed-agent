package qa

type AttachmentKind string

const (
	AttachmentDocument AttachmentKind = "document"
	AttachmentPhoto    AttachmentKind = "photo"
)

type AttachmentRef struct {
	Source string
	Kind   AttachmentKind
}

type Attachment struct {
	Name    string
	Kind    AttachmentKind
	Content []byte
}
