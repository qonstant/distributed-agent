package usecase

import (
	"context"

	"github.com/qonstant/distributed-agent/internal/application/port"
	"github.com/qonstant/distributed-agent/internal/domain/access"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type GetSampleAttachments struct {
	Policy         access.Policy
	AttachmentRefs []qa.AttachmentRef
	Attachments    port.AttachmentResolver
	Title          string
}

func (uc GetSampleAttachments) Execute(ctx context.Context, user access.User) (qa.Response, error) {
	if err := uc.Policy.Authorize(user); err != nil {
		return qa.Response{}, err
	}

	attachments, err := uc.Attachments.Resolve(ctx, uc.AttachmentRefs)
	if err != nil {
		return qa.Response{}, err
	}

	return qa.Response{
		Text:        uc.Title,
		Attachments: attachments,
	}, nil
}
