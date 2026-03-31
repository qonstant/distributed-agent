package usecase

import (
	"context"
	"errors"
	"testing"

	"github.com/qonstant/distributed-agent/internal/domain/access"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

func TestGetStartMessageExecute(t *testing.T) {
	t.Parallel()

	uc := GetStartMessage{Policy: access.NewPolicy(fakeAccessDirectory{
		findFn: func(context.Context, int64) (access.Record, bool, error) {
			return access.Record{TelegramID: 42, HasAccess: true}, true, nil
		},
	})}

	text, err := uc.Execute(context.Background(), access.User{TelegramID: 42, Username: "allowed"})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if text == "" {
		t.Fatal("Execute() returned empty text")
	}
}

func TestGetHelpMessageExecuteRejectsUnauthorizedUser(t *testing.T) {
	t.Parallel()

	uc := GetHelpMessage{Policy: access.NewPolicy(fakeAccessDirectory{
		findFn: func(context.Context, int64) (access.Record, bool, error) {
			return access.Record{}, false, nil
		},
	})}
	_, err := uc.Execute(context.Background(), access.User{TelegramID: 77, Username: "denied"})
	if !errors.Is(err, access.ErrUnauthorized) {
		t.Fatalf("Execute() error = %v, want %v", err, access.ErrUnauthorized)
	}
}

func TestGetSampleAttachmentsExecute(t *testing.T) {
	t.Parallel()

	refs := []qa.AttachmentRef{{Source: "photo.png", Kind: qa.AttachmentPhoto}}
	wantAttachments := []qa.Attachment{{Name: "photo.png", Kind: qa.AttachmentPhoto, Content: []byte("img")}}

	uc := GetSampleAttachments{
		Policy: access.NewPolicy(fakeAccessDirectory{
			findFn: func(context.Context, int64) (access.Record, bool, error) {
				return access.Record{TelegramID: 42, HasAccess: true}, true, nil
			},
		}),
		AttachmentRefs: refs,
		Attachments: fakeAttachmentResolver{
			resolveFn: func(_ context.Context, gotRefs []qa.AttachmentRef) ([]qa.Attachment, error) {
				if len(gotRefs) != len(refs) || gotRefs[0].Source != refs[0].Source {
					t.Fatalf("Resolve() refs = %#v, want %#v", gotRefs, refs)
				}
				return wantAttachments, nil
			},
		},
		Title: "Album title",
	}

	response, err := uc.Execute(context.Background(), access.User{TelegramID: 42, Username: "allowed"})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if got, want := response.Text, "Album title"; got != want {
		t.Fatalf("response.Text = %q, want %q", got, want)
	}
	if len(response.Attachments) != 1 || response.Attachments[0].Name != "photo.png" {
		t.Fatalf("response.Attachments = %#v, want %#v", response.Attachments, wantAttachments)
	}
}
