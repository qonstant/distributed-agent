package usecase

import (
	"context"
	"errors"
	"testing"

	"github.com/qonstant/distributed-agent/internal/domain/access"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type fakeAnswerSource struct {
	askFn func(context.Context, qa.Question) (qa.DraftResponse, error)
}

func (f fakeAnswerSource) Ask(ctx context.Context, question qa.Question) (qa.DraftResponse, error) {
	return f.askFn(ctx, question)
}

type fakeAttachmentResolver struct {
	resolveFn func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error)
}

func (f fakeAttachmentResolver) Resolve(ctx context.Context, refs []qa.AttachmentRef) ([]qa.Attachment, error) {
	return f.resolveFn(ctx, refs)
}

func TestAskQuestionExecute(t *testing.T) {
	t.Parallel()

	authorizedUser := access.User{Username: "allowed"}

	t.Run("rejects unauthorized user", func(t *testing.T) {
		t.Parallel()

		uc := AskQuestion{
			Policy: access.NewPolicy("allowed"),
			Answers: fakeAnswerSource{
				askFn: func(context.Context, qa.Question) (qa.DraftResponse, error) {
					t.Fatal("Ask should not be called")
					return qa.DraftResponse{}, nil
				},
			},
			Attachments: fakeAttachmentResolver{
				resolveFn: func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error) {
					t.Fatal("Resolve should not be called")
					return nil, nil
				},
			},
		}

		_, err := uc.Execute(context.Background(), access.User{Username: "denied"}, "hello")
		if !errors.Is(err, access.ErrUnauthorized) {
			t.Fatalf("Execute() error = %v, want %v", err, access.ErrUnauthorized)
		}
	})

	t.Run("returns text only response", func(t *testing.T) {
		t.Parallel()

		asked := false
		uc := AskQuestion{
			Policy: access.NewPolicy("allowed"),
			Answers: fakeAnswerSource{
				askFn: func(_ context.Context, question qa.Question) (qa.DraftResponse, error) {
					asked = true
					if got, want := question.Text, "hello"; got != want {
						t.Fatalf("question.Text = %q, want %q", got, want)
					}
					return qa.DraftResponse{Text: "answer"}, nil
				},
			},
			Attachments: fakeAttachmentResolver{
				resolveFn: func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error) {
					t.Fatal("Resolve should not be called")
					return nil, nil
				},
			},
		}

		response, err := uc.Execute(context.Background(), authorizedUser, " hello ")
		if err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		if !asked {
			t.Fatal("Ask was not called")
		}
		if got, want := response.Text, "answer"; got != want {
			t.Fatalf("response.Text = %q, want %q", got, want)
		}
		if len(response.Attachments) != 0 {
			t.Fatalf("len(response.Attachments) = %d, want 0", len(response.Attachments))
		}
	})

	t.Run("resolves attachments from draft response", func(t *testing.T) {
		t.Parallel()

		var gotRefs []qa.AttachmentRef
		uc := AskQuestion{
			Policy: access.NewPolicy("allowed"),
			Answers: fakeAnswerSource{
				askFn: func(context.Context, qa.Question) (qa.DraftResponse, error) {
					return qa.DraftResponse{
						Text: "answer",
						AttachmentRefs: []qa.AttachmentRef{
							{Source: "file.pdf", Kind: qa.AttachmentDocument},
						},
					}, nil
				},
			},
			Attachments: fakeAttachmentResolver{
				resolveFn: func(_ context.Context, refs []qa.AttachmentRef) ([]qa.Attachment, error) {
					gotRefs = refs
					return []qa.Attachment{
						{Name: "file.pdf", Kind: qa.AttachmentDocument, Content: []byte("pdf")},
					}, nil
				},
			},
		}

		response, err := uc.Execute(context.Background(), authorizedUser, "hello")
		if err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		if len(gotRefs) != 1 || gotRefs[0].Source != "file.pdf" {
			t.Fatalf("Resolve() refs = %#v, want one file.pdf ref", gotRefs)
		}
		if len(response.Attachments) != 1 {
			t.Fatalf("len(response.Attachments) = %d, want 1", len(response.Attachments))
		}
	})

	t.Run("returns error from answer source", func(t *testing.T) {
		t.Parallel()

		wantErr := errors.New("answer source failed")
		uc := AskQuestion{
			Policy: access.NewPolicy("allowed"),
			Answers: fakeAnswerSource{
				askFn: func(context.Context, qa.Question) (qa.DraftResponse, error) {
					return qa.DraftResponse{}, wantErr
				},
			},
			Attachments: fakeAttachmentResolver{
				resolveFn: func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error) {
					t.Fatal("Resolve should not be called")
					return nil, nil
				},
			},
		}

		_, err := uc.Execute(context.Background(), authorizedUser, "hello")
		if !errors.Is(err, wantErr) {
			t.Fatalf("Execute() error = %v, want %v", err, wantErr)
		}
	})

	t.Run("returns question validation error", func(t *testing.T) {
		t.Parallel()

		uc := AskQuestion{
			Policy: access.NewPolicy("allowed"),
			Answers: fakeAnswerSource{
				askFn: func(context.Context, qa.Question) (qa.DraftResponse, error) {
					t.Fatal("Ask should not be called")
					return qa.DraftResponse{}, nil
				},
			},
			Attachments: fakeAttachmentResolver{
				resolveFn: func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error) {
					t.Fatal("Resolve should not be called")
					return nil, nil
				},
			},
		}

		_, err := uc.Execute(context.Background(), authorizedUser, "   ")
		if !errors.Is(err, qa.ErrEmptyQuestion) {
			t.Fatalf("Execute() error = %v, want %v", err, qa.ErrEmptyQuestion)
		}
	})

	t.Run("returns partial response when attachment resolution fails", func(t *testing.T) {
		t.Parallel()

		wantErr := errors.New("resolve failed")
		uc := AskQuestion{
			Policy: access.NewPolicy("allowed"),
			Answers: fakeAnswerSource{
				askFn: func(context.Context, qa.Question) (qa.DraftResponse, error) {
					return qa.DraftResponse{
						Text: "answer",
						AttachmentRefs: []qa.AttachmentRef{
							{Source: "file.pdf", Kind: qa.AttachmentDocument},
						},
					}, nil
				},
			},
			Attachments: fakeAttachmentResolver{
				resolveFn: func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error) {
					return nil, wantErr
				},
			},
		}

		response, err := uc.Execute(context.Background(), authorizedUser, "hello")
		if !errors.Is(err, wantErr) {
			t.Fatalf("Execute() error = %v, want %v", err, wantErr)
		}
		if got, want := response.Text, "answer"; got != want {
			t.Fatalf("response.Text = %q, want %q", got, want)
		}
	})
}
