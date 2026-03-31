package usecase

import (
	"context"
	"errors"
	"testing"

	"github.com/qonstant/distributed-agent/internal/domain/access"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type fakeAnswerSource struct {
	askFn            func(context.Context, qa.Question) (qa.DraftResponse, error)
	askWithHistoryFn func(context.Context, qa.Question, []qa.ConversationMessage) (qa.DraftResponse, error)
}

func (f fakeAnswerSource) Ask(ctx context.Context, question qa.Question) (qa.DraftResponse, error) {
	return f.askFn(ctx, question)
}

func (f fakeAnswerSource) AskWithHistory(ctx context.Context, question qa.Question, history []qa.ConversationMessage) (qa.DraftResponse, error) {
	if f.askWithHistoryFn != nil {
		return f.askWithHistoryFn(ctx, question, history)
	}
	return f.Ask(ctx, question)
}

type fakeAttachmentResolver struct {
	resolveFn func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error)
}

func (f fakeAttachmentResolver) Resolve(ctx context.Context, refs []qa.AttachmentRef) ([]qa.Attachment, error) {
	return f.resolveFn(ctx, refs)
}

type fakeAccessDirectory struct {
	findFn func(context.Context, int64) (access.Record, bool, error)
}

func (f fakeAccessDirectory) FindByTelegramID(ctx context.Context, telegramID int64) (access.Record, bool, error) {
	return f.findFn(ctx, telegramID)
}

type fakeConversationMemory struct {
	contextFn      func(context.Context, int64) (qa.ConversationContext, error)
	rememberTurnFn func(context.Context, int64, string, string, string) error
}

func (f fakeConversationMemory) Context(ctx context.Context, ownerID int64) (qa.ConversationContext, error) {
	return f.contextFn(ctx, ownerID)
}

func (f fakeConversationMemory) RememberTurn(ctx context.Context, ownerID int64, conversationID, userText, assistantText string) error {
	return f.rememberTurnFn(ctx, ownerID, conversationID, userText, assistantText)
}

func TestAskQuestionExecute(t *testing.T) {
	t.Parallel()

	authorizedUser := access.User{TelegramID: 42, Username: "allowed"}

	t.Run("rejects unauthorized user", func(t *testing.T) {
		t.Parallel()

		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{}, false, nil
				},
			}),
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

		_, err := uc.Execute(context.Background(), access.User{TelegramID: 77, Username: "denied"}, "hello")
		if !errors.Is(err, access.ErrUnauthorized) {
			t.Fatalf("Execute() error = %v, want %v", err, access.ErrUnauthorized)
		}
	})

	t.Run("returns text only response", func(t *testing.T) {
		t.Parallel()

		asked := false
		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(_ context.Context, telegramID int64) (access.Record, bool, error) {
					if got, want := telegramID, authorizedUser.TelegramID; got != want {
						t.Fatalf("FindByTelegramID() telegramID = %d, want %d", got, want)
					}
					return access.Record{TelegramID: telegramID, HasAccess: true}, true, nil
				},
			}),
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
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, HasAccess: true}, true, nil
				},
			}),
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
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, HasAccess: true}, true, nil
				},
			}),
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
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, HasAccess: true}, true, nil
				},
			}),
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
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, HasAccess: true}, true, nil
				},
			}),
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

	t.Run("passes history to contextual answer source and remembers turn", func(t *testing.T) {
		t.Parallel()

		history := []qa.ConversationMessage{
			{Role: qa.ConversationRoleUser, Text: "old question", Timestamp: 1},
			{Role: qa.ConversationRoleAssistant, Text: "old answer", Timestamp: 2},
		}

		var remembered struct {
			ownerID        int64
			conversationID string
			userText       string
			assistantText  string
		}

		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, HasAccess: true}, true, nil
				},
			}),
			Answers: fakeAnswerSource{
				askFn: func(context.Context, qa.Question) (qa.DraftResponse, error) {
					t.Fatal("Ask should not be called when AskWithHistory is available")
					return qa.DraftResponse{}, nil
				},
				askWithHistoryFn: func(_ context.Context, question qa.Question, gotHistory []qa.ConversationMessage) (qa.DraftResponse, error) {
					if got, want := question.Text, "hello"; got != want {
						t.Fatalf("question.Text = %q, want %q", got, want)
					}
					if len(gotHistory) != len(history) {
						t.Fatalf("len(history) = %d, want %d", len(gotHistory), len(history))
					}
					for i := range history {
						if gotHistory[i] != history[i] {
							t.Fatalf("history[%d] = %+v, want %+v", i, gotHistory[i], history[i])
						}
					}
					return qa.DraftResponse{Text: "new answer"}, nil
				},
			},
			Attachments: fakeAttachmentResolver{
				resolveFn: func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error) {
					t.Fatal("Resolve should not be called")
					return nil, nil
				},
			},
			Memory: fakeConversationMemory{
				contextFn: func(_ context.Context, ownerID int64) (qa.ConversationContext, error) {
					if got, want := ownerID, authorizedUser.TelegramID; got != want {
						t.Fatalf("Context() ownerID = %d, want %d", got, want)
					}
					return qa.ConversationContext{
						ID:       "conv-1",
						Messages: history,
					}, nil
				},
				rememberTurnFn: func(_ context.Context, ownerID int64, conversationID, userText, assistantText string) error {
					remembered.ownerID = ownerID
					remembered.conversationID = conversationID
					remembered.userText = userText
					remembered.assistantText = assistantText
					return nil
				},
			},
		}

		response, err := uc.Execute(context.Background(), authorizedUser, "hello")
		if err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		if got, want := response.Text, "new answer"; got != want {
			t.Fatalf("response.Text = %q, want %q", got, want)
		}
		if got, want := remembered.ownerID, authorizedUser.TelegramID; got != want {
			t.Fatalf("remembered.ownerID = %d, want %d", got, want)
		}
		if got, want := remembered.conversationID, "conv-1"; got != want {
			t.Fatalf("remembered.conversationID = %q, want %q", got, want)
		}
		if got, want := remembered.userText, "hello"; got != want {
			t.Fatalf("remembered.userText = %q, want %q", got, want)
		}
		if got, want := remembered.assistantText, "new answer"; got != want {
			t.Fatalf("remembered.assistantText = %q, want %q", got, want)
		}
	})

	t.Run("continues when conversation memory is unavailable", func(t *testing.T) {
		t.Parallel()

		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, HasAccess: true}, true, nil
				},
			}),
			Answers: fakeAnswerSource{
				askFn: func(context.Context, qa.Question) (qa.DraftResponse, error) {
					return qa.DraftResponse{Text: "answer"}, nil
				},
			},
			Attachments: fakeAttachmentResolver{
				resolveFn: func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error) {
					t.Fatal("Resolve should not be called")
					return nil, nil
				},
			},
			Memory: fakeConversationMemory{
				contextFn: func(context.Context, int64) (qa.ConversationContext, error) {
					return qa.ConversationContext{}, errors.New("redis unavailable")
				},
				rememberTurnFn: func(context.Context, int64, string, string, string) error {
					return errors.New("redis unavailable")
				},
			},
		}

		response, err := uc.Execute(context.Background(), authorizedUser, "hello")
		if err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		if got, want := response.Text, "answer"; got != want {
			t.Fatalf("response.Text = %q, want %q", got, want)
		}
	})
}
