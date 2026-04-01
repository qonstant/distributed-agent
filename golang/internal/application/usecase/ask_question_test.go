package usecase

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/qonstant/distributed-agent/internal/domain/access"
	"github.com/qonstant/distributed-agent/internal/domain/persistence"
	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type fakeAnswerSource struct {
	askFn                      func(context.Context, qa.Question) (qa.DraftResponse, error)
	askWithConversationFn      func(context.Context, qa.Question, string) (qa.DraftResponse, error)
	askWithConversationNamedFn func(context.Context, qa.Question, string, string) (qa.DraftResponse, error)
}

func (f fakeAnswerSource) Ask(ctx context.Context, question qa.Question) (qa.DraftResponse, error) {
	return f.askFn(ctx, question)
}

func (f fakeAnswerSource) AskWithConversation(ctx context.Context, question qa.Question, conversationID, preferredName string) (qa.DraftResponse, error) {
	if f.askWithConversationNamedFn != nil {
		return f.askWithConversationNamedFn(ctx, question, conversationID, preferredName)
	}
	if f.askWithConversationFn != nil {
		return f.askWithConversationFn(ctx, question, conversationID)
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
	rememberTurnFn func(context.Context, int64, string, string, string, []qa.ConversationAttachment) error
}

func (f fakeConversationMemory) Context(ctx context.Context, ownerID int64) (qa.ConversationContext, error) {
	return f.contextFn(ctx, ownerID)
}

func (f fakeConversationMemory) RememberTurn(
	ctx context.Context,
	ownerID int64,
	conversationID, userText, assistantText string,
	assistantAttachments []qa.ConversationAttachment,
) error {
	return f.rememberTurnFn(ctx, ownerID, conversationID, userText, assistantText, assistantAttachments)
}

type fakeTurnEventPublisher struct {
	publishFn func(context.Context, persistence.TurnEvent) error
}

func (f fakeTurnEventPublisher) PublishTurn(ctx context.Context, event persistence.TurnEvent) error {
	return f.publishFn(ctx, event)
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
					return access.Record{TelegramID: telegramID, Username: "Stored Name"}, true, nil
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

	t.Run("asks how to call user when preferred name is missing", func(t *testing.T) {
		t.Parallel()

		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID}, true, nil
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
		}

		response, err := uc.Execute(context.Background(), authorizedUser, "hello")
		if err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		if got, want := response.Text, "answer\n\n"+preferredNamePrompt; got != want {
			t.Fatalf("response.Text = %q, want %q", got, want)
		}
	})

	t.Run("stores preferred name from classification action", func(t *testing.T) {
		t.Parallel()

		var published persistence.TurnEvent
		asked := false
		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID}, true, nil
				},
			}),
			Answers: fakeAnswerSource{
				askFn: func(context.Context, qa.Question) (qa.DraftResponse, error) {
					asked = true
					return qa.DraftResponse{
						Classification: &qa.MessageClassification{
							Intent:        "CHIT_CHAT",
							ProfileAction: "set_preferred_name",
							PreferredName: "Test User",
							DetectedLanguage: "en",
						},
					}, nil
				},
			},
			Attachments: fakeAttachmentResolver{
				resolveFn: func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error) {
					t.Fatal("Resolve should not be called")
					return nil, nil
				},
			},
			TurnEvents: fakeTurnEventPublisher{
				publishFn: func(_ context.Context, event persistence.TurnEvent) error {
					published = event
					return nil
				},
			},
		}

		response, err := uc.Execute(context.Background(), authorizedUser, "Call me Test User")
		if err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		if !asked {
			t.Fatal("Ask should be called so the classifier can extract the preferred name")
		}
		if got, want := response.Text, "Nice to meet you, Test User! I'll call you that."; got != want {
			t.Fatalf("response.Text = %q, want %q", got, want)
		}
		if got, want := published.User.Username, "Test User"; got != want {
			t.Fatalf("published.User.Username = %q, want %q", got, want)
		}
	})

	t.Run("updates preferred name from classification rename action", func(t *testing.T) {
		t.Parallel()

		var published persistence.TurnEvent
		asked := false
		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, Username: "Роман"}, true, nil
				},
			}),
			Answers: fakeAnswerSource{
				askFn: func(context.Context, qa.Question) (qa.DraftResponse, error) {
					asked = true
					return qa.DraftResponse{
						Classification: &qa.MessageClassification{
							Intent:        "CHIT_CHAT",
							ProfileAction: "set_preferred_name",
							PreferredName: "Heisenberg",
							DetectedLanguage: "ru",
						},
					}, nil
				},
			},
			Attachments: fakeAttachmentResolver{
				resolveFn: func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error) {
					t.Fatal("Resolve should not be called")
					return nil, nil
				},
			},
			TurnEvents: fakeTurnEventPublisher{
				publishFn: func(_ context.Context, event persistence.TurnEvent) error {
					published = event
					return nil
				},
			},
		}

		response, err := uc.Execute(context.Background(), authorizedUser, "Неа, зовут меня теперь Heisenberg")
		if err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		if !asked {
			t.Fatal("Ask should be called so the classifier can detect the rename")
		}
		if got, want := response.Text, "Приятно познакомиться, Heisenberg! Буду звать тебя так."; got != want {
			t.Fatalf("response.Text = %q, want %q", got, want)
		}
		if got, want := published.User.Username, "Heisenberg"; got != want {
			t.Fatalf("published.User.Username = %q, want %q", got, want)
		}
	})

	t.Run("resolves attachments from draft response", func(t *testing.T) {
		t.Parallel()

		var gotRefs []qa.AttachmentRef
		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, Username: "Stored Name"}, true, nil
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
					return access.Record{TelegramID: authorizedUser.TelegramID, Username: "Stored Name"}, true, nil
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
					return access.Record{TelegramID: authorizedUser.TelegramID, Username: "Stored Name"}, true, nil
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
		remembered := false
		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, Username: "Stored Name"}, true, nil
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
			Memory: fakeConversationMemory{
				contextFn: func(context.Context, int64) (qa.ConversationContext, error) {
					return qa.ConversationContext{ID: "conv-fail"}, nil
				},
				rememberTurnFn: func(context.Context, int64, string, string, string, []qa.ConversationAttachment) error {
					remembered = true
					return nil
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
		if remembered {
			t.Fatal("RememberTurn() should not be called when attachment resolution fails")
		}
	})

	t.Run("passes conversation id to answer source and remembers turn", func(t *testing.T) {
		t.Parallel()

		var remembered struct {
			ownerID        int64
			conversationID string
			userText       string
			assistantText  string
		}
		var published persistence.TurnEvent
		now := time.Unix(1774920000, 0).UTC()

		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, Username: "Stored Name"}, true, nil
				},
			}),
			Answers: fakeAnswerSource{
				askFn: func(context.Context, qa.Question) (qa.DraftResponse, error) {
					t.Fatal("Ask should not be called when AskWithConversation is available")
					return qa.DraftResponse{}, nil
				},
				askWithConversationNamedFn: func(_ context.Context, question qa.Question, conversationID, preferredName string) (qa.DraftResponse, error) {
					if got, want := question.Text, "hello"; got != want {
						t.Fatalf("question.Text = %q, want %q", got, want)
					}
					if got, want := conversationID, "conv-1"; got != want {
						t.Fatalf("conversationID = %q, want %q", got, want)
					}
					if got, want := preferredName, "Stored Name"; got != want {
						t.Fatalf("preferredName = %q, want %q", got, want)
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
					return qa.ConversationContext{ID: "conv-1"}, nil
				},
				rememberTurnFn: func(
					_ context.Context,
					ownerID int64,
					conversationID, userText, assistantText string,
					assistantAttachments []qa.ConversationAttachment,
				) error {
					remembered.ownerID = ownerID
					remembered.conversationID = conversationID
					remembered.userText = userText
					remembered.assistantText = assistantText
					if len(assistantAttachments) != 0 {
						t.Fatalf("assistantAttachments = %#v, want empty", assistantAttachments)
					}
					return nil
				},
			},
			TurnEvents: fakeTurnEventPublisher{
				publishFn: func(_ context.Context, event persistence.TurnEvent) error {
					published = event
					return nil
				},
			},
			Now: func() time.Time { return now },
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
		if got, want := published.Conversation.Key, "conv-1"; got != want {
			t.Fatalf("published.Conversation.Key = %q, want %q", got, want)
		}
		if got, want := published.UserMessage.Text, "hello"; got != want {
			t.Fatalf("published.UserMessage.Text = %q, want %q", got, want)
		}
		if got, want := published.User.Username, "Stored Name"; got != want {
			t.Fatalf("published.User.Username = %q, want %q", got, want)
		}
	})

	t.Run("continues when conversation memory is unavailable", func(t *testing.T) {
		t.Parallel()

		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, Username: "Stored Name"}, true, nil
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
				rememberTurnFn: func(context.Context, int64, string, string, string, []qa.ConversationAttachment) error {
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

	t.Run("remembers resolved attachment metadata", func(t *testing.T) {
		t.Parallel()

		var rememberedAttachments []qa.ConversationAttachment
		uc := AskQuestion{
			Policy: access.NewPolicy(fakeAccessDirectory{
				findFn: func(context.Context, int64) (access.Record, bool, error) {
					return access.Record{TelegramID: authorizedUser.TelegramID, Username: "Stored Name"}, true, nil
				},
			}),
			Answers: fakeAnswerSource{
				askFn: func(context.Context, qa.Question) (qa.DraftResponse, error) {
					return qa.DraftResponse{
						Text: "Here is the sample.",
						AttachmentRefs: []qa.AttachmentRef{
							{Source: "docs/sample.pdf", Kind: qa.AttachmentDocument},
							{Source: "photos/example.png", Kind: qa.AttachmentPhoto},
						},
					}, nil
				},
			},
			Attachments: fakeAttachmentResolver{
				resolveFn: func(context.Context, []qa.AttachmentRef) ([]qa.Attachment, error) {
					return []qa.Attachment{
						{Name: "sample.pdf", Kind: qa.AttachmentDocument, Content: []byte("pdf")},
						{Name: "example.png", Kind: qa.AttachmentPhoto, Content: []byte("img")},
					}, nil
				},
			},
			Memory: fakeConversationMemory{
				contextFn: func(context.Context, int64) (qa.ConversationContext, error) {
					return qa.ConversationContext{ID: "conv-2"}, nil
				},
				rememberTurnFn: func(
					_ context.Context,
					_ int64,
					_ string,
					_ string,
					_ string,
					assistantAttachments []qa.ConversationAttachment,
				) error {
					rememberedAttachments = append([]qa.ConversationAttachment(nil), assistantAttachments...)
					return nil
				},
			},
		}

		response, err := uc.Execute(context.Background(), authorizedUser, "send sample")
		if err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		if len(response.Attachments) != 2 {
			t.Fatalf("len(response.Attachments) = %d, want 2", len(response.Attachments))
		}
		if len(rememberedAttachments) != 2 {
			t.Fatalf("len(rememberedAttachments) = %d, want 2", len(rememberedAttachments))
		}
		if got, want := rememberedAttachments[0], (qa.ConversationAttachment{Source: "docs/sample.pdf", Name: "sample.pdf", Kind: qa.AttachmentDocument}); got != want {
			t.Fatalf("rememberedAttachments[0] = %#v, want %#v", got, want)
		}
		if got, want := rememberedAttachments[1], (qa.ConversationAttachment{Source: "photos/example.png", Name: "example.png", Kind: qa.AttachmentPhoto}); got != want {
			t.Fatalf("rememberedAttachments[1] = %#v, want %#v", got, want)
		}
	})
}
