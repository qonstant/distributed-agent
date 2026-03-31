package chatmemory

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/qonstant/distributed-agent/internal/domain/qa"
)

type fakeStore struct {
	values    map[string]string
	valueTTLs map[string]time.Duration
	lists     map[string][]string
	listTTLs  map[string]time.Duration
}

func newFakeStore() *fakeStore {
	return &fakeStore{
		values:    make(map[string]string),
		valueTTLs: make(map[string]time.Duration),
		lists:     make(map[string][]string),
		listTTLs:  make(map[string]time.Duration),
	}
}

func (f *fakeStore) Get(_ context.Context, key string) (string, bool, error) {
	value, ok := f.values[key]
	return value, ok, nil
}

func (f *fakeStore) Set(_ context.Context, key, value string, ttl time.Duration) error {
	f.values[key] = value
	f.valueTTLs[key] = ttl
	return nil
}

func (f *fakeStore) Exists(_ context.Context, key string) (bool, error) {
	_, ok := f.lists[key]
	return ok, nil
}

func (f *fakeStore) AppendConversationTurn(
	_ context.Context,
	key, userPayload, assistantPayload string,
	maxItems int64,
	ttl time.Duration,
) error {
	f.lists[key] = append([]string{userPayload}, f.lists[key]...)
	f.lists[key] = append([]string{assistantPayload}, f.lists[key]...)
	if int64(len(f.lists[key])) > maxItems {
		f.lists[key] = f.lists[key][:maxItems]
	}
	f.listTTLs[key] = ttl
	return nil
}

func TestMemoryRememberTurnAndContext(t *testing.T) {
	t.Parallel()

	now := time.Unix(1774920000, 0)
	store := newFakeStore()
	memory := New(store, Config{
		TTL:      2 * time.Hour,
		MaxItems: 8,
		Now: func() time.Time {
			return now
		},
	})

	ctx := context.Background()

	conversation, err := memory.Context(ctx, 42)
	if err != nil {
		t.Fatalf("Context() error = %v", err)
	}
	if conversation.ID == "" {
		t.Fatal("Context().ID is empty")
	}

	if err := memory.RememberTurn(ctx, 42, conversation.ID, "hello", "world", nil); err != nil {
		t.Fatalf("RememberTurn() error = %v", err)
	}

	if got, want := store.values[memory.activeConversationKey(42)], conversation.ID; got != want {
		t.Fatalf("active conversation id = %q, want %q", got, want)
	}
	if got, want := store.valueTTLs[memory.activeConversationKey(42)], 2*time.Hour; got != want {
		t.Fatalf("active conversation ttl = %s, want %s", got, want)
	}
	if got, want := store.listTTLs[memory.messagesKey(conversation.ID)], 2*time.Hour; got != want {
		t.Fatalf("messages ttl = %s, want %s", got, want)
	}

	reloaded, err := memory.Context(ctx, 42)
	if err != nil {
		t.Fatalf("Context() after RememberTurn error = %v", err)
	}
	if got, want := reloaded.ID, conversation.ID; got != want {
		t.Fatalf("Context().ID = %q, want %q", got, want)
	}
}

func TestMemoryKeepsOnlyLatestEightMessages(t *testing.T) {
	t.Parallel()

	current := time.Unix(1774920000, 0)
	store := newFakeStore()
	memory := New(store, Config{
		TTL:      2 * time.Hour,
		MaxItems: 8,
		Now: func() time.Time {
			return current
		},
	})

	ctx := context.Background()
	conversation, err := memory.Context(ctx, 42)
	if err != nil {
		t.Fatalf("Context() error = %v", err)
	}

	for turn := 1; turn <= 5; turn++ {
		current = current.Add(time.Second)
		if err := memory.RememberTurn(
			ctx,
			42,
			conversation.ID,
			fmt.Sprintf("user-%d", turn),
			fmt.Sprintf("assistant-%d", turn),
			nil,
		); err != nil {
			t.Fatalf("RememberTurn() turn %d error = %v", turn, err)
		}
	}

	reloaded, err := memory.Context(ctx, 42)
	if err != nil {
		t.Fatalf("Context() error = %v", err)
	}
	if got := reloaded.ID; got != conversation.ID {
		t.Fatalf("Context().ID = %q, want %q", got, conversation.ID)
	}
	if len(store.lists[memory.messagesKey(conversation.ID)]) != 8 {
		t.Fatalf("len(list) = %d, want 8", len(store.lists[memory.messagesKey(conversation.ID)]))
	}
}

func TestMemoryStartsNewConversationWhenListExpired(t *testing.T) {
	t.Parallel()

	current := time.Unix(1774920000, 0)
	store := newFakeStore()
	memory := New(store, Config{
		TTL:      2 * time.Hour,
		MaxItems: 8,
		Now: func() time.Time {
			return current
		},
	})

	store.values[memory.activeConversationKey(42)] = "stale-conversation"
	current = current.Add(time.Second)

	conversation, err := memory.Context(context.Background(), 42)
	if err != nil {
		t.Fatalf("Context() error = %v", err)
	}
	if got := conversation.ID; got == "stale-conversation" {
		t.Fatalf("Context().ID = %q, want new conversation id", got)
	}
}

func TestMemoryTruncatesAssistantMessageBeforeSaving(t *testing.T) {
	t.Parallel()

	store := newFakeStore()
	memory := New(store, Config{
		TTL:               2 * time.Hour,
		MaxItems:          8,
		AssistantMaxChars: 15,
		Now: func() time.Time {
			return time.Unix(1774920000, 0)
		},
	})

	ctx := context.Background()
	conversation, err := memory.Context(ctx, 42)
	if err != nil {
		t.Fatalf("Context() error = %v", err)
	}

	if err := memory.RememberTurn(ctx, 42, conversation.ID, "hello", "abcdefghijklmnopqrstuv", nil); err != nil {
		t.Fatalf("RememberTurn() error = %v", err)
	}

	payloads := store.lists[memory.messagesKey(conversation.ID)]
	if len(payloads) != 2 {
		t.Fatalf("len(payloads) = %d, want 2", len(payloads))
	}

	var assistant qa.ConversationMessage
	if err := json.Unmarshal([]byte(payloads[0]), &assistant); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if got, want := assistant.Role, qa.ConversationRoleAssistant; got != want {
		t.Fatalf("assistant.Role = %q, want %q", got, want)
	}
	if got, want := assistant.Text, "abcde\n...\nrstuv"; got != want {
		t.Fatalf("assistant.Text = %q, want %q", got, want)
	}

	var user qa.ConversationMessage
	if err := json.Unmarshal([]byte(payloads[1]), &user); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if got, want := user.Text, "hello"; got != want {
		t.Fatalf("user.Text = %q, want %q", got, want)
	}
}

func TestMemoryStoresAssistantAttachments(t *testing.T) {
	t.Parallel()

	store := newFakeStore()
	memory := New(store, Config{
		TTL:      2 * time.Hour,
		MaxItems: 8,
		Now: func() time.Time {
			return time.Unix(1774920000, 0)
		},
	})

	ctx := context.Background()
	conversation, err := memory.Context(ctx, 42)
	if err != nil {
		t.Fatalf("Context() error = %v", err)
	}

	err = memory.RememberTurn(
		ctx,
		42,
		conversation.ID,
		"send it again",
		"Here is the file.",
		[]qa.ConversationAttachment{
			{Name: "sample.pdf", Kind: qa.AttachmentDocument},
			{Name: "example.png", Kind: qa.AttachmentPhoto},
		},
	)
	if err != nil {
		t.Fatalf("RememberTurn() error = %v", err)
	}

	payloads := store.lists[memory.messagesKey(conversation.ID)]
	if len(payloads) != 2 {
		t.Fatalf("len(payloads) = %d, want 2", len(payloads))
	}

	var assistant qa.ConversationMessage
	if err := json.Unmarshal([]byte(payloads[0]), &assistant); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if len(assistant.Attachments) != 2 {
		t.Fatalf("len(assistant.Attachments) = %d, want 2", len(assistant.Attachments))
	}
	if got, want := assistant.Attachments[0], (qa.ConversationAttachment{Name: "sample.pdf", Kind: qa.AttachmentDocument}); got != want {
		t.Fatalf("assistant.Attachments[0] = %#v, want %#v", got, want)
	}
	if got, want := assistant.Attachments[1], (qa.ConversationAttachment{Name: "example.png", Kind: qa.AttachmentPhoto}); got != want {
		t.Fatalf("assistant.Attachments[1] = %#v, want %#v", got, want)
	}
}
