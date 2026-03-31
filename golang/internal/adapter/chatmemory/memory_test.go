package chatmemory

import (
	"context"
	"fmt"
	"testing"
	"time"
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

func (f *fakeStore) LRange(_ context.Context, key string, start, stop int64) ([]string, error) {
	values := f.lists[key]
	if len(values) == 0 || start >= int64(len(values)) {
		return nil, nil
	}
	if stop >= int64(len(values)) {
		stop = int64(len(values) - 1)
	}

	out := make([]string, 0, stop-start+1)
	for i := start; i <= stop; i++ {
		out = append(out, values[i])
	}
	return out, nil
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
	if len(conversation.Messages) != 0 {
		t.Fatalf("len(Context().Messages) = %d, want 0", len(conversation.Messages))
	}

	if err := memory.RememberTurn(ctx, 42, conversation.ID, "hello", "world"); err != nil {
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
	if len(reloaded.Messages) != 2 {
		t.Fatalf("len(Context().Messages) = %d, want 2", len(reloaded.Messages))
	}
	if got, want := reloaded.Messages[0].Role, "user"; got != want {
		t.Fatalf("messages[0].Role = %q, want %q", got, want)
	}
	if got, want := reloaded.Messages[0].Text, "hello"; got != want {
		t.Fatalf("messages[0].Text = %q, want %q", got, want)
	}
	if got, want := reloaded.Messages[1].Role, "assistant"; got != want {
		t.Fatalf("messages[1].Role = %q, want %q", got, want)
	}
	if got, want := reloaded.Messages[1].Text, "world"; got != want {
		t.Fatalf("messages[1].Text = %q, want %q", got, want)
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
		); err != nil {
			t.Fatalf("RememberTurn() turn %d error = %v", turn, err)
		}
	}

	reloaded, err := memory.Context(ctx, 42)
	if err != nil {
		t.Fatalf("Context() error = %v", err)
	}
	if len(reloaded.Messages) != 8 {
		t.Fatalf("len(Context().Messages) = %d, want 8", len(reloaded.Messages))
	}

	wantTexts := []string{
		"user-2",
		"assistant-2",
		"user-3",
		"assistant-3",
		"user-4",
		"assistant-4",
		"user-5",
		"assistant-5",
	}
	for i, want := range wantTexts {
		if got := reloaded.Messages[i].Text; got != want {
			t.Fatalf("messages[%d].Text = %q, want %q", i, got, want)
		}
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
