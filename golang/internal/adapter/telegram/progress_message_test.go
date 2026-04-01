package telegram

import "testing"

func TestThinkingFramesForText(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		text string
		want []string
	}{
		{
			name: "latin text uses english thinking",
			text: "hello there",
			want: thinkingFramesLatin,
		},
		{
			name: "cyrillic text uses russian thinking",
			text: "привет как дела",
			want: thinkingFramesCyrillic,
		},
		{
			name: "mixed text prefers cyrillic when present",
			text: "Call me Роман",
			want: thinkingFramesCyrillic,
		},
		{
			name: "no letters defaults to english thinking",
			text: "123 ?!",
			want: thinkingFramesLatin,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := thinkingFramesForText(tt.text)
			if len(got) != len(tt.want) {
				t.Fatalf("len(got) = %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("got[%d] = %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestProgressFactsForText(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		text string
		want []progressFact
	}{
		{
			name: "latin text uses english facts",
			text: "hello there",
			want: europeFactsLatin,
		},
		{
			name: "cyrillic text uses russian facts",
			text: "привет как дела",
			want: europeFactsCyrillic,
		},
		{
			name: "mixed text prefers russian facts when cyrillic is present",
			text: "Study в Европе",
			want: europeFactsCyrillic,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := progressFactsForText(tt.text)
			if len(got) != len(tt.want) {
				t.Fatalf("len(got) = %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("got[%d] = %#v, want %#v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestRenderProgressText(t *testing.T) {
	t.Parallel()

	english := renderProgressText(thinkingFramesLatin, europeFactsLatin, 1)
	if english != "Thinking..\n\nThe Bologna Process helped align degree structures across much of Europe into bachelor-master-doctorate cycles.\nSource: Bologna Process" {
		t.Fatalf("english progress text = %q", english)
	}

	russian := renderProgressText(thinkingFramesCyrillic, europeFactsCyrillic, 2)
	if russian != "Думаю...\n\nПрограмма Erasmus+ поддерживает учебную мобильность и обмены между многими европейскими странами.\nИсточник: Erasmus+" {
		t.Fatalf("russian progress text = %q", russian)
	}
}
