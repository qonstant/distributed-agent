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

	english := renderProgressText(thinkingFramesLatin, europeFactsLatin[1], 1)
	if english != "Thinking..\n\n<b>Interesting fact:</b>\n<blockquote>There is a Victor Hugo Street in every town in France.</blockquote>\n<i>Source: France</i>" {
		t.Fatalf("english progress text = %q", english)
	}

	russian := renderProgressText(thinkingFramesCyrillic, europeFactsCyrillic[2], 2)
	if russian != "Думаю...\n\n<b>Интересный факт:</b>\n<blockquote>Риму, который называют Вечным городом, почти 3 000 лет, а столицей Италии он стал в 1871 году.</blockquote>\n<i>Источник: Италия</i>" {
		t.Fatalf("russian progress text = %q", russian)
	}
}

func TestPickProgressFact(t *testing.T) {
	t.Parallel()

	got := pickProgressFact(europeFactsLatin, 4)
	if got != europeFactsLatin[4] {
		t.Fatalf("pickProgressFact(...) = %#v, want %#v", got, europeFactsLatin[4])
	}
}

func TestRenderProgressTextKeepsSameFactAcrossSteps(t *testing.T) {
	t.Parallel()

	fact := europeFactsLatin[0]
	first := renderProgressText(thinkingFramesLatin, fact, 0)
	second := renderProgressText(thinkingFramesLatin, fact, 2)

	if first != "Thinking.\n\n<b>Interesting fact:</b>\n<blockquote>The national anthem of Spain has no words.</blockquote>\n<i>Source: Spain</i>" {
		t.Fatalf("first progress text = %q", first)
	}
	if second != "Thinking...\n\n<b>Interesting fact:</b>\n<blockquote>The national anthem of Spain has no words.</blockquote>\n<i>Source: Spain</i>" {
		t.Fatalf("second progress text = %q", second)
	}
}
