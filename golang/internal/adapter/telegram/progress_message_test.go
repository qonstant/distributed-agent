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
