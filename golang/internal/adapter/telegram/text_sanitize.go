package telegram

import (
	"path/filepath"
	"strings"
	"unicode/utf8"
)

func sanitizeUTF8(value string) string {
	if utf8.ValidString(value) {
		return value
	}
	return string([]rune(value))
}

func trimRunes(value string, max int) string {
	runes := []rune(value)
	if len(runes) <= max {
		return value
	}
	return string(runes[:max])
}

func safeFilename(name string) string {
	name = sanitizeUTF8(name)
	name = filepath.Base(name)
	if name == "" {
		return "file"
	}

	var out []rune
	for _, r := range name {
		switch {
		case r == '\t' || r == '\n' || r == '\r':
			out = append(out, '_')
		case r < 32 || r == 0x7f:
			out = append(out, '_')
		default:
			out = append(out, r)
		}
	}

	name = strings.TrimSpace(string(out))
	if name == "" {
		return "file"
	}
	return name
}
