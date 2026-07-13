package unisearchhttp

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/qonstant/distributed-agent/internal/domain/unisearch"
)

type Searcher interface {
	SearchUniversities(ctx context.Context, query unisearch.Query) (unisearch.ResultSet, error)
}

type Server struct {
	searcher Searcher
}

func NewServer(searcher Searcher) *Server {
	return &Server{searcher: searcher}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/health", s.handleHealth)
	mux.HandleFunc("/api/universities/search", s.handleUniversitySearch)
	return requestLogger(mux)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleUniversitySearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method_not_allowed"})
		return
	}

	if s.searcher == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "search_not_configured"})
		return
	}

	limit := 20
	if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed <= 0 {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_limit"})
			return
		}
		if parsed > 50 {
			parsed = 50
		}
		limit = parsed
	}

	query := unisearch.Query{
		Text:        strings.TrimSpace(r.URL.Query().Get("q")),
		SubjectSlug: strings.TrimSpace(r.URL.Query().Get("subject_slug")),
		Limit:       limit,
	}

	result, err := s.searcher.SearchUniversities(r.Context(), query)
	if err != nil {
		log.Printf("uni search failed: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "search_failed"})
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"ranking_year": result.RankingYear,
		"count":        len(result.Items),
		"items":        result.Items,
		"query": map[string]any{
			"q":            query.Text,
			"subject_slug": query.SubjectSlug,
			"limit":        query.Limit,
		},
	})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func requestLogger(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		startedAt := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("[uni-search-api] %s %s (%s)", r.Method, r.URL.Path, time.Since(startedAt))
	})
}
