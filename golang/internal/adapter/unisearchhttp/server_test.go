package unisearchhttp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/qonstant/distributed-agent/internal/domain/unisearch"
)

type fakeSearcher struct {
	lastQuery unisearch.Query
	result    unisearch.ResultSet
	err       error
}

func (f *fakeSearcher) SearchUniversities(_ context.Context, query unisearch.Query) (unisearch.ResultSet, error) {
	f.lastQuery = query
	return f.result, f.err
}

func TestUniversitySearchHandlerReturnsJSON(t *testing.T) {
	t.Parallel()

	searcher := &fakeSearcher{
		result: unisearch.ResultSet{
			RankingYear: 2027,
			Items: []unisearch.University{
				{UniversityID: "bologna", DisplayName: "Universita di Bologna"},
			},
		},
	}

	req := httptest.NewRequest(http.MethodGet, "/api/universities/search?q=bologna&subject_slug=law&limit=9", nil)
	rec := httptest.NewRecorder()

	NewServer(searcher).Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	if searcher.lastQuery.Text != "bologna" {
		t.Fatalf("expected query text to be propagated, got %q", searcher.lastQuery.Text)
	}
	if searcher.lastQuery.SubjectSlug != "law" {
		t.Fatalf("expected subject slug to be propagated, got %q", searcher.lastQuery.SubjectSlug)
	}
	if searcher.lastQuery.Limit != 9 {
		t.Fatalf("expected limit 9, got %d", searcher.lastQuery.Limit)
	}

	var payload struct {
		RankingYear int                    `json:"ranking_year"`
		Count       int                    `json:"count"`
		Items       []unisearch.University `json:"items"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if payload.RankingYear != 2027 {
		t.Fatalf("expected ranking year 2027, got %d", payload.RankingYear)
	}
	if payload.Count != 1 {
		t.Fatalf("expected count 1, got %d", payload.Count)
	}
}

func TestUniversitySearchHandlerRejectsBadLimit(t *testing.T) {
	t.Parallel()

	req := httptest.NewRequest(http.MethodGet, "/api/universities/search?limit=nope", nil)
	rec := httptest.NewRecorder()

	NewServer(&fakeSearcher{}).Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}
