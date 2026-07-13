package postgres

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	pq "github.com/lib/pq"

	"github.com/qonstant/distributed-agent/internal/domain/unisearch"
)

type UniversitySearchRepository struct {
	db *sql.DB
}

func NewUniversitySearchRepository(dbURL string) (*UniversitySearchRepository, error) {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("open postgres connection: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}

	return &UniversitySearchRepository{db: db}, nil
}

func (r *UniversitySearchRepository) Close() error {
	if r == nil || r.db == nil {
		return nil
	}
	return r.db.Close()
}

func (r *UniversitySearchRepository) SearchUniversities(
	ctx context.Context,
	query unisearch.Query,
) (unisearch.ResultSet, error) {
	if r == nil || r.db == nil {
		return unisearch.ResultSet{}, fmt.Errorf("postgres university search repository is not initialized")
	}

	limit := query.Limit
	if limit <= 0 {
		limit = 20
	}
	if limit > 50 {
		limit = 50
	}

	rankingYear, err := r.latestRankingYear(ctx)
	if err != nil {
		return unisearch.ResultSet{}, err
	}
	if rankingYear == 0 {
		return unisearch.ResultSet{RankingYear: 0, Items: []unisearch.University{}}, nil
	}

	searchText := strings.TrimSpace(query.Text)
	searchPattern := "%" + searchText + "%"
	subjectSlug := strings.TrimSpace(query.SubjectSlug)

	const universitiesQuery = `
		SELECT
			u.university_id,
			u.display_name,
			u.official_name,
			u.city,
			u.province,
			u.region,
			u.university_sector::text,
			u.official_website,
			u.qs_profile_url,
			g.qs_global_rank_display,
			g.qs_global_rank_numeric,
			g.qs_global_italy_rank,
			g.qs_global_score
		FROM universities u
		LEFT JOIN university_global_rankings g
			ON g.university_id = u.university_id
			AND g.ranking_year = $1
		WHERE
			(
				$2 = ''
				OR lower(u.display_name) = lower($2)
				OR lower(COALESCE(u.official_name, '')) = lower($2)
				OR u.display_name ILIKE $3
				OR COALESCE(u.official_name, '') ILIKE $3
				OR COALESCE(u.city, '') ILIKE $3
				OR COALESCE(u.region, '') ILIKE $3
				OR EXISTS (
					SELECT 1
					FROM university_subject_rankings usr
					JOIN subjects s ON s.subject_slug = usr.subject_slug
					WHERE usr.university_id = u.university_id
					  AND usr.ranking_year = $1
					  AND s.subject_name ILIKE $3
				)
			)
			AND (
				$4 = ''
				OR EXISTS (
					SELECT 1
					FROM university_subject_rankings usr
					WHERE usr.university_id = u.university_id
					  AND usr.ranking_year = $1
					  AND usr.subject_slug = $4
				)
			)
		ORDER BY
			CASE
				WHEN $2 = '' THEN 0
				WHEN lower(u.display_name) = lower($2) THEN 100
				WHEN lower(COALESCE(u.official_name, '')) = lower($2) THEN 95
				WHEN u.display_name ILIKE $3 THEN 80
				WHEN COALESCE(u.official_name, '') ILIKE $3 THEN 75
				WHEN EXISTS (
					SELECT 1
					FROM university_subject_rankings usr
					JOIN subjects s ON s.subject_slug = usr.subject_slug
					WHERE usr.university_id = u.university_id
					  AND usr.ranking_year = $1
					  AND s.subject_name ILIKE $3
				) THEN 60
				WHEN COALESCE(u.city, '') ILIKE $3 THEN 50
				WHEN COALESCE(u.region, '') ILIKE $3 THEN 40
				ELSE 10
			END DESC,
			CASE WHEN g.qs_global_italy_rank IS NULL THEN 1 ELSE 0 END,
			g.qs_global_italy_rank NULLS LAST,
			u.display_name ASC
		LIMIT $5
	`

	rows, err := r.db.QueryContext(
		ctx,
		universitiesQuery,
		rankingYear,
		searchText,
		searchPattern,
		subjectSlug,
		limit,
	)
	if err != nil {
		return unisearch.ResultSet{}, fmt.Errorf("search universities: %w", err)
	}
	defer rows.Close()

	items := make([]unisearch.University, 0, limit)
	universityIDs := make([]string, 0, limit)
	indexByID := make(map[string]int, limit)

	for rows.Next() {
		var (
			item              unisearch.University
			officialName      sql.NullString
			city              sql.NullString
			province          sql.NullString
			region            sql.NullString
			sector            sql.NullString
			officialWebsite   sql.NullString
			qsProfileURL      sql.NullString
			globalRankDisplay sql.NullString
			globalRankNumeric sql.NullInt64
			globalItalyRank   sql.NullInt64
			globalScore       sql.NullFloat64
		)

		if err := rows.Scan(
			&item.UniversityID,
			&item.DisplayName,
			&officialName,
			&city,
			&province,
			&region,
			&sector,
			&officialWebsite,
			&qsProfileURL,
			&globalRankDisplay,
			&globalRankNumeric,
			&globalItalyRank,
			&globalScore,
		); err != nil {
			return unisearch.ResultSet{}, fmt.Errorf("scan university search row: %w", err)
		}

		item.OfficialName = nullString(officialName)
		item.City = nullString(city)
		item.Province = nullString(province)
		item.Region = nullString(region)
		item.UniversitySector = nullString(sector)
		item.OfficialWebsite = nullString(officialWebsite)
		item.QSProfileURL = nullString(qsProfileURL)

		if globalRankDisplay.Valid || globalRankNumeric.Valid || globalItalyRank.Valid || globalScore.Valid {
			item.QSGlobalRank = &unisearch.GlobalRanking{
				RankDisplay: nullString(globalRankDisplay),
				RankNumeric: nullInt(globalRankNumeric),
				ItalyRank:   nullInt(globalItalyRank),
				Score:       nullFloat(globalScore),
			}
		}

		indexByID[item.UniversityID] = len(items)
		universityIDs = append(universityIDs, item.UniversityID)
		items = append(items, item)
	}

	if err := rows.Err(); err != nil {
		return unisearch.ResultSet{}, fmt.Errorf("iterate university search rows: %w", err)
	}

	if len(items) == 0 {
		return unisearch.ResultSet{RankingYear: rankingYear, Items: items}, nil
	}

	const subjectsQuery = `
		SELECT
			usr.university_id,
			usr.subject_slug,
			s.subject_name,
			usr.qs_subject_rank_display,
			usr.qs_subject_rank_numeric,
			usr.qs_subject_italy_rank,
			usr.qs_subject_score
		FROM university_subject_rankings usr
		JOIN subjects s ON s.subject_slug = usr.subject_slug
		WHERE usr.ranking_year = $1
		  AND usr.university_id = ANY($2)
		  AND ($3 = '' OR usr.subject_slug = $3)
		ORDER BY
			usr.university_id,
			CASE WHEN $3 <> '' AND usr.subject_slug = $3 THEN 0 ELSE 1 END,
			usr.qs_subject_italy_rank NULLS LAST,
			s.subject_name ASC
	`

	subjectRows, err := r.db.QueryContext(
		ctx,
		subjectsQuery,
		rankingYear,
		pq.Array(universityIDs),
		subjectSlug,
	)
	if err != nil {
		return unisearch.ResultSet{}, fmt.Errorf("query matching subjects: %w", err)
	}
	defer subjectRows.Close()

	subjectCountByUniversity := make(map[string]int, len(universityIDs))
	for subjectRows.Next() {
		var (
			universityID string
			subject      unisearch.SubjectRanking
			rankDisplay  sql.NullString
			rankNumeric  sql.NullInt64
			italyRank    sql.NullInt64
			score        sql.NullFloat64
		)

		if err := subjectRows.Scan(
			&universityID,
			&subject.SubjectSlug,
			&subject.SubjectName,
			&rankDisplay,
			&rankNumeric,
			&italyRank,
			&score,
		); err != nil {
			return unisearch.ResultSet{}, fmt.Errorf("scan subject search row: %w", err)
		}

		idx, ok := indexByID[universityID]
		if !ok {
			continue
		}
		if subjectSlug == "" && subjectCountByUniversity[universityID] >= 3 {
			continue
		}

		subject.RankDisplay = nullString(rankDisplay)
		subject.RankNumeric = nullInt(rankNumeric)
		subject.ItalyRank = nullInt(italyRank)
		subject.Score = nullFloat(score)

		items[idx].MatchingSubjects = append(items[idx].MatchingSubjects, subject)
		subjectCountByUniversity[universityID]++
	}

	if err := subjectRows.Err(); err != nil {
		return unisearch.ResultSet{}, fmt.Errorf("iterate subject search rows: %w", err)
	}

	return unisearch.ResultSet{
		RankingYear: rankingYear,
		Items:       items,
	}, nil
}

func (r *UniversitySearchRepository) latestRankingYear(ctx context.Context) (int, error) {
	const query = `
		SELECT COALESCE(
			(SELECT MAX(ranking_year) FROM university_global_rankings),
			(SELECT MAX(ranking_year) FROM university_subject_rankings),
			0
		)
	`

	var rankingYear int
	if err := r.db.QueryRowContext(ctx, query).Scan(&rankingYear); err != nil {
		return 0, fmt.Errorf("query latest ranking year: %w", err)
	}
	return rankingYear, nil
}

func nullString(value sql.NullString) string {
	if !value.Valid {
		return ""
	}
	return value.String
}

func nullInt(value sql.NullInt64) *int {
	if !value.Valid {
		return nil
	}
	out := int(value.Int64)
	return &out
}

func nullFloat(value sql.NullFloat64) *float64 {
	if !value.Valid {
		return nil
	}
	out := value.Float64
	return &out
}
