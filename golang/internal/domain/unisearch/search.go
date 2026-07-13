package unisearch

type Query struct {
	Text        string
	SubjectSlug string
	Limit       int
}

type ResultSet struct {
	RankingYear int          `json:"ranking_year"`
	Items       []University `json:"items"`
}

type University struct {
	UniversityID     string           `json:"university_id"`
	DisplayName      string           `json:"display_name"`
	OfficialName     string           `json:"official_name,omitempty"`
	City             string           `json:"city,omitempty"`
	Province         string           `json:"province,omitempty"`
	Region           string           `json:"region,omitempty"`
	UniversitySector string           `json:"university_sector,omitempty"`
	OfficialWebsite  string           `json:"official_website,omitempty"`
	QSProfileURL     string           `json:"qs_profile_url,omitempty"`
	QSGlobalRank     *GlobalRanking   `json:"qs_global_rank,omitempty"`
	MatchingSubjects []SubjectRanking `json:"matching_subjects,omitempty"`
}

type GlobalRanking struct {
	RankDisplay string   `json:"rank_display,omitempty"`
	RankNumeric *int     `json:"rank_numeric,omitempty"`
	ItalyRank   *int     `json:"italy_rank,omitempty"`
	Score       *float64 `json:"score,omitempty"`
}

type SubjectRanking struct {
	SubjectSlug string   `json:"subject_slug"`
	SubjectName string   `json:"subject_name"`
	RankDisplay string   `json:"rank_display,omitempty"`
	RankNumeric *int     `json:"rank_numeric,omitempty"`
	ItalyRank   *int     `json:"italy_rank,omitempty"`
	Score       *float64 `json:"score,omitempty"`
}
