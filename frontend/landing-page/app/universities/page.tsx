"use client"

import { FormEvent, useEffect, useState } from "react"
import Link from "next/link"
import { ArrowRight, ExternalLink, GraduationCap, Loader2, MapPin, Search, Sparkles } from "lucide-react"
import { Footer } from "@/components/footer"
import { useLanguage } from "@/components/language-provider"
import { Button } from "@/components/ui/button"

type SearchSubject = {
  subject_slug: string
  subject_name: string
  italy_rank?: number | null
  rank_display?: string
}

type SearchUniversity = {
  university_id: string
  display_name: string
  city?: string
  region?: string
  official_website?: string
  qs_profile_url?: string
  qs_global_rank?: {
    italy_rank?: number | null
    rank_display?: string
  } | null
  matching_subjects?: SearchSubject[]
}

const pageCopy = {
  en: {
    pill: "Public University Explorer",
    title: "Explore universities before you ever pay.",
    intro:
      "Browse strong-fit options, compare cities and study styles, and get a clearer sense of where your profile belongs before moving to a paid plan.",
    searchPlaceholder: "Search universities, cities, or programs...",
    filtersNote: "MVP search is live here first. Next filters can be subject, city, region, and ranking range.",
    searchButton: "Search",
    searchHint: "Try Bologna, Milan, law, medicine, or computer science.",
    searchResultsTitle: "Search results",
    searchResultsSubtitle: "Internal API result preview for your uni-search MVP.",
    topResultsTitle: "Top Italian universities right now",
    topResultsSubtitle: "Default view when no query is entered.",
    loading: "Searching universities...",
    empty: "No universities matched this search yet.",
    searchError: "Search is temporarily unavailable.",
    officialWebsite: "Official website",
    qsProfile: "QS profile",
    italyRank: "Italy rank",
    globalRank: "Global rank",
    matchingSubjects: "Matching subjects",
    rankingYearLabel: "QS ranking year",
    fitTitle: "Match by academic fit",
    fitText: "Show universities based on degree level, field, GPA strength, and language of study.",
    cityTitle: "Compare cities",
    cityText: "Let users narrow down Milan, Turin, Bologna, Padua, and other destinations by lifestyle and budget.",
    paidTitle: "Lead into paid guidance",
    paidText: "After users shortlist options, we can offer saved results, a custom roadmap, and full admissions help.",
    cardsTitle: "Starter university cards",
    cardsText: "A public directory can start simple, then evolve into real filtering.",
    unlock: "Unlock full shortlist",
    preview: "Public preview",
    saveDirection: "Save this direction",
    italy: "Italy",
    cards: [
      {
        name: "Politecnico di Milano",
        city: "Milan",
        focus: "Engineering, design, architecture",
        badge: "Scholarship-friendly",
        logo: "/logos/polimi.jpg",
      },
      {
        name: "Universita di Bologna",
        city: "Bologna",
        focus: "Research, humanities, international programs",
        badge: "Wide program mix",
        logo: "/logos/unibo.jpg",
      },
      {
        name: "Politecnico di Torino",
        city: "Turin",
        focus: "STEM, innovation, industry links",
        badge: "Strong value",
        logo: "/logos/polito.jpg",
      },
    ],
  },
  ru: {
    pill: "Открытый каталог университетов",
    title: "Изучайте университеты еще до оплаты.",
    intro:
      "Смотрите подходящие варианты, сравнивайте города и стили обучения, чтобы лучше понять, куда подходит ваш профиль, прежде чем переходить к платному сопровождению.",
    searchPlaceholder: "Поиск по университетам, городам или программам...",
    filtersNote: "Сначала здесь работает MVP-поиск. Следующими можно добавить фильтры по направлению, городу, региону и диапазону рейтинга.",
    searchButton: "Искать",
    searchHint: "Попробуйте Bologna, Milan, law, medicine или computer science.",
    searchResultsTitle: "Результаты поиска",
    searchResultsSubtitle: "Черновой просмотр внутреннего API для MVP поиска университетов.",
    topResultsTitle: "Топ университетов Италии сейчас",
    topResultsSubtitle: "Базовый список, когда запрос не введен.",
    loading: "Ищем университеты...",
    empty: "По этому запросу пока ничего не найдено.",
    searchError: "Поиск временно недоступен.",
    officialWebsite: "Официальный сайт",
    qsProfile: "Профиль QS",
    italyRank: "Ранг по Италии",
    globalRank: "Глобальный ранг",
    matchingSubjects: "Подходящие предметы",
    rankingYearLabel: "Год рейтинга QS",
    fitTitle: "Подбор по академическому профилю",
    fitText: "Показывайте университеты по уровню обучения, направлению, силе GPA и языку программы.",
    cityTitle: "Сравнение городов",
    cityText: "Пользователь сможет сузить выбор между Миланом, Турином, Болоньей, Падуей и другими городами по бюджету и образу жизни.",
    paidTitle: "Переход к платному сопровождению",
    paidText: "После шортлиста можно предложить сохранение результатов, персональную дорожную карту и полную помощь с поступлением.",
    cardsTitle: "Стартовые карточки университетов",
    cardsText: "Публичный каталог может начаться просто, а позже вырасти в полноценный фильтр и матчинг.",
    unlock: "Открыть полный шортлист",
    preview: "Открытый просмотр",
    saveDirection: "Сохранить это направление",
    italy: "Италия",
    cards: [
      {
        name: "Politecnico di Milano",
        city: "Милан",
        focus: "Инженерия, дизайн, архитектура",
        badge: "Подходит для стипендий",
        logo: "/logos/polimi.jpg",
      },
      {
        name: "Universita di Bologna",
        city: "Болонья",
        focus: "Исследования, гуманитарные науки, международные программы",
        badge: "Широкий выбор программ",
        logo: "/logos/unibo.jpg",
      },
      {
        name: "Politecnico di Torino",
        city: "Турин",
        focus: "STEM, инновации, связи с индустрией",
        badge: "Сильная ценность",
        logo: "/logos/polito.jpg",
      },
    ],
  },
  kk: {
    pill: "Ашық университет каталогы",
    title: "Төлем жасамай тұрып университеттерді зерттеңіз.",
    intro:
      "Профиліңізге сай келетін бағыттарды көріп, қалалар мен оқу стильдерін салыстырыңыз, содан кейін ғана ақылы қызметке өтуді шешіңіз.",
    searchPlaceholder: "Университет, қала немесе бағдарлама бойынша іздеу...",
    filtersNote: "Алдымен осы жерде MVP іздеу жұмыс істейді. Келесі қадамда мамандық, қала, өңір және рейтинг ауқымы бойынша сүзгілер қосуға болады.",
    searchButton: "Іздеу",
    searchHint: "Bologna, Milan, law, medicine немесе computer science деп көріңіз.",
    searchResultsTitle: "Іздеу нәтижелері",
    searchResultsSubtitle: "Университет іздеу MVP-іне арналған ішкі API нәтижелерінің алдын ала көрінісі.",
    topResultsTitle: "Қазір Италиядағы үздік университеттер",
    topResultsSubtitle: "Сұрау бос болғандағы әдепкі тізім.",
    loading: "Университеттер ізделіп жатыр...",
    empty: "Бұл сұрауға сәйкес университет табылмады.",
    searchError: "Іздеу уақытша қолжетімсіз.",
    officialWebsite: "Ресми сайт",
    qsProfile: "QS профилі",
    italyRank: "Италиядағы орны",
    globalRank: "Жаһандық орын",
    matchingSubjects: "Сәйкес пәндер",
    rankingYearLabel: "QS рейтинг жылы",
    fitTitle: "Академиялық сәйкестік бойынша іріктеу",
    fitText: "Университеттерді оқу деңгейі, мамандық, GPA күші және оқу тілі бойынша көрсетуге болады.",
    cityTitle: "Қалаларды салыстыру",
    cityText: "Пайдаланушылар Милан, Турин, Болонья, Падуя және басқа бағыттарды өмір салты мен бюджет бойынша қысқарта алады.",
    paidTitle: "Ақылы сүйемелдеуге көшу",
    paidText: "Қысқа тізім жасалғаннан кейін нәтижені сақтау, жеке жол картасы және толық түсу көмегін ұсынуға болады.",
    cardsTitle: "Бастапқы университет карталары",
    cardsText: "Ашық каталогты қарапайым бастап, кейін толық сүзгілеу мен матчингке айналдыруға болады.",
    unlock: "Толық шортлистті ашу",
    preview: "Ашық алдын ала қарау",
    saveDirection: "Осы бағытты сақтау",
    italy: "Италия",
    cards: [
      {
        name: "Politecnico di Milano",
        city: "Милан",
        focus: "Инженерия, дизайн, архитектура",
        badge: "Шәкіртақыға ыңғайлы",
        logo: "/logos/polimi.jpg",
      },
      {
        name: "Universita di Bologna",
        city: "Болонья",
        focus: "Зерттеу, гуманитарлық бағыттар, халықаралық бағдарламалар",
        badge: "Бағдарлама таңдауы кең",
        logo: "/logos/unibo.jpg",
      },
      {
        name: "Politecnico di Torino",
        city: "Турин",
        focus: "STEM, инновация, индустриямен байланыс",
        badge: "Құнына сай мықты",
        logo: "/logos/polito.jpg",
      },
    ],
  },
} as const

export default function UniversitiesPage() {
  const { language } = useLanguage()
  const copy = pageCopy[language]
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<SearchUniversity[]>([])
  const [rankingYear, setRankingYear] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void runSearch("")
  }, [])

  async function runSearch(nextQuery: string) {
    setIsLoading(true)
    setError(null)

    try {
      const params = new URLSearchParams({ limit: "12" })
      if (nextQuery.trim()) {
        params.set("q", nextQuery.trim())
      }

      const response = await fetch(`/api/universities/search?${params.toString()}`, {
        method: "GET",
        cache: "no-store",
      })

      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload?.error || "search_failed")
      }

      setResults(Array.isArray(payload.items) ? payload.items : [])
      setRankingYear(typeof payload.ranking_year === "number" ? payload.ranking_year : null)
    } catch {
      setResults([])
      setRankingYear(null)
      setError(copy.searchError)
    } finally {
      setIsLoading(false)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void runSearch(query)
  }

  return (
    <main className="min-h-screen pt-32">
      <section className="px-4 pb-14 pt-6">
        <div className="mx-auto max-w-6xl rounded-[2rem] border border-white/10 bg-white/[0.04] px-6 py-10 shadow-[0_30px_90px_-40px_rgba(0,0,0,0.95)] md:px-10">
          <div className="max-w-3xl space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/25 bg-cyan-400/10 px-4 py-2 text-sm font-medium text-cyan-200">
              <Sparkles className="h-4 w-4" />
              {copy.pill}
            </div>
            <div className="space-y-4">
              <h1 className="text-4xl font-bold tracking-tight text-white md:text-6xl">
                {copy.title}
              </h1>
              <p className="max-w-2xl text-lg leading-relaxed text-white/70">
                {copy.intro}
              </p>
            </div>
          </div>

          <div className="mt-8 grid gap-4 md:grid-cols-[1.6fr_0.8fr]">
            <form onSubmit={handleSubmit} className="flex items-center gap-3 rounded-[1.5rem] border border-white/10 bg-black/30 px-5 py-3">
              <Search className="h-5 w-5 text-cyan-300" />
              <input
                type="text"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={copy.searchPlaceholder}
                className="w-full bg-transparent text-base text-white outline-none placeholder:text-white/35"
              />
              <Button
                type="submit"
                className="rounded-full bg-cyan-300 px-5 text-black hover:bg-cyan-200"
                disabled={isLoading}
              >
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : copy.searchButton}
              </Button>
            </form>
            <div className="rounded-[1.5rem] border border-white/10 bg-black/30 px-5 py-4 text-sm text-white/70">
              <p>{copy.filtersNote}</p>
              <p className="mt-2 text-white/45">{copy.searchHint}</p>
            </div>
          </div>
        </div>
      </section>

      <section className="px-4 pb-8">
        <div className="mx-auto max-w-6xl rounded-[1.75rem] border border-white/10 bg-white/[0.03] p-6 shadow-[0_22px_70px_-38px_rgba(0,0,0,0.95)]">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div>
              <h2 className="text-2xl font-semibold text-white md:text-3xl">
                {query.trim() ? copy.searchResultsTitle : copy.topResultsTitle}
              </h2>
              <p className="mt-2 text-sm text-white/60">
                {query.trim() ? copy.searchResultsSubtitle : copy.topResultsSubtitle}
              </p>
            </div>
            {rankingYear ? (
              <div className="rounded-full border border-white/10 bg-black/30 px-4 py-2 text-xs uppercase tracking-[0.2em] text-cyan-200">
                {copy.rankingYearLabel}: {rankingYear}
              </div>
            ) : null}
          </div>

          {error ? (
            <div className="mt-6 rounded-2xl border border-red-400/25 bg-red-500/10 px-4 py-3 text-sm text-red-100">
              {error}
            </div>
          ) : null}

          {isLoading ? (
            <div className="mt-6 flex items-center gap-3 rounded-2xl border border-white/10 bg-black/20 px-4 py-4 text-white/70">
              <Loader2 className="h-4 w-4 animate-spin text-cyan-300" />
              {copy.loading}
            </div>
          ) : null}

          {!isLoading && !error && results.length === 0 ? (
            <div className="mt-6 rounded-2xl border border-white/10 bg-black/20 px-4 py-4 text-sm text-white/65">
              {copy.empty}
            </div>
          ) : null}

          {!isLoading && results.length > 0 ? (
            <div className="mt-6 grid gap-5 md:grid-cols-2 xl:grid-cols-3">
              {results.map((university) => (
                <article
                  key={university.university_id}
                  className="rounded-[1.5rem] border border-white/10 bg-black/25 p-5 shadow-[0_22px_70px_-38px_rgba(0,0,0,0.95)]"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{university.display_name}</h3>
                      <p className="mt-2 text-sm text-white/55">
                        {[university.city, university.region].filter(Boolean).join(", ") || copy.italy}
                      </p>
                    </div>
                    {university.qs_global_rank?.italy_rank ? (
                      <div className="rounded-2xl border border-cyan-300/25 bg-cyan-300/10 px-3 py-2 text-right text-xs text-cyan-100">
                        <div className="uppercase tracking-[0.16em] text-cyan-200/80">{copy.italyRank}</div>
                        <div className="mt-1 text-lg font-semibold">#{university.qs_global_rank.italy_rank}</div>
                      </div>
                    ) : null}
                  </div>

                  {university.qs_global_rank?.rank_display ? (
                    <p className="mt-4 text-sm text-white/65">
                      {copy.globalRank}: <span className="text-white">{university.qs_global_rank.rank_display}</span>
                    </p>
                  ) : null}

                  {university.matching_subjects?.length ? (
                    <div className="mt-4">
                      <p className="text-xs uppercase tracking-[0.18em] text-white/45">{copy.matchingSubjects}</p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {university.matching_subjects.map((subject) => (
                          <span
                            key={`${university.university_id}-${subject.subject_slug}`}
                            className="rounded-full border border-white/10 bg-white/[0.06] px-3 py-1 text-xs text-white/75"
                          >
                            {subject.subject_name}
                            {subject.italy_rank ? ` · #${subject.italy_rank}` : ""}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  <div className="mt-5 flex flex-wrap gap-3">
                    {university.official_website ? (
                      <a
                        href={university.official_website}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.06] px-4 py-2 text-sm text-white transition hover:bg-white/[0.1]"
                      >
                        {copy.officialWebsite}
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    ) : null}
                    {university.qs_profile_url ? (
                      <a
                        href={university.qs_profile_url}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-transparent px-4 py-2 text-sm text-white/75 transition hover:bg-white/[0.06] hover:text-white"
                      >
                        {copy.qsProfile}
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    ) : null}
                  </div>
                </article>
              ))}
            </div>
          ) : null}
        </div>
      </section>

      <section className="px-4 pb-8">
        <div className="mx-auto grid max-w-6xl gap-4 md:grid-cols-3">
          <div className="rounded-[1.5rem] border border-white/10 bg-card/60 p-6">
            <GraduationCap className="mb-4 h-8 w-8 text-cyan-300" />
            <h2 className="text-xl font-semibold text-white">{copy.fitTitle}</h2>
            <p className="mt-3 text-sm leading-6 text-white/65">
              {copy.fitText}
            </p>
          </div>
          <div className="rounded-[1.5rem] border border-white/10 bg-card/60 p-6">
            <MapPin className="mb-4 h-8 w-8 text-cyan-300" />
            <h2 className="text-xl font-semibold text-white">{copy.cityTitle}</h2>
            <p className="mt-3 text-sm leading-6 text-white/65">
              {copy.cityText}
            </p>
          </div>
          <div className="rounded-[1.5rem] border border-white/10 bg-card/60 p-6">
            <Sparkles className="mb-4 h-8 w-8 text-cyan-300" />
            <h2 className="text-xl font-semibold text-white">{copy.paidTitle}</h2>
            <p className="mt-3 text-sm leading-6 text-white/65">
              {copy.paidText}
            </p>
          </div>
        </div>
      </section>

      <section className="px-4 pb-20">
        <div className="mx-auto max-w-6xl">
          <div className="mb-6 flex items-end justify-between gap-4">
            <div>
              <h2 className="text-2xl font-semibold text-white md:text-3xl">{copy.cardsTitle}</h2>
              <p className="mt-2 text-white/60">{copy.cardsText}</p>
            </div>
            <Button asChild variant="outline" className="rounded-full border-white/15 bg-transparent text-white hover:bg-white/8">
              <Link href="/#signup">
                {copy.unlock} <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
          </div>

          <div className="grid gap-5 md:grid-cols-3">
            {copy.cards.map((university) => (
              <article
                key={university.name}
                className="overflow-hidden rounded-[1.75rem] border border-white/10 bg-white/[0.035] shadow-[0_22px_70px_-38px_rgba(0,0,0,0.95)]"
              >
                <div className="flex items-center justify-between border-b border-white/8 px-5 py-4">
                  <span className="rounded-full border border-cyan-300/25 bg-cyan-300/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-cyan-200">
                    {university.badge}
                  </span>
                  <span className="text-xs text-white/45">{copy.preview}</span>
                </div>
                <div className="space-y-5 p-5">
                  <div className="flex items-center gap-4">
                    <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-white p-2">
                      <img src={university.logo} alt={university.name} className="h-12 w-12 object-contain" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">{university.name}</h3>
                      <p className="text-sm text-white/55">{university.city}, {copy.italy}</p>
                    </div>
                  </div>
                  <p className="text-sm leading-6 text-white/70">{university.focus}</p>
                  <Button asChild className="w-full rounded-full bg-white text-black hover:bg-white/90">
                    <Link href="/#signup">{copy.saveDirection}</Link>
                  </Button>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </main>
  )
}
