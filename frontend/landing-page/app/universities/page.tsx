"use client"

import Link from "next/link"
import { ArrowRight, GraduationCap, MapPin, Search, Sparkles } from "lucide-react"
import { Footer } from "@/components/footer"
import { useLanguage } from "@/components/language-provider"
import { Button } from "@/components/ui/button"

const pageCopy = {
  en: {
    pill: "Public University Explorer",
    title: "Explore universities before you ever pay.",
    intro:
      "Browse strong-fit options, compare cities and study styles, and get a clearer sense of where your profile belongs before moving to a paid plan.",
    searchPlaceholder: "Search universities, cities, or programs...",
    filtersNote: "Filters and matching logic can go here next: budget, language, city, scholarship need.",
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
    filtersNote: "Дальше сюда можно добавить фильтры и подбор: бюджет, язык, город, потребность в стипендии.",
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
    filtersNote: "Келесі қадамда мұнда сүзгілер мен сәйкестендіру логикасын қосуға болады: бюджет, тіл, қала, шәкіртақы қажеттілігі.",
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
            <div className="flex items-center gap-3 rounded-[1.5rem] border border-white/10 bg-black/30 px-5 py-4">
              <Search className="h-5 w-5 text-cyan-300" />
              <input
                type="text"
                placeholder={copy.searchPlaceholder}
                className="w-full bg-transparent text-base text-white outline-none placeholder:text-white/35"
              />
            </div>
            <div className="rounded-[1.5rem] border border-white/10 bg-black/30 px-5 py-4 text-sm text-white/70">
              {copy.filtersNote}
            </div>
          </div>
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
