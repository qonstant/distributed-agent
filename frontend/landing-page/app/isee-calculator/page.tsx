"use client"

import Link from "next/link"
import { useState } from "react"
import { Calculator, CheckCircle2, CircleDollarSign, FileText, Home, ShieldAlert, Users } from "lucide-react"
import { Footer } from "@/components/footer"
import { useLanguage } from "@/components/language-provider"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

const SQM_VALUE_EUR = 500

const BASE_SCALE_COEFFICIENTS: Record<number, number> = {
  1: 1,
  2: 1.57,
  3: 2.04,
  4: 2.46,
  5: 2.85,
}

const exampleValues = {
  grossIncome: "10000",
  realEstateSqm: "80",
  movableAssets: "5000",
  familyMembers: "4",
  disabledMembers: "0",
}

const calculatorCopy = {
  en: {
    pill: "Equivalent ISEE Calculator",
    title: "Calculate your equivalent ISEE in a few inputs.",
    intro:
      "This public tool uses the formula for students residing abroad with family residing abroad: family gross income, plus 20% of real-estate value, plus 20% of movable assets, divided by the equivalent scale coefficient.",
    formulaLabel: "Formula",
    formulaEquals: "Equivalent ISEE =",
    formulaNumerator:
      "Gross Income + 20% of Real Estate + 20% of Movable Assets",
    formulaDenominator: "Scale Coefficient",
    formulaNote: "Real-estate value is estimated as",
    formulaNoteStrong: "1 square meter = 500 EUR",
    liveResult: "Live result",
    liveResultDesc: "Pre-filled with the official example so you can see how it works immediately.",
    resultLabel: "Equivalent ISEE",
    resultLive: "This estimate updates live as you edit the values.",
    resultNeedMembers: "Enter at least the family members count to compute the result.",
    breakdownGrossIncome: "Gross family income",
    breakdownRealEstateValue: "Real-estate value",
    breakdownWeightedRealEstate: "20% of real-estate value",
    breakdownWeightedMovable: "20% of movable assets",
    breakdownNumerator: "Numerator total",
    breakdownScale: "Scale coefficient",
    formTitle: "Enter your family data",
    formDesc:
      "Keep it simple: total square meters for family-owned real estate and total movable assets in euros.",
    grossIncomeLabel: "Gross family income (EUR)",
    grossIncomeHelp: "This is the total income of all family members combined for the year.",
    realEstateLabel: "Real-estate area (sqm)",
    realEstateHelp: "Enter the total square meters of real estate owned by all family members.",
    movableAssetsLabel: "Movable assets (EUR)",
    movableAssetsHelp: "This is the total amount of the family's money in bank accounts, savings, and similar financial assets.",
    familyMembersLabel: "Family members",
    familyMembersHelp: "Count everyone included in the family household for the calculation.",
    disabledMembersLabel: "Family members with disabilities",
    disabledMembersHelp: "The scale coefficient increases by",
    disabledMembersHelpStrong: "0.5",
    disabledMembersHelpTail: "for each family member with disabilities.",
    loadExample: "Load official example",
    clearAll: "Clear all",
    scaleRulesTitle: "Scale coefficient rules",
    scaleRulesDesc: "The coefficient depends on household size and disability adjustments.",
    memberSingle: "family member",
    memberPlural: "family members",
    scaleRulesExtraLead: "From the",
    scaleRulesExtraLeadStrong: "6th family member onwards",
    scaleRulesExtraMid: "add",
    scaleRulesExtraMidStrong: "0.35",
    scaleRulesExtraTail: "for each additional member.",
    scaleRulesDisabledLead: "For each family member with disabilities, add",
    scaleRulesDisabledStrong: "0.5",
    importantNote: "Important note",
    importantText:
      "This is a helpful estimate, not an official ISEE certificate. Final eligibility can still depend on documents, translations, valuation details, and local university or DSU rules.",
    checklist: [
      "Gross family income for the year",
      "Total family real-estate square meters",
      "Movable assets such as bank balances or savings",
      "Total household members",
      "How many household members have disabilities",
    ],
    ctaScholarship: "Get full scholarship help",
    ctaUniversities: "Explore universities",
  },
  ru: {
    pill: "Калькулятор эквивалентного ISEE",
    title: "Рассчитайте свой эквивалентный ISEE за несколько шагов.",
    intro:
      "Этот публичный инструмент использует формулу для студентов, проживающих за границей вместе с семьей за границей: валовый доход семьи плюс 20% стоимости недвижимости плюс 20% движимого имущества, разделенные на коэффициент эквивалентной шкалы.",
    formulaLabel: "Формула",
    formulaEquals: "Эквивалентный ISEE =",
    formulaNumerator:
      "Валовый доход + 20% недвижимости + 20% движимого имущества",
    formulaDenominator: "Коэффициент шкалы",
    formulaNote: "Стоимость недвижимости оценивается как",
    formulaNoteStrong: "1 квадратный метр = 500 EUR",
    liveResult: "Результат в реальном времени",
    liveResultDesc: "Страница уже заполнена официальным примером, чтобы вы сразу увидели, как это работает.",
    resultLabel: "Эквивалентный ISEE",
    resultLive: "Оценка обновляется сразу, как только вы меняете значения.",
    resultNeedMembers: "Введите хотя бы количество членов семьи, чтобы выполнить расчет.",
    breakdownGrossIncome: "Валовый доход семьи",
    breakdownRealEstateValue: "Стоимость недвижимости",
    breakdownWeightedRealEstate: "20% стоимости недвижимости",
    breakdownWeightedMovable: "20% движимого имущества",
    breakdownNumerator: "Сумма числителя",
    breakdownScale: "Коэффициент шкалы",
    formTitle: "Введите данные семьи",
    formDesc:
      "Все максимально просто: укажите общую площадь семейной недвижимости и общую сумму движимого имущества в евро.",
    grossIncomeLabel: "Валовый доход семьи (EUR)",
    grossIncomeHelp: "Это общий доход всех членов семьи за год.",
    realEstateLabel: "Площадь недвижимости (кв. м)",
    realEstateHelp: "Укажите общую площадь недвижимости, принадлежащей всем членам семьи.",
    movableAssetsLabel: "Движимое имущество (EUR)",
    movableAssetsHelp: "Это общая сумма денег семьи на банковских счетах, сбережениях и других похожих финансовых активах.",
    familyMembersLabel: "Члены семьи",
    familyMembersHelp: "Укажите всех людей, которые входят в состав семьи для этого расчета.",
    disabledMembersLabel: "Члены семьи с инвалидностью",
    disabledMembersHelp: "Коэффициент шкалы увеличивается на",
    disabledMembersHelpStrong: "0.5",
    disabledMembersHelpTail: "за каждого члена семьи с инвалидностью.",
    loadExample: "Загрузить официальный пример",
    clearAll: "Очистить все",
    scaleRulesTitle: "Правила коэффициента шкалы",
    scaleRulesDesc: "Коэффициент зависит от размера семьи и дополнительных надбавок за инвалидность.",
    memberSingle: "член семьи",
    memberPlural: "члена семьи",
    scaleRulesExtraLead: "Начиная с",
    scaleRulesExtraLeadStrong: "6-го члена семьи",
    scaleRulesExtraMid: "добавляется",
    scaleRulesExtraMidStrong: "0.35",
    scaleRulesExtraTail: "за каждого следующего члена семьи.",
    scaleRulesDisabledLead: "За каждого члена семьи с инвалидностью добавляется",
    scaleRulesDisabledStrong: "0.5",
    importantNote: "Важное примечание",
    importantText:
      "Это полезная оценка, а не официальный сертификат ISEE. Окончательный результат все равно может зависеть от документов, переводов, деталей оценки имущества и правил конкретного университета или DSU.",
    checklist: [
      "Годовой валовый доход семьи",
      "Общая площадь семейной недвижимости",
      "Движимое имущество: счета, накопления и т.д.",
      "Общее количество членов семьи",
      "Сколько членов семьи имеют инвалидность",
    ],
    ctaScholarship: "Получить полную помощь по стипендии",
    ctaUniversities: "Смотреть университеты",
  },
  kk: {
    pill: "Эквивалентті ISEE калькуляторы",
    title: "Эквивалентті ISEE көрсеткішін бірнеше қадамда есептеңіз.",
    intro:
      "Бұл ашық құрал шетелде тұратын студенттер мен шетелде тұратын отбасыға арналған формуланы қолданады: отбасының жалпы табысы, оған жылжымайтын мүлік құнының 20%-ы және жылжымалы активтердің 20%-ы қосылып, эквивалент шкаласының коэффициентіне бөлінеді.",
    formulaLabel: "Формула",
    formulaEquals: "Эквивалентті ISEE =",
    formulaNumerator:
      "Жалпы табыс + Жылжымайтын мүліктің 20%-ы + Жылжымалы активтердің 20%-ы",
    formulaDenominator: "Шкала коэффициенті",
    formulaNote: "Жылжымайтын мүлік құны былай есептеледі:",
    formulaNoteStrong: "1 шаршы метр = 500 EUR",
    liveResult: "Нәтиже бірден көрінеді",
    liveResultDesc: "Ресми мысал бірден толтырылған, сондықтан есептеудің қалай жұмыс істейтінін дереу көресіз.",
    resultLabel: "Эквивалентті ISEE",
    resultLive: "Мәндерді өзгерткен сайын есептеу бірден жаңарады.",
    resultNeedMembers: "Есептеу үшін кемінде отбасы мүшелерінің санын енгізіңіз.",
    breakdownGrossIncome: "Отбасының жалпы табысы",
    breakdownRealEstateValue: "Жылжымайтын мүлік құны",
    breakdownWeightedRealEstate: "Жылжымайтын мүлік құнының 20%-ы",
    breakdownWeightedMovable: "Жылжымалы активтердің 20%-ы",
    breakdownNumerator: "Алымның жалпы сомасы",
    breakdownScale: "Шкала коэффициенті",
    formTitle: "Отбасы деректерін енгізіңіз",
    formDesc:
      "Барынша жеңіл еттік: отбасы иелігіндегі мүліктің жалпы шаршы метрін және барлық жылжымалы активтерді еуромен енгізіңіз.",
    grossIncomeLabel: "Отбасының жалпы табысы (EUR)",
    grossIncomeHelp: "Бұл отбасының барлық мүшелерінің бір жылдағы жиынтық табысы.",
    realEstateLabel: "Жылжымайтын мүлік ауданы (ш.м.)",
    realEstateHelp: "Барлық отбасы мүшелеріне тиесілі жылжымайтын мүліктің жалпы шаршы метрін енгізіңіз.",
    movableAssetsLabel: "Жылжымалы активтер (EUR)",
    movableAssetsHelp: "Бұл отбасының банк шоттарындағы ақшасының, жинақтарының және ұқсас қаржылық активтерінің жалпы сомасы.",
    familyMembersLabel: "Отбасы мүшелері",
    familyMembersHelp: "Осы есепке кіретін отбасының барлық мүшелерін санаңыз.",
    disabledMembersLabel: "Мүгедектігі бар отбасы мүшелері",
    disabledMembersHelp: "Шкала коэффициенті",
    disabledMembersHelpStrong: "0.5",
    disabledMembersHelpTail: "мүгедектігі бар әрбір отбасы мүшесі үшін өседі.",
    loadExample: "Ресми мысалды жүктеу",
    clearAll: "Барлығын тазарту",
    scaleRulesTitle: "Шкала коэффициентінің ережелері",
    scaleRulesDesc: "Коэффициент отбасы санына және мүгедектікке байланысты қосымша өсімге тәуелді.",
    memberSingle: "отбасы мүшесі",
    memberPlural: "отбасы мүшесі",
    scaleRulesExtraLead: "",
    scaleRulesExtraLeadStrong: "6-шы отбасы мүшесінен бастап",
    scaleRulesExtraMid: "",
    scaleRulesExtraMidStrong: "0.35",
    scaleRulesExtraTail: "әрбір қосымша адам үшін қосылады.",
    scaleRulesDisabledLead: "Мүгедектігі бар әрбір отбасы мүшесіне",
    scaleRulesDisabledStrong: "0.5",
    importantNote: "Маңызды ескерту",
    importantText:
      "Бұл ресми ISEE сертификаты емес, тек пайдалы алдын ала есеп. Соңғы нәтиже құжаттарға, аудармаларға, мүлікті бағалау егжей-тегжейіне және университет не DSU талаптарына байланысты өзгеруі мүмкін.",
    checklist: [
      "Отбасының жылдық жалпы табысы",
      "Отбасы иелігіндегі жылжымайтын мүліктің жалпы шаршы метрі",
      "Банк шоттары, жинақ және басқа жылжымалы активтер",
      "Отбасы мүшелерінің жалпы саны",
      "Мүгедектігі бар отбасы мүшелерінің саны",
    ],
    ctaScholarship: "Стипендия бойынша толық көмек алу",
    ctaUniversities: "Университеттерді көру",
  },
} as const

function parsePositiveNumber(value: string) {
  const normalized = value.replace(",", ".").trim()
  if (!normalized) return 0
  const parsed = Number(normalized)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0
}

function parseWholeNumber(value: string) {
  const parsed = Math.floor(parsePositiveNumber(value))
  return parsed > 0 ? parsed : 0
}

function getScaleCoefficient(familyMembers: number, disabledMembers: number) {
  if (familyMembers <= 0) return 0

  let coefficient =
    familyMembers <= 5
      ? BASE_SCALE_COEFFICIENTS[familyMembers]
      : BASE_SCALE_COEFFICIENTS[5] + 0.35 * (familyMembers - 5)

  coefficient += disabledMembers * 0.5
  return coefficient
}

function formatMoney(value: number, locale: string) {
  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 2,
  }).format(value)
}

function formatNumber(value: number, locale: string) {
  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  }).format(value)
}

function BreakdownRow({
  label,
  value,
  subtle,
}: {
  label: string
  value: string
  subtle?: boolean
}) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-2xl border border-white/8 bg-black/20 px-4 py-3">
      <span className={`text-sm ${subtle ? "text-white/55" : "text-white/75"}`}>{label}</span>
      <span className={`text-right text-sm font-semibold ${subtle ? "text-white/70" : "text-white"}`}>{value}</span>
    </div>
  )
}

export default function IseeCalculatorPage() {
  const { language } = useLanguage()
  const copy = calculatorCopy[language]
  const locale = language === "ru" ? "ru-RU" : language === "kk" ? "kk-KZ" : "en-US"
  const [grossIncome, setGrossIncome] = useState(exampleValues.grossIncome)
  const [realEstateSqm, setRealEstateSqm] = useState(exampleValues.realEstateSqm)
  const [movableAssets, setMovableAssets] = useState(exampleValues.movableAssets)
  const [familyMembers, setFamilyMembers] = useState(exampleValues.familyMembers)
  const [disabledMembers, setDisabledMembers] = useState(exampleValues.disabledMembers)

  const grossIncomeValue = parsePositiveNumber(grossIncome)
  const realEstateSqmValue = parsePositiveNumber(realEstateSqm)
  const movableAssetsValue = parsePositiveNumber(movableAssets)
  const familyMembersValue = parseWholeNumber(familyMembers)
  const disabledMembersRaw = parseWholeNumber(disabledMembers)
  const disabledMembersValue =
    familyMembersValue > 0 ? Math.min(disabledMembersRaw, familyMembersValue) : 0

  const realEstateValue = realEstateSqmValue * SQM_VALUE_EUR
  const weightedRealEstateValue = realEstateValue * 0.2
  const weightedMovableAssetsValue = movableAssetsValue * 0.2
  const numerator = grossIncomeValue + weightedRealEstateValue + weightedMovableAssetsValue
  const scaleCoefficient = getScaleCoefficient(familyMembersValue, disabledMembersValue)
  const equivalentIsee = scaleCoefficient > 0 ? numerator / scaleCoefficient : 0

  const loadExample = () => {
    setGrossIncome(exampleValues.grossIncome)
    setRealEstateSqm(exampleValues.realEstateSqm)
    setMovableAssets(exampleValues.movableAssets)
    setFamilyMembers(exampleValues.familyMembers)
    setDisabledMembers(exampleValues.disabledMembers)
  }

  const clearAll = () => {
    setGrossIncome("")
    setRealEstateSqm("")
    setMovableAssets("")
    setFamilyMembers("")
    setDisabledMembers("")
  }

  return (
    <main className="min-h-screen pt-32">
      <section className="px-4 pb-10 pt-6">
        <div className="mx-auto max-w-6xl rounded-[2rem] border border-white/10 bg-white/[0.04] px-6 py-8 shadow-[0_30px_90px_-40px_rgba(0,0,0,0.95)] md:px-10 md:py-10">
          <div className="grid gap-8 md:grid-cols-[1.15fr_0.85fr]">
            <div className="space-y-6">
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/25 bg-cyan-400/10 px-4 py-2 text-sm font-medium text-cyan-200">
                <Calculator className="h-4 w-4" />
                {copy.pill}
              </div>

              <div className="space-y-4">
                <h1 className="text-4xl font-bold tracking-tight text-white md:text-6xl">
                  {copy.title}
                </h1>
                <p className="max-w-3xl text-lg leading-relaxed text-white/70">
                  {copy.intro}
                </p>
              </div>

              <div className="rounded-[1.75rem] border border-white/10 bg-black/25 p-5">
                <div className="text-sm uppercase tracking-[0.24em] text-cyan-200/80">{copy.formulaLabel}</div>
                <div className="mt-4 flex flex-col gap-4 text-white md:flex-row md:items-center">
                  <div className="text-lg font-semibold md:shrink-0">{copy.formulaEquals}</div>
                  <div className="flex-1 rounded-[1.25rem] border border-white/8 bg-white/[0.03] px-4 py-4">
                    <div className="text-center text-sm leading-6 text-white/88 md:text-base">
                      {copy.formulaNumerator}
                    </div>
                    <div className="my-3 h-px w-full bg-linear-to-r from-transparent via-cyan-300/70 to-transparent" />
                    <div className="text-center text-sm font-semibold uppercase tracking-[0.18em] text-cyan-100/88 md:text-[0.95rem]">
                      {copy.formulaDenominator}
                    </div>
                  </div>
                </div>
                <p className="mt-3 text-sm leading-6 text-white/60">
                  {copy.formulaNote} <span className="font-semibold text-white/85">{copy.formulaNoteStrong}</span>.
                </p>
              </div>
            </div>

            <Card className="rounded-[2rem] border-white/10 bg-card/75 shadow-[0_26px_80px_-40px_rgba(0,0,0,0.95)]">
              <CardHeader className="px-6 pt-6">
                <CardTitle className="text-2xl text-white">{copy.liveResult}</CardTitle>
                <CardDescription className="text-white/55">
                  {copy.liveResultDesc}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 px-6 pb-6">
                <div className="rounded-[1.75rem] border border-cyan-300/15 bg-linear-to-br from-cyan-300/14 via-sky-400/10 to-blue-500/14 p-5">
                  <div className="text-sm uppercase tracking-[0.24em] text-cyan-200/85">{copy.resultLabel}</div>
                  <div className="mt-3 text-4xl font-bold tracking-tight text-white">
                    {scaleCoefficient > 0 ? formatMoney(equivalentIsee, locale) : "--"}
                  </div>
                  <p className="mt-3 text-sm text-white/65">
                    {scaleCoefficient > 0
                      ? copy.resultLive
                      : copy.resultNeedMembers}
                  </p>
                </div>

                <BreakdownRow label={copy.breakdownGrossIncome} value={formatMoney(grossIncomeValue, locale)} />
                <BreakdownRow label={copy.breakdownRealEstateValue} value={formatMoney(realEstateValue, locale)} subtle />
                <BreakdownRow label={copy.breakdownWeightedRealEstate} value={formatMoney(weightedRealEstateValue, locale)} />
                <BreakdownRow label={copy.breakdownWeightedMovable} value={formatMoney(weightedMovableAssetsValue, locale)} />
                <BreakdownRow label={copy.breakdownNumerator} value={formatMoney(numerator, locale)} />
                <BreakdownRow label={copy.breakdownScale} value={scaleCoefficient > 0 ? formatNumber(scaleCoefficient, locale) : "--"} />
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      <section className="px-4 pb-8">
        <div className="mx-auto grid max-w-6xl gap-6 lg:grid-cols-[1fr_0.9fr]">
            <Card className="rounded-[2rem] border-white/10 bg-white/[0.035]">
              <CardHeader className="px-6 pt-6">
                <CardTitle className="text-2xl text-white">{copy.formTitle}</CardTitle>
                <CardDescription className="text-white/55">
                  {copy.formDesc}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5 px-6 pb-6">
                <div className="grid gap-5 md:grid-cols-2">
                  <label className="block">
                    <span className="mb-2 flex items-center gap-2 text-sm font-medium text-white/80">
                      <CircleDollarSign className="h-4 w-4 text-cyan-300" />
                      {copy.grossIncomeLabel}
                    </span>
                  <Input
                    inputMode="decimal"
                    placeholder="10000"
                    value={grossIncome}
                    onChange={(event) => setGrossIncome(event.target.value)}
                    className="h-12 rounded-2xl border-white/10 bg-black/25 px-4 text-white placeholder:text-white/30"
                  />
                  <p className="mt-2 text-xs leading-5 text-white/45">{copy.grossIncomeHelp}</p>
                </label>

                <label className="block">
                    <span className="mb-2 flex items-center gap-2 text-sm font-medium text-white/80">
                      <Home className="h-4 w-4 text-cyan-300" />
                      {copy.realEstateLabel}
                    </span>
                  <Input
                    inputMode="decimal"
                    placeholder="80"
                    value={realEstateSqm}
                    onChange={(event) => setRealEstateSqm(event.target.value)}
                    className="h-12 rounded-2xl border-white/10 bg-black/25 px-4 text-white placeholder:text-white/30"
                  />
                  <p className="mt-2 text-xs leading-5 text-white/45">{copy.realEstateHelp}</p>
                </label>

                <label className="block">
                    <span className="mb-2 flex items-center gap-2 text-sm font-medium text-white/80">
                      <FileText className="h-4 w-4 text-cyan-300" />
                      {copy.movableAssetsLabel}
                    </span>
                  <Input
                    inputMode="decimal"
                    placeholder="5000"
                    value={movableAssets}
                    onChange={(event) => setMovableAssets(event.target.value)}
                    className="h-12 rounded-2xl border-white/10 bg-black/25 px-4 text-white placeholder:text-white/30"
                  />
                  <p className="mt-2 text-xs leading-5 text-white/45">{copy.movableAssetsHelp}</p>
                </label>

                <label className="block">
                    <span className="mb-2 flex items-center gap-2 text-sm font-medium text-white/80">
                      <Users className="h-4 w-4 text-cyan-300" />
                      {copy.familyMembersLabel}
                    </span>
                  <Input
                    inputMode="numeric"
                    placeholder="4"
                    value={familyMembers}
                    onChange={(event) => setFamilyMembers(event.target.value)}
                    className="h-12 rounded-2xl border-white/10 bg-black/25 px-4 text-white placeholder:text-white/30"
                  />
                  <p className="mt-2 text-xs leading-5 text-white/45">{copy.familyMembersHelp}</p>
                </label>
              </div>

              <label className="block">
                <span className="mb-2 flex items-center gap-2 text-sm font-medium text-white/80">
                  <CheckCircle2 className="h-4 w-4 text-cyan-300" />
                  {copy.disabledMembersLabel}
                </span>
                <Input
                  inputMode="numeric"
                  placeholder="0"
                  value={disabledMembers}
                  onChange={(event) => setDisabledMembers(event.target.value)}
                  className="h-12 rounded-2xl border-white/10 bg-black/25 px-4 text-white placeholder:text-white/30"
                />
                <p className="mt-2 text-xs text-white/45">
                  {copy.disabledMembersHelp} <span className="font-semibold text-white/75">{copy.disabledMembersHelpStrong}</span> {copy.disabledMembersHelpTail}
                </p>
              </label>

              <div className="flex flex-wrap gap-3 pt-1">
                <Button
                  onClick={loadExample}
                  className="rounded-full bg-linear-to-r from-cyan-300 via-sky-400 to-blue-500 text-black hover:opacity-95"
                >
                  {copy.loadExample}
                </Button>
                <Button
                  variant="outline"
                  onClick={clearAll}
                  className="rounded-full border-white/15 bg-transparent text-white hover:bg-white/8"
                >
                  {copy.clearAll}
                </Button>
              </div>
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card className="rounded-[2rem] border-white/10 bg-white/[0.035]">
              <CardHeader className="px-6 pt-6">
                <CardTitle className="text-2xl text-white">{copy.scaleRulesTitle}</CardTitle>
                <CardDescription className="text-white/55">
                  {copy.scaleRulesDesc}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 px-6 pb-6">
                {[1, 2, 3, 4, 5].map((members) => (
                  <BreakdownRow
                    key={members}
                    label={`${members} ${members > 1 ? copy.memberPlural : copy.memberSingle}`}
                    value={formatNumber(BASE_SCALE_COEFFICIENTS[members], locale)}
                  />
                ))}
                <div className="rounded-2xl border border-white/8 bg-black/20 px-4 py-4 text-sm leading-6 text-white/68">
                  {copy.scaleRulesExtraLead} <span className="font-semibold text-white">{copy.scaleRulesExtraLeadStrong}</span> {copy.scaleRulesExtraMid}{" "}
                  <span className="font-semibold text-white">{copy.scaleRulesExtraMidStrong}</span> {copy.scaleRulesExtraTail}
                  <br />
                  {copy.scaleRulesDisabledLead} <span className="font-semibold text-white">{copy.scaleRulesDisabledStrong}</span>.
                </div>
              </CardContent>
            </Card>

            <Card className="rounded-[2rem] border-white/10 bg-white/[0.035]">
              <CardHeader className="px-6 pt-6">
                <CardTitle className="flex items-center gap-2 text-2xl text-white">
                  <ShieldAlert className="h-5 w-5 text-amber-300" />
                  {copy.importantNote}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-5 px-6 pb-6">
                <p className="text-sm leading-6 text-white/65">
                  {copy.importantText}
                </p>

                <div className="space-y-3">
                  {copy.checklist.map((item) => (
                    <div
                      key={item}
                      className="flex items-center gap-3 rounded-2xl border border-white/8 bg-black/20 px-4 py-3"
                    >
                      <CheckCircle2 className="h-4 w-4 text-cyan-300" />
                      <span className="text-sm text-white/75">{item}</span>
                    </div>
                  ))}
                </div>

                <div className="flex flex-wrap gap-3">
                  <Button asChild className="rounded-full bg-white text-black hover:bg-white/90">
                    <Link href="/#signup">{copy.ctaScholarship}</Link>
                  </Button>
                  <Button asChild variant="outline" className="rounded-full border-white/15 bg-transparent text-white hover:bg-white/8">
                    <Link href="/universities">{copy.ctaUniversities}</Link>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  )
}
