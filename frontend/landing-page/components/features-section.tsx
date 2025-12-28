"use client"

import { useLanguage } from "./language-provider"
import { GraduationCap, FileText, Briefcase, MessageSquare, Bot } from "lucide-react"

const content = {
  en: {
    title: "One-Stop Solution for All Things University Applications",
    features: [
      {
        icon: GraduationCap,
        title: "Find the Perfect Colleges for You",
        description: "Craft a balanced list of target universities based on your grades, goals and preferences.",
      },
      {
        icon: FileText,
        title: "Write a Strong Essay",
        description: "Generate ideas, drafts and get clear feedback with suggestions how to improve your story.",
      },
      {
        icon: Briefcase,
        title: "Build a Strong Profile",
        description: "Identify what's missing, find internships and frame your activities in the best way possible.",
      },
      {
        icon: MessageSquare,
        title: "Practice for Every University Interview",
        description: "Run mock interviews with our agent that knows 8/10 questions you're likely to get.",
      },
      {
        icon: Bot,
        title: "Be in Touch with AI Mentor",
        description: "Who will send all the needed documents and guide you through the entire application process.",
      },
    ],
  },
  ru: {
    title: "Универсальное решение для поступления в университеты",
    features: [
      {
        icon: GraduationCap,
        title: "Найдите Идеальные Университеты",
        description:
          "Составьте сбалансированный список целевых университетов на основе ваших оценок, целей и предпочтений.",
      },
      {
        icon: FileText,
        title: "Напишите Сильное Эссе",
        description:
          "Генерируйте идеи, черновики и получайте четкую обратную связь с предложениями по улучшению вашей истории.",
      },
      {
        icon: Briefcase,
        title: "Создайте Сильный Профиль",
        description:
          "Определите, чего не хватает, найдите стажировки и представьте свою деятельность наилучшим образом.",
      },
      {
        icon: MessageSquare,
        title: "Практикуйтесь для Интервью",
        description:
          "Проводите тренировочные интервью с нашим агентом, который знает 8 из 10 вопросов, которые вам зададут.",
      },
      {
        icon: Bot,
        title: "Общайтесь с AI Ментором",
        description: "Который отправит все необходимые документы и проведет вас через весь процесс подачи заявки.",
      },
    ],
  },
  kk: {
    title: "Университетке түсу үшін барлық нәрсе үшін бір терезе шешімі",
    features: [
      {
        icon: GraduationCap,
        title: "Сізге арналған тамаша университеттерді табыңыз",
        description:
          "Бағаларыңызға, мақсаттарыңызға және қалауларыңызға негізделген теңдестірілген университеттер тізімін жасаңыз.",
      },
      {
        icon: FileText,
        title: "Күшті эссе жазыңыз",
        description:
          "Идеяларды, жобаларды жасаңыз және тарихыңызды жақсарту бойынша ұсыныстармен нақты кері байланыс алыңыз.",
      },
      {
        icon: Briefcase,
        title: "Күшті профиль құрыңыз",
        description:
          "Не жетіспейтінін анықтаңыз, тәжірибелерді табыңыз және қызметіңізді мүмкін болатын ең жақсы жолмен көрсетіңіз.",
      },
      {
        icon: MessageSquare,
        title: "Әр университеттік сұхбатқа дайындалыңыз",
        description: "Сізге берілуі мүмкін 10 сұрақтың 8-ін білетін агентімізбен жаттығу сұхбаттарын өткізіңіз.",
      },
      {
        icon: Bot,
        title: "AI Менторымен байланыста болыңыз",
        description: "Ол барлық қажетті құжаттарды жібереді және өтініш беру процесі арқылы сізді жетелейді.",
      },
    ],
  },
}

export function FeaturesSection() {
  const { language } = useLanguage()
  const t = content[language]

  return (
    <section className="py-20 px-4 bg-gradient-to-b from-background to-muted/20">
      <div className="max-w-7xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-16 text-balance">{t.title}</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {t.features.map((feature, index) => (
            <div
              key={index}
              className="group relative bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl p-8 hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:shadow-primary/10"
            >
              <div className="flex flex-col gap-4">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>

                <h3 className="text-xl font-semibold text-foreground">{feature.title}</h3>

                <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
              </div>

              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
