"use client"

import { useLanguage } from "@/components/language-provider"
import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"

export function HeroSection() {
  const { t } = useLanguage()

  const scrollToSignup = () => {
    document.getElementById("signup")?.scrollIntoView({ behavior: "smooth" })
  }

  return (
    <section className="relative min-h-screen flex items-center justify-center px-4 py-20">
      <div className="max-w-5xl mx-auto text-center space-y-8">
        <div className="space-y-4">
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-balance">
            {t("hero_title")} <span className="text-accent">{t("hero_title_highlight")}</span>
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed text-pretty">
            {t("hero_subtitle")}
          </p>
        </div>
        <div className="flex gap-4 justify-center">
          <Button size="lg" onClick={scrollToSignup} className="gap-2">
            {t("hero_cta")} <ArrowRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </section>
  )
}
