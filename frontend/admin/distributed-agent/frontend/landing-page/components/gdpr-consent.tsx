"use client"

import { useState, useEffect } from "react"
import { useLanguage } from "@/components/language-provider"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

export function GDPRConsent() {
  const { t } = useLanguage()
  const [showConsent, setShowConsent] = useState(false)

  useEffect(() => {
    const consent = localStorage.getItem("nomadmit_gdpr_consent")
    if (!consent) {
      setShowConsent(true)
    }
  }, [])

  const handleAccept = () => {
    localStorage.setItem("nomadmit_gdpr_consent", "accepted")
    setShowConsent(false)
  }

  const handleSettings = () => {
    alert("Cookie settings would open here in production")
  }

  if (!showConsent) return null

  return (
    <div className="fixed bottom-0 left-0 right-0 p-4 z-50">
      <Card className="max-w-4xl mx-auto p-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-muted-foreground">
            {t("gdpr_text")}{" "}
            <a href="/privacy" className="text-accent hover:underline">
              {t("gdpr_privacy")}
            </a>
          </p>
          <div className="flex gap-2 flex-shrink-0">
            <Button variant="outline" onClick={handleSettings}>
              {t("gdpr_settings")}
            </Button>
            <Button onClick={handleAccept}>{t("gdpr_accept")}</Button>
          </div>
        </div>
      </Card>
    </div>
  )
}
