"use client"

import { useLanguage } from "@/components/language-provider"

export function Footer() {
  const { t } = useLanguage()

  return (
    <footer className="py-12 px-4 border-t">
      <div className="max-w-7xl mx-auto text-center">
        <p className="text-sm text-muted-foreground">{t("footer_tagline")}</p>
        <p className="text-xs text-muted-foreground mt-2">
          © {new Date().getFullYear()} nomadmit. All rights reserved.
        </p>
      </div>
    </footer>
  )
}
