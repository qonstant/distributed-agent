"use client"

import type React from "react"
import { useEffect, useRef, useState } from "react"
import { useLanguage } from "@/components/language-provider"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Rocket } from "lucide-react"

export function SignupSection() {
  const { t } = useLanguage()
  const [email, setEmail] = useState("")
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle")
  const [scriptUrl, setScriptUrl] = useState<string | null>(null)
  const [scriptError, setScriptError] = useState<string | null>(null)

  const formRef = useRef<HTMLFormElement | null>(null)
  const iframeRef = useRef<HTMLIFrameElement | null>(null)

  // Read the env var at runtime (inside useEffect to avoid hydration mismatches)
  useEffect(() => {
    const raw = (process.env.NEXT_PUBLIC_SCRIPT_URL as string | undefined) ?? (typeof window !== "undefined" ? (window as any).__NEXT_PUBLIC_SCRIPT_URL : undefined)

    if (!raw) {
      setScriptUrl(null)
      setScriptError("Signup endpoint not configured.")
      return
    }

    try {
      // validate URL
      // (URL throws if invalid)
      new URL(raw)
      setScriptUrl(raw)
      setScriptError(null)
    } catch {
      setScriptUrl(null)
      setScriptError("Configured signup URL is not a valid URL.")
    }
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!scriptUrl) {
      setScriptError("Signup endpoint not configured or invalid.")
      setStatus("error")
      setTimeout(() => setStatus("idle"), 3000)
      return
    }

    setStatus("loading")
    console.log("[v0] Form submitting with email:", email)
    console.log("[v0] Timestamp:", new Date().toISOString())

    try {
      // Set timestamp field value before submit
      if (formRef.current) {
        const tsField = formRef.current.querySelector<HTMLInputElement>('input[name="timestamp"]')
        if (tsField) tsField.value = new Date().toISOString()
        formRef.current.submit()
      } else {
        console.error("[v0] Form ref missing")
      }

      // Keep the old behavior: assume success after a short delay because response is via iframe
      setTimeout(() => {
        setStatus("success")
        setEmail("")
        setTimeout(() => setStatus("idle"), 5000)
      }, 1000)
    } catch (error) {
      console.error("[v0] Signup error:", error)
      setStatus("error")
      setTimeout(() => setStatus("idle"), 3000)
    }
  }

  return (
    <section id="signup" className="py-20 px-4">
      <div className="max-w-2xl mx-auto">
        <Card className="p-8 md:p-12 space-y-6 bg-gradient-to-br from-background to-accent/5">
          <div className="text-center space-y-4">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-accent/10 border border-accent rounded-full text-accent">
              <Rocket className="w-4 h-4" />
              <span className="text-sm font-medium">{t("launching_soon")}</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-pretty">{t("signup_title")}</h2>
            <p className="text-muted-foreground text-lg">{t("signup_subtitle")}</p>
          </div>

          {/* show configuration error to admins/devs */}
          {scriptError && (
            <div className="p-3 text-sm bg-yellow-50 border border-yellow-200 text-yellow-800 rounded">
              {scriptError} {process.env.NODE_ENV === "development" && "(set NEXT_PUBLIC_SCRIPT_URL in .env.local)"}
            </div>
          )}

          <form
            ref={formRef}
            onSubmit={handleSubmit}
            action={scriptUrl ?? "#"}
            method="POST"
            target="hidden_iframe"
            className="space-y-4"
          >
            {/* the visible input is named so the form posts the email directly */}
            <Input
              name="email"
              type="email"
              placeholder={t("email_placeholder")}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              disabled={status === "loading"}
              className="h-12 text-base"
            />

            {/* timestamp hidden field is set right before submit */}
            <input type="hidden" name="timestamp" value={new Date().toISOString()} />

            <div className="space-y-4">
              <Button type="submit" className="w-full h-12 text-base" disabled={status === "loading" || !scriptUrl}>
                {status === "loading" ? t("submitting") : t("submit_button")}
              </Button>
            </div>
          </form>

          <iframe ref={iframeRef} name="hidden_iframe" style={{ display: "none" }} title="Hidden form target" />

          {status === "success" && (
            <div className="p-4 bg-accent/10 border border-accent rounded-md text-accent text-center">
              {t("success_message")}
            </div>
          )}

          {status === "error" && (
            <div className="p-4 bg-destructive/10 border border-destructive rounded-md text-destructive text-center">
              {t("error_message")}
            </div>
          )}
        </Card>
      </div>
    </section>
  )
}
