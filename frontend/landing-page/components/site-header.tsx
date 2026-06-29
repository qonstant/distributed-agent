"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { LanguageSwitcher } from "@/components/language-provider"
import { cn } from "@/lib/utils"

const navItems = [
  { href: "/", label: "Home" },
  { href: "/universities", label: "Universities" },
  { href: "/isee-calculator", label: "ISEE Calculator" },
  { href: "/#faq", label: "FAQ" },
]

export function SiteHeader() {
  const pathname = usePathname()

  return (
    <header className="fixed inset-x-0 top-0 z-50 px-4 pt-4">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 rounded-[2rem] border border-white/10 bg-black/45 px-4 py-3 shadow-[0_24px_80px_-28px_rgba(0,0,0,0.85)] backdrop-blur-xl md:px-6">
        <Link
          href="/"
          className="flex items-center gap-2 rounded-full px-2 py-1 text-lg font-semibold tracking-tight text-white transition hover:text-accent"
        >
          <span className="flex h-9 w-9 items-center justify-center rounded-full bg-linear-to-br from-cyan-300 via-sky-400 to-blue-500 text-black shadow-[0_0_24px_rgba(56,189,248,0.45)]">
            <Sparkles className="h-4 w-4" />
          </span>
          <span>nomadmit</span>
        </Link>

        <nav className="order-3 flex w-full items-center justify-start gap-1 overflow-x-auto pb-1 md:order-2 md:w-auto md:justify-center md:pb-0">
          {navItems.map((item) => {
            const active = item.href === "/"
              ? pathname === "/"
              : item.href.startsWith("/#")
                ? pathname === "/"
                : pathname.startsWith(item.href)

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "shrink-0 rounded-full px-4 py-2 text-sm font-medium transition",
                  active
                    ? "bg-white text-black"
                    : "text-white/72 hover:bg-white/8 hover:text-white",
                )}
              >
                {item.label}
              </Link>
            )
          })}
        </nav>

        <div className="order-2 flex items-center gap-2 md:order-3">
          <div className="hidden md:block">
            <LanguageSwitcher />
          </div>
          <Button
            asChild
            className="rounded-full border-0 bg-linear-to-r from-cyan-300 via-sky-400 to-blue-500 px-5 text-black shadow-[0_0_30px_rgba(59,130,246,0.35)] hover:opacity-95"
          >
            <Link href="/#signup">Get Access</Link>
          </Button>
        </div>

        <div className="order-4 block md:hidden">
          <LanguageSwitcher />
        </div>
      </div>
    </header>
  )
}
