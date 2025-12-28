"use client"

import { HeroSection } from "@/components/hero-section"
import { UniversityLogos } from "@/components/university-logos"
import { FeaturesSection } from "@/components/features-section"
import { SignupSection } from "@/components/signup-section"
import { AIChatWidget } from "@/components/ai-chat-widget"
import { FAQSection } from "@/components/faq-section"
import { Footer } from "@/components/footer"
import { GDPRConsent } from "@/components/gdpr-consent"
import { LanguageProvider } from "@/components/language-provider"

export default function HomePage() {
  return (
    <LanguageProvider>
      <main className="min-h-screen">
        <HeroSection />
        <UniversityLogos />
        <FeaturesSection />
        <SignupSection />
        <FAQSection />
        <AIChatWidget />
        <Footer />
        <GDPRConsent />
      </main>
    </LanguageProvider>
  )
}
