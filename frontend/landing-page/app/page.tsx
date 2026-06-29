"use client"

import { HeroSection } from "@/components/hero-section"
import { UniversityLogos } from "@/components/university-logos"
import { FeaturesSection } from "@/components/features-section"
import { SignupSection } from "@/components/signup-section"
import { FAQSection } from "@/components/faq-section"
import { Footer } from "@/components/footer"
import { GDPRConsent } from "@/components/gdpr-consent"

export default function HomePage() {
  return (
    <main className="min-h-screen">
      <HeroSection />
      <UniversityLogos />
      <FeaturesSection />
      <SignupSection />
      <FAQSection />
      <Footer />
      <GDPRConsent />
    </main>
  )
}
