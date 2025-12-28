"use client"

import { useLanguage } from "@/components/language-provider"
import React from "react"

export function UniversityLogos() {
  const { t } = useLanguage()

  const universities = [
    { name: "University of Turin", logo: "/logos/unito.jpg" },
    { name: "University of Milan", logo: "/logos/unimi.jpg" },
    { name: "Politecnico di Torino", logo: "/logos/polito.jpg" },
    { name: "Università di Padova", logo: "/logos/padova.jpg" },
    { name: "Università di Bologna", logo: "/logos/unibo.jpg" },
    { name: "Politecnico di Milano", logo: "/logos/polimi.jpg" },
  ]

  // small helper to set fallback image on error
  const handleImgError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    e.currentTarget.src = "/logos/placeholder.svg"
  }

  return (
    <section className="py-20 px-4 bg-card/50">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-balance">
            {t("universities_title")}
          </h2>
        </div>

        <div className="relative overflow-hidden">
          <div className="flex animate-scroll">
            {/* First set of logos */}
            {universities.map((uni) => (
              <div
                key={uni.name}
                className="flex-shrink-0 flex flex-col items-center justify-center gap-3 p-6 mx-4 rounded-lg hover:bg-accent/10 hover:scale-105 transition-all duration-300 group min-w-[200px]"
              >
                <div className="relative w-24 h-24 bg-white rounded-lg p-2 flex items-center justify-center group-hover:shadow-lg group-hover:shadow-accent/20 transition-shadow">
                  <img
                    src={uni.logo}
                    alt={`${uni.name} logo`}
                    onError={handleImgError}
                    className="w-20 h-20 object-contain"
                  />
                </div>
                <div className="text-center">
                  <div className="text-sm font-semibold text-foreground group-hover:text-accent transition-colors">
                    {uni.name}
                  </div>
                </div>
              </div>
            ))}

            {/* Duplicate set for seamless loop */}
            {universities.map((uni) => (
              <div
                key={`${uni.name}-duplicate`}
                className="flex-shrink-0 flex flex-col items-center justify-center gap-3 p-6 mx-4 rounded-lg hover:bg-accent/10 hover:scale-105 transition-all duration-300 group min-w-[200px]"
              >
                <div className="relative w-24 h-24 bg-white rounded-lg p-2 flex items-center justify-center group-hover:shadow-lg group-hover:shadow-accent/20 transition-shadow">
                  <img
                    src={uni.logo}
                    alt={`${uni.name} logo`}
                    onError={handleImgError}
                    className="w-20 h-20 object-contain"
                  />
                </div>
                <div className="text-center">
                  <div className="text-sm font-semibold text-foreground group-hover:text-accent transition-colors">
                    {uni.name}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-12 text-center">
          <div className="inline-flex items-center gap-2 px-6 py-3 bg-accent/10 rounded-full">
            <span className="text-2xl font-bold text-accent">92%</span>
            <span className="text-muted-foreground">Success Rate</span>
          </div>
        </div>
      </div>
    </section>
  )
}
