import { ImageResponse } from "next/og"

export const size = {
  width: 1200,
  height: 630,
}
export const contentType = "image/png"

export default function TwitterImage() {
  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          position: "relative",
          background:
            "radial-gradient(circle at top left, rgba(34,211,238,0.28), transparent 32%), radial-gradient(circle at bottom right, rgba(59,130,246,0.24), transparent 30%), linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%)",
          color: "#f8fafc",
          fontFamily: "Arial, sans-serif",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 52,
            left: 58,
            display: "flex",
            alignItems: "center",
            gap: 18,
          }}
        >
          <div
            style={{
              width: 76,
              height: 76,
              borderRadius: 9999,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "linear-gradient(135deg, #67e8f9 0%, #38bdf8 55%, #3b82f6 100%)",
              color: "#020617",
              fontSize: 34,
              fontWeight: 800,
              boxShadow: "0 0 54px rgba(56,189,248,0.38)",
            }}
          >
            N
          </div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ fontSize: 34, fontWeight: 700, letterSpacing: -0.8 }}>nomadmit</div>
            <div style={{ fontSize: 19, color: "rgba(226,232,240,0.72)" }}>
              European education guidance
            </div>
          </div>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            padding: "0 70px",
            width: "100%",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              marginBottom: 24,
              color: "#67e8f9",
              fontSize: 20,
              letterSpacing: 4,
              textTransform: "uppercase",
            }}
          >
            Study in Europe
          </div>
          <div
            style={{
              fontSize: 64,
              lineHeight: 1.06,
              fontWeight: 800,
              letterSpacing: -2.6,
              maxWidth: 820,
              textWrap: "balance" as const,
            }}
          >
            Your path to European education
          </div>
          <div
            style={{
              marginTop: 24,
              maxWidth: 760,
              fontSize: 28,
              lineHeight: 1.35,
              color: "rgba(226,232,240,0.82)",
            }}
          >
            Admissions, scholarships, visa support, and university discovery for international
            students applying to top Italian and European universities.
          </div>
        </div>

        <div
          style={{
            position: "absolute",
            right: 62,
            bottom: 58,
            display: "flex",
            gap: 14,
            fontSize: 22,
            color: "rgba(248,250,252,0.82)",
          }}
        >
          <div
            style={{
              borderRadius: 9999,
              border: "1px solid rgba(255,255,255,0.14)",
              background: "rgba(255,255,255,0.05)",
              padding: "12px 22px",
            }}
          >
            University search
          </div>
          <div
            style={{
              borderRadius: 9999,
              border: "1px solid rgba(103,232,249,0.24)",
              background: "rgba(103,232,249,0.08)",
              padding: "12px 22px",
              color: "#a5f3fc",
            }}
          >
            Italy-focused
          </div>
        </div>
      </div>
    ),
    size,
  )
}
