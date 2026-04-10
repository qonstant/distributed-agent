import type React from "react"
import type { Metadata } from "next"
import Script from "next/script"
import { Geist } from "next/font/google"
import { Analytics } from "@vercel/analytics/next"
import "./globals.css"

const _geist = Geist({ subsets: ["latin", "cyrillic"] })

export const metadata: Metadata = {
  title: "nomadmit - Your Path to European Education",
  description:
    "Expert guidance for international students applying to top European universities. We help with admissions, scholarships, visa support, and all bureaucracy.",
  generator: "v0.app",
  icons: {
    icon: [
      { url: "/icon-light-32x32.png", media: "(prefers-color-scheme: light)" },
      { url: "/icon-dark-32x32.png", media: "(prefers-color-scheme: dark)" },
      { url: "/icon.svg", type: "image/svg+xml" },
    ],
    apple: "/apple-icon.png",
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={_geist.className}>
      <head>
        {/* Remove extension-injected attributes BEFORE React hydration */}
        <Script id="remove-ext-attrs" strategy="beforeInteractive">
          {`(function () {
            try {
              // Patterns of attribute names commonly injected by extensions
              var attrRegex = /^(data-(gr-|gr_ext|new-gr|new_|lt-|new-gr-c|new-gr-c-s)|data-gr-ext-installed|data-new-gr-c-s-check-loaded|data-lt-installed|data-gr-installed|suppresshydrationwarning)/i;

              // Helper to remove matching attributes from an element
              function removeMatchingAttrs(el) {
                if (!el || !el.attributes) return;
                var attrs = Array.prototype.slice.call(el.attributes);
                for (var i = 0; i < attrs.length; i++) {
                  try {
                    if (attrRegex.test(attrs[i].name)) el.removeAttribute(attrs[i].name);
                  } catch (e) { /* ignore attribute removal errors */ }
                }
              }

              // Clean the root html and body elements
              removeMatchingAttrs(document.documentElement);
              removeMatchingAttrs(document.body);

              // Also remove from any elements that extensions commonly touch
              var selectors = [
                '[data-gr-ext-installed]',
                '[data-new-gr-c-s-check-loaded]',
                '[data-lt-installed]',
                '[suppresshydrationwarning]',
                '[data-gr-installed]'
              ];

              selectors.forEach(function(sel) {
                try {
                  var nodes = document.querySelectorAll(sel);
                  for (var j = 0; j < nodes.length; j++) {
                    removeMatchingAttrs(nodes[j]);
                  }
                } catch (e) { /* ignore */ }
              });

              // As a final pass, remove attributes matching the regex from all elements
              // (limited to a reasonable number to avoid long runs)
              try {
                var all = document.getElementsByTagName('*');
                var max = Math.min(all.length, 2000); // cap to avoid hammering DOM
                for (var k = 0; k < max; k++) {
                  removeMatchingAttrs(all[k]);
                }
              } catch (e) { /* ignore */ }

            } catch (err) {
              // swallow errors to avoid blocking hydration
            }
          })();`}
        </Script>
      </head>

      <body className="font-sans antialiased">
        {children}
        <Analytics />
      </body>
    </html>
  )
}