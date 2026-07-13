import { NextRequest, NextResponse } from "next/server"

function getSearchApiBaseUrl() {
  return process.env.UNI_SEARCH_API_BASE_URL?.trim() || ""
}

export async function GET(request: NextRequest) {
  const baseUrl = getSearchApiBaseUrl()
  if (!baseUrl) {
    return NextResponse.json(
      { error: "uni_search_api_not_configured" },
      { status: 503 },
    )
  }

  let upstream: URL
  try {
    upstream = new URL("/api/universities/search", baseUrl)
  } catch {
    return NextResponse.json(
      { error: "uni_search_api_url_invalid" },
      { status: 500 },
    )
  }

  request.nextUrl.searchParams.forEach((value, key) => {
    upstream.searchParams.set(key, value)
  })

  try {
    const response = await fetch(upstream, {
      method: "GET",
      cache: "no-store",
      headers: {
        Accept: "application/json",
      },
    })

    const contentType = response.headers.get("content-type") || ""
    if (contentType.includes("application/json")) {
      const payload = await response.json()
      return NextResponse.json(payload, { status: response.status })
    }

    const text = await response.text()
    return NextResponse.json(
      { error: "upstream_non_json", detail: text },
      { status: response.status || 502 },
    )
  } catch {
    return NextResponse.json(
      { error: "uni_search_api_unreachable" },
      { status: 502 },
    )
  }
}
