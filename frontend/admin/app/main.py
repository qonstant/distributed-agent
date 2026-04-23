import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import public, admin, admin_auth
from .api import admin_ui

app = FastAPI(title="Telegram Access Control (Async)")


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", os.getenv("ADMIN_CORS_ORIGINS", "*"))
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


# Routers
app.include_router(public.router)
app.include_router(admin.router)
app.include_router(admin_auth.router)
app.include_router(admin_ui.router)
