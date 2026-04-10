from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import public, admin, admin_auth
from app.api import admin_ui

app = FastAPI(title="Telegram Access Control (Async)")

# CORS (в dev можно "*", в проде — конкретные домены)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠ заменить в production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(public.router)
app.include_router(admin.router)
app.include_router(admin_auth.router)
app.include_router(admin_ui.router)