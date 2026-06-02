import os
from datetime import date, datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import HTTP_302_FOUND

from ..auth import authenticate_admin, create_access_token, decode_access_token
from ..crud import (
    classify_message_by_admin,
    create_or_update_user,
    extend_user_access,
    get_all_admin_actions,
    get_all_users,
    get_user_by_telegram_id,
    get_user_by_username,
    get_message_by_id,
    get_user_profile,
    get_user_usage_events,
    search_users_by_username,
    set_user_access,
    set_user_blocked,
    update_user_identity,
    verify_user_payment,
)
from ..db import get_db
from ..lightrag_jobs import get_lightrag_job_manager
from ..schemas import ClassifierIntent

router = APIRouter(prefix="/admin-ui", tags=["admin-ui"], include_in_schema=False)
templates = Jinja2Templates(directory="app/templates")
CLASSIFIER_INTENTS = [intent.value for intent in ClassifierIntent]
lightrag_jobs = get_lightrag_job_manager()


def cookie_secure_enabled() -> bool:
    raw = os.getenv("ADMIN_COOKIE_SECURE")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return os.getenv("APP_ENV", "development").strip().lower() in {"prod", "production"}


def render_template(request: Request, name: str, context: dict, status_code: int = 200):
    payload = {"request": request, **context}
    return templates.TemplateResponse(request, name, payload, status_code=status_code)


def normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def parse_optional_date(value: str | None) -> date | None:
    cleaned = normalize_optional_text(value)
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned)
    except ValueError:
        pass
    try:
        return datetime.strptime(cleaned, "%d.%m.%Y").date()
    except ValueError:
        return None


def format_filter_date(value: date | None) -> str:
    return value.strftime("%d.%m.%Y") if value else ""


def parse_optional_int(value: str | None) -> int | None:
    cleaned = normalize_optional_text(value)
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def sort_messages_from_conversations(conversations):
    messages = []
    for conversation in conversations:
        messages.extend(conversation.messages)
    return sorted(messages, key=lambda item: item.created_at or datetime.min, reverse=True)


async def get_current_admin_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    token = request.cookies.get("admin_token")
    if not token:
        return None

    try:
        payload = decode_access_token(token)
    except Exception:
        return None

    username = payload.get("sub")
    if not username:
        return None

    admin_user = await get_user_by_username(db, username)
    if not admin_user or not admin_user.is_admin:
        return None

    return admin_user


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return render_template(request, "login.html", {})


@router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    admin = await authenticate_admin(db, username, password)
    if not admin:
        return render_template(
            request,
            "login.html",
            {"error": "Invalid credentials"},
            status_code=401,
        )

    access_token = create_access_token({"sub": admin.username, "is_admin": True})
    response = RedirectResponse(url="/admin-ui/dashboard", status_code=HTTP_302_FOUND)
    response.set_cookie(
        key="admin_token",
        value=access_token,
        httponly=True,
        secure=cookie_secure_enabled(),
        samesite="lax",
    )
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/admin-ui/login", status_code=HTTP_302_FOUND)
    response.delete_cookie("admin_token")
    return response


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    q: str = "",
    access: str | None = None,
    blocked: str | None = None,
    page: int = 1,
    page_size: int = 50,
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    users = await search_users_by_username(db, q) if q else await get_all_users(db)

    filtered_users = []
    for user in users:
        if access == "granted" and not user.has_access:
            continue
        if access == "revoked" and user.has_access:
            continue
        if blocked == "blocked" and not user.is_blocked:
            continue
        if blocked == "active" and user.is_blocked:
            continue
        filtered_users.append(user)

    start = max((page - 1) * page_size, 0)
    end = start + page_size
    paginated_users = filtered_users[start:end]

    return render_template(
        request,
        "dashboard.html",
        {
            "users": paginated_users,
            "query": q,
            "access_filter": access,
            "blocked_filter": blocked,
            "admin_user": admin_user,
            "page": page,
            "page_size": page_size,
            "total_users": len(filtered_users),
        },
    )


@router.get("/users/{telegram_id}", response_class=HTMLResponse)
async def user_profile(
    request: Request,
    telegram_id: int,
    date_from: str | None = None,
    date_to: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    parsed_date_from = parse_optional_date(date_from)
    parsed_date_to = parse_optional_date(date_to)

    profile = await get_user_profile(db, telegram_id)
    if not profile:
        return HTMLResponse("User not found", status_code=404)

    conversations = sorted(
        profile.conversations,
        key=lambda item: item.updated_at or item.created_at or datetime.min,
        reverse=True,
    )
    messages = sort_messages_from_conversations(conversations)
    usage_events = sorted(
        profile.usage_events,
        key=lambda item: item.created_at or datetime.min,
        reverse=True,
    )
    filtered_usage_events = [
        event
        for event in usage_events
        if (not parsed_date_from or event.created_at.date() >= parsed_date_from)
        and (not parsed_date_to or event.created_at.date() <= parsed_date_to)
    ]

    return render_template(
        request,
        "user_profile.html",
        {
            "profile": profile,
            "conversations": conversations,
            "messages": messages,
            "usage_events": filtered_usage_events,
            "admin_actions_as_target": profile.admin_actions_as_target,
            "admin_actions_as_admin": profile.admin_actions_as_admin,
            "admin_user": admin_user,
            "classifier_intents": CLASSIFIER_INTENTS,
            "filters": {
                "date_from": format_filter_date(parsed_date_from),
                "date_to": format_filter_date(parsed_date_to),
            },
        },
    )


@router.post("/users")
async def create_user_ui(
    request: Request,
    telegram_id: int = Form(...),
    username: str = Form(""),
    first_name: str = Form(""),
    last_name: str = Form(""),
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    user = await create_or_update_user(
        db=db,
        telegram_id=telegram_id,
        username=normalize_optional_text(username),
        first_name=normalize_optional_text(first_name),
        last_name=normalize_optional_text(last_name),
        admin_user_id=admin_user.id,
    )
    return RedirectResponse(f"/admin-ui/users/{user.telegram_id}", status_code=HTTP_302_FOUND)


@router.post("/users/{telegram_id}/identity")
async def update_user_identity_ui(
    request: Request,
    telegram_id: int,
    username: str = Form(""),
    first_name: str = Form(""),
    last_name: str = Form(""),
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    user = await update_user_identity(
        db=db,
        telegram_id=telegram_id,
        username=normalize_optional_text(username),
        first_name=normalize_optional_text(first_name),
        last_name=normalize_optional_text(last_name),
        admin_user_id=admin_user.id,
    )
    if not user:
        return HTMLResponse("User not found", status_code=404)

    return RedirectResponse(f"/admin-ui/users/{telegram_id}", status_code=HTTP_302_FOUND)


@router.get("/lightrag", response_class=HTMLResponse)
async def lightrag_page(
    request: Request,
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    return render_template(
        request,
        "lightrag.html",
        {
            "admin_user": admin_user,
            "job": lightrag_jobs.snapshot(),
        },
    )


@router.post("/lightrag/start-full")
async def start_lightrag_full(
    request: Request,
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    lightrag_jobs.start("full")
    return RedirectResponse("/admin-ui/lightrag", status_code=HTTP_302_FOUND)


@router.post("/lightrag/continue")
async def continue_lightrag_build(
    request: Request,
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    lightrag_jobs.start("continue")
    return RedirectResponse("/admin-ui/lightrag", status_code=HTTP_302_FOUND)


@router.post("/lightrag/stop")
async def stop_lightrag_build(
    request: Request,
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    lightrag_jobs.stop()
    return RedirectResponse("/admin-ui/lightrag", status_code=HTTP_302_FOUND)


@router.post("/grant/{telegram_id}")
async def grant_user(
    request: Request,
    telegram_id: int,
    notes: str = Form(""),
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    user = await set_user_access(
        db,
        telegram_id,
        True,
        admin_user_id=admin_user.id,
        notes=notes or None,
    )
    if not user:
        return HTMLResponse("User not found", status_code=404)

    return RedirectResponse("/admin-ui/dashboard", status_code=HTTP_302_FOUND)


@router.post("/revoke/{telegram_id}")
async def revoke_user(
    request: Request,
    telegram_id: int,
    notes: str = Form(""),
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    user = await set_user_access(
        db,
        telegram_id,
        False,
        admin_user_id=admin_user.id,
        notes=notes or None,
    )
    if not user:
        return HTMLResponse("User not found", status_code=404)

    return RedirectResponse("/admin-ui/dashboard", status_code=HTTP_302_FOUND)


@router.post("/block/{telegram_id}")
async def block_user(
    request: Request,
    telegram_id: int,
    notes: str = Form(""),
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    user = await set_user_blocked(
        db,
        telegram_id,
        True,
        admin_user_id=admin_user.id,
        notes=notes or None,
    )
    if not user:
        return HTMLResponse("User not found", status_code=404)

    return RedirectResponse("/admin-ui/dashboard", status_code=HTTP_302_FOUND)


@router.post("/unblock/{telegram_id}")
async def unblock_user(
    request: Request,
    telegram_id: int,
    notes: str = Form(""),
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    user = await set_user_blocked(
        db,
        telegram_id,
        False,
        admin_user_id=admin_user.id,
        notes=notes or None,
    )
    if not user:
        return HTMLResponse("User not found", status_code=404)

    return RedirectResponse("/admin-ui/dashboard", status_code=HTTP_302_FOUND)


@router.post("/verify-payment/{telegram_id}")
async def verify_payment_user(
    request: Request,
    telegram_id: int,
    notes: str = Form(""),
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    user = await get_user_by_telegram_id(db, telegram_id)
    if not user:
        return HTMLResponse("User not found", status_code=404)

    await verify_user_payment(
        db,
        telegram_id,
        admin_user_id=admin_user.id,
        notes=notes or None,
    )
    return RedirectResponse(
        request.headers.get("referer") or f"/admin-ui/users/{telegram_id}",
        status_code=HTTP_302_FOUND,
    )


@router.post("/extend-access/{telegram_id}")
async def extend_access_user(
    request: Request,
    telegram_id: int,
    days: int = Form(30),
    notes: str = Form(""),
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    user = await extend_user_access(
        db,
        telegram_id,
        admin_user_id=admin_user.id,
        days=days,
        notes=notes or None,
    )
    if not user:
        return HTMLResponse("User not found", status_code=404)

    return RedirectResponse(
        request.headers.get("referer") or f"/admin-ui/users/{telegram_id}",
        status_code=HTTP_302_FOUND,
    )


@router.get("/reports/admin-actions", response_class=HTMLResponse)
async def admin_actions_report(
    request: Request,
    admin_user_id: str | None = None,
    target_user_id: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    parsed_admin_user_id = parse_optional_int(admin_user_id)
    parsed_target_user_id = parse_optional_int(target_user_id)
    parsed_entity_id = parse_optional_int(entity_id)
    parsed_date_from = parse_optional_date(date_from)
    parsed_date_to = parse_optional_date(date_to)
    parsed_entity_type = normalize_optional_text(entity_type)

    actions = await get_all_admin_actions(
        db=db,
        admin_user_id=parsed_admin_user_id,
        target_user_id=parsed_target_user_id,
        entity_type=parsed_entity_type,
        entity_id=parsed_entity_id,
        date_from=parsed_date_from,
        date_to=parsed_date_to,
    )

    return render_template(
        request,
        "reports_admin_actions.html",
        {
            "actions": actions,
            "filters": {
                "admin_user_id": parsed_admin_user_id,
                "target_user_id": parsed_target_user_id,
                "entity_type": parsed_entity_type,
                "entity_id": parsed_entity_id,
                "date_from": format_filter_date(parsed_date_from),
                "date_to": format_filter_date(parsed_date_to),
            },
            "admin_user": admin_user,
        },
    )


@router.get("/reports/usage-events", response_class=HTMLResponse)
async def usage_events_report(
    request: Request,
    user_id: str | None = None,
    conversation_id: str | None = None,
    message_id: str | None = None,
    event_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    parsed_user_id = parse_optional_int(user_id)
    parsed_conversation_id = parse_optional_int(conversation_id)
    parsed_message_id = parse_optional_int(message_id)
    parsed_event_type = normalize_optional_text(event_type)
    parsed_date_from = parse_optional_date(date_from)
    parsed_date_to = parse_optional_date(date_to)

    events = await get_user_usage_events(
        db=db,
        user_id=parsed_user_id,
        conversation_id=parsed_conversation_id,
        message_id=parsed_message_id,
        event_type=parsed_event_type,
        date_from=parsed_date_from,
        date_to=parsed_date_to,
    )

    total_input_tokens = sum(event.input_tokens for event in events)
    total_output_tokens = sum(event.output_tokens for event in events)
    total_tokens = total_input_tokens + total_output_tokens
    total_cost = sum((event.estimated_cost for event in events), Decimal("0"))

    return render_template(
        request,
        "reports_usage_events.html",
        {
            "events": events,
            "summary": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": total_cost,
            },
            "filters": {
                "user_id": parsed_user_id,
                "conversation_id": parsed_conversation_id,
                "message_id": parsed_message_id,
                "event_type": parsed_event_type,
                "date_from": format_filter_date(parsed_date_from),
                "date_to": format_filter_date(parsed_date_to),
            },
            "admin_user": admin_user,
        },
    )


@router.get("/users/{telegram_id}/messages", response_class=HTMLResponse)
async def user_messages(
    request: Request,
    telegram_id: int,
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    user = await get_user_profile(db, telegram_id)
    if not user:
        return HTMLResponse("User not found", status_code=404)

    messages = sort_messages_from_conversations(user.conversations)
    return render_template(
        request,
        "messages.html",
        {
            "user": user,
            "messages": messages,
            "admin_user": admin_user,
            "classifier_intents": CLASSIFIER_INTENTS,
        },
    )


@router.post("/messages/{message_id}/classify")
async def classify_message_ui(
    request: Request,
    message_id: int,
    intent: str = Form(...),
    explanation: str = Form(""),
    detected_language: str = Form(""),
    classifier_model: str = Form(""),
    classifier_version: str = Form(""),
    notes: str = Form(""),
    db: AsyncSession = Depends(get_db),
    admin_user=Depends(get_current_admin_user),
):
    if not admin_user:
        return RedirectResponse("/admin-ui/login", status_code=HTTP_302_FOUND)

    message = await get_message_by_id(db, message_id)
    if not message:
        return HTMLResponse("Message not found", status_code=404)

    await classify_message_by_admin(
        db=db,
        message_id=message_id,
        admin_user_id=admin_user.id,
        intent=intent,
        explanation=explanation or None,
        detected_language=detected_language or None,
        classifier_model=classifier_model or None,
        classifier_version=classifier_version or None,
        notes=notes or None,
    )

    return RedirectResponse(
        request.headers.get("referer") or f"/admin-ui/users/{message.conversation.user.telegram_id}/messages",
        status_code=HTTP_302_FOUND,
    )
