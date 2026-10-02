"""System routes: health, mode, status, auth, notifications, configuration."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from ...logging_setup import get_logger
from ...monitoring.notifications import EventKind
from ...settings import get_settings
from ...timeutil import now_ist
from ..security import get_auth, sanitize_for_client
from ..state import AppState, get_app_state

log = get_logger(__name__, component="api.system")
router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.get("/health", summary="System health")
def health(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    return state.health_report()


@router.get("/system/health", summary="System health (aliased)")
def system_health(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    return state.health_report()


@router.get("/system/mode", summary="Current operating mode")
def mode(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    return state.mode_summary()


@router.get("/system/status", summary="Full system status for the dashboard")
def status(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    settings = get_settings()
    try:
        portfolio = state.portfolio.update_from_paper(state.paper)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("portfolio update failed", context={"error": str(exc)})
        portfolio = state.portfolio.snapshot()

    broker_status = state.broker.status()
    return sanitize_for_client(
        {
            "app": {
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
                "server_time": now_ist().isoformat(),
            },
            "mode": state.mode_summary(),
            "kill_switch": state.kill_switch.to_dict(),
            "portfolio": portfolio,
            "broker": {
                "mode": broker_status.get("mode"),
                "authenticated": broker_status.get("auth", {}).get("authenticated"),
                "token_source": broker_status.get("auth", {}).get("token_source"),
                "token_valid_today": broker_status.get("auth", {}).get("token_probably_valid_today"),
                "instrument_master_loaded": broker_status.get("instrument_master", {}).get("loaded"),
                "instrument_count": broker_status.get("instrument_master", {}).get("count"),
                "static_ip_configured": broker_status.get("static_ip", {}).get("configured"),
            },
            "data": {
                "source": state.provider.data_source,
                "is_simulated": state.provider.is_simulated,
                "reason": state.provider.status().reason,
                "data_safe_mode": state.data_quality.data_safe_mode,
                "quality_summary": (
                    state.data_quality.last_report.summary if state.data_quality.last_report else "not evaluated yet"
                ),
            },
            "market": state.calendar.status(),
            "strategy": {
                "version": state.strategy_version,
                "family": (state.champion_config.get("champion") or {}).get("strategy_family"),
                "config_hash": state.strategy.config_hash if state.strategy else None,
            },
            "risk": state.risk_engine.limits_summary(),
            "reconciliation": state.reconciler.status(),
            "slippage": state.slippage.summary().to_dict(),
            "counters": state.counters.to_dict(),
            "notifications": state.notifications.status(),
            "preflight": (
                state.preflight.last_report().to_dict() if state.preflight.last_report() else None
            ),
        }
    )


@router.get("/system/notifications", summary="Notification history")
def notifications(
    limit: int = 50,
    severity: Optional[str] = None,
    state: AppState = Depends(get_app_state),
) -> Dict[str, Any]:
    return {
        "items": state.notifications.history(limit=max(1, min(500, limit)), severity=severity),
        "status": state.notifications.status(),
    }


@router.post("/system/notifications/test", summary="Send a test notification")
def test_notification(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    notification = state.notifications.notify(
        EventKind.SYSTEM_ERROR,
        "Test notification",
        "This is a test from the dashboard. Delivery channels are working.",
        severity="INFO",
    )
    return {"ok": True, "notification": notification.to_dict()}


@router.get("/system/config", summary="Effective configuration (secrets redacted)")
def configuration(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    return sanitize_for_client(
        {
            "settings": get_settings().redacted(),
            "domains": state.config.all(),
            "override_files": str(state.config.override_dir),
        }
    )


class ConfigPatch(BaseModel):
    domain: str
    patch: Dict[str, Any]
    reason: str = "dashboard edit"


@router.post("/system/config", summary="Apply a runtime configuration override")
def patch_config(body: ConfigPatch, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    if body.domain not in state.config.NAMES:
        raise HTTPException(status_code=400, detail=f"unknown config domain '{body.domain}'")
    if body.domain == "risk":
        # Risk limits may be TIGHTENED through the dashboard, never loosened
        # beyond the configured ceilings without editing the file deliberately.
        risky = {"per_trade", "portfolio", "daily_limits"}
        if any(key in body.patch for key in risky):
            raise HTTPException(
                status_code=403,
                detail=(
                    "risk limits cannot be relaxed through the API. Edit config/risk.yaml "
                    "deliberately, or tighten the limits only."
                ),
            )
    try:
        merged = state.config.set_override(body.domain, body.patch)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log.warning("configuration overridden", context={"domain": body.domain, "reason": body.reason})
    return {"ok": True, "domain": body.domain, "config": merged}


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
@router.get("/auth/status", summary="Dashboard authentication status")
def auth_status(request: Request) -> Dict[str, Any]:
    return get_auth().status(request)


@router.post("/auth/login", summary="Log in to the dashboard")
def login(body: LoginRequest, request: Request, response: Response) -> Dict[str, Any]:
    auth = get_auth()
    client = request.client.host if request.client else "unknown"
    ok, message = auth.verify_credentials(body.username, body.password, client_key=client)
    if not ok:
        raise HTTPException(status_code=401, detail=message)
    token, session = auth.create_session(body.username)
    auth.set_cookie(response, token)
    return {
        "ok": True,
        "user": body.username,
        "csrf_token": session.csrf_token,
        "message": "logged in",
    }


@router.post("/auth/logout", summary="Log out of the dashboard")
def logout(response: Response, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    get_auth().clear_cookie(response)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
@router.get("/system/scheduler", summary="Scheduled jobs")
def scheduler_status(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    if state.scheduler is None:
        return {"running": False, "jobs": [], "note": "the scheduler has not been started"}
    return state.scheduler.status()


@router.post("/system/scheduler/run", summary="Run a scheduled job now")
def run_job(
    job: str = Body(..., embed=True),
    state: AppState = Depends(get_app_state),
) -> Dict[str, Any]:
    if state.scheduler is None:
        raise HTTPException(status_code=400, detail="the scheduler is not running")
    return state.scheduler.run_now(job)


__all__ = ["router"]
