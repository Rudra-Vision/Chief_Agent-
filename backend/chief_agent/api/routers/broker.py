"""Broker routes: Upstox connection, funds, positions, orders, reconciliation.

All broker calls happen SERVER-SIDE. Access tokens are never returned to the
browser; only sanitised status objects are.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from ...logging_setup import get_logger
from ...settings import OperatingMode
from ...timeutil import now_ist
from ..security import get_auth, sanitize_for_client
from ..state import AppState, get_app_state

log = get_logger(__name__, component="api.broker")
router = APIRouter()


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
@router.get("/broker/status", summary="Upstox connection status")
def broker_status(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    return sanitize_for_client(state.broker.status())


@router.get("/broker/upstox/login", summary="Start the Upstox OAuth login")
def upstox_login(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    try:
        info = state.broker.auth.build_login_url()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "url": info["url"],
        "state": info["state"],
        "instructions": (
            "Open this URL, log in to Upstox, and you will be redirected back here. "
            "The access token is stored server-side only and is never sent to the browser."
        ),
    }


@router.get("/broker/upstox/callback", summary="Upstox OAuth redirect target")
def upstox_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    app_state: AppState = Depends(get_app_state),
) -> Any:
    if error:
        return RedirectResponse(url=f"/?upstox_error={error}")
    if not code:
        return RedirectResponse(url="/?upstox_error=missing_code")
    try:
        record = app_state.broker.auth.exchange_code_for_token(code, state=state)
    except Exception as exc:
        log.error("token exchange failed", context={"error": str(exc)})
        return RedirectResponse(url=f"/?upstox_error={str(exc)[:120]}")
    verification = app_state.broker.auth.verify()
    if verification.get("ok"):
        app_state.provider.switch_to_live(app_state.broker)
    return RedirectResponse(url="/?upstox_connected=1")


class TokenBody(BaseModel):
    access_token: str


@router.post("/broker/upstox/token", summary="Attach an existing Upstox access token")
def attach_token(body: TokenBody, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    if not body.access_token.strip():
        raise HTTPException(status_code=400, detail="access_token is required")
    state.broker.auth.use_existing_token(body.access_token.strip(), source="manual")
    verification = state.broker.auth.verify()
    if verification.get("ok"):
        state.provider.switch_to_live(state.broker)
    return sanitize_for_client({"ok": bool(verification.get("ok")), "verification": verification})


@router.post("/broker/upstox/logout", summary="Log out of Upstox")
def upstox_logout(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    ok = state.broker.auth.logout()
    return {"ok": ok, "message": "logged out" if ok else "the broker logout call failed; the local token was cleared"}


@router.get("/broker/preflight", summary="LIVE-mode compliance checklist")
def preflight(deep: bool = Query(False), state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    report = state.preflight.run(
        deep=deep,
        components={
            "broker": state.broker,
            "risk_engine": state.risk_engine,
            "kill_switch": state.kill_switch,
            "strategy_version": state.strategy_version,
            "data_quality": {"safe_to_trade": not state.data_quality.data_safe_mode,
                             "summary": state.data_quality.last_report.summary if state.data_quality.last_report else ""},
            "reconciliation": state.reconciler.last_report.to_dict() if state.reconciler.last_report else None,
        },
    )
    return sanitize_for_client(report.to_dict())


# --------------------------------------------------------------------------- #
# Funds / positions / holdings
# --------------------------------------------------------------------------- #
@router.get("/broker/funds", summary="Funds and margins")
def funds(segment: str = "SEC", state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    if not state.broker.has_token:
        paper = state.paper.snapshot()["account"]
        return {
            "live": False,
            "note": "No Upstox token - showing the internal paper account instead.",
            "paper_account": paper,
            "funds": None,
        }
    try:
        snapshot = state.broker.portfolio.snapshot()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return sanitize_for_client({"live": True, **snapshot})


@router.get("/broker/positions", summary="Upstox positions")
def broker_positions(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    if not state.broker.has_token:
        return {"live": False, "positions": [], "note": "no Upstox token configured"}
    try:
        positions = state.broker.positions.positions()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "live": True,
        "positions": [
            {
                "instrument_key": p.instrument_key,
                "symbol": p.symbol,
                "quantity": p.quantity,
                "average_price": p.average_price,
                "last_price": p.last_price,
                "product": p.product,
                "direction": p.direction,
                "pnl": p.pnl,
            }
            for p in positions
        ],
    }


@router.get("/broker/holdings", summary="Upstox holdings")
def holdings(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    if not state.broker.has_token:
        return {"live": False, "holdings": []}
    try:
        rows = state.broker.positions.holdings()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "live": True,
        "holdings": [
            {
                "instrument_key": row.instrument_key,
                "symbol": row.symbol,
                "quantity": row.quantity,
                "average_price": row.average_price,
                "last_price": row.last_price,
                "pnl": row.pnl,
            }
            for row in rows
        ],
    }


# --------------------------------------------------------------------------- #
# Orders
# --------------------------------------------------------------------------- #
@router.get("/broker/orders", summary="Order book (broker) with internal orders")
def order_book(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    internal = [
        {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "transaction_type": order.transaction_type,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "order_type": order.order_type,
            "status": order.status,
            "average_fill_price": order.average_fill_price,
            "leg": order.leg,
            "fees": order.fees,
        }
        for order in state.paper.orders.values()
    ][-100:]
    broker_orders: List[Dict[str, Any]] = []
    error = None
    if state.broker.has_token:
        try:
            broker_orders = state.broker.orders.order_book()
        except Exception as exc:
            error = str(exc)
    return {
        "live": state.broker.has_token,
        "broker_orders": broker_orders,
        "internal_orders": internal,
        "error": error,
        "note": "In PAPER mode no broker order is ever sent; internal orders are simulated.",
    }


@router.get("/broker/trades", summary="Executed trades today")
def trade_book(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    if not state.broker.has_token:
        return {"live": False, "trades": [], "closed_trades": state.paper.closed_trades[-100:]}
    try:
        trades = state.broker.orders.trades_for_day()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"live": True, "trades": trades, "closed_trades": state.paper.closed_trades[-100:]}


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #
@router.get("/broker/reconciliation", summary="Position reconciliation status")
def reconciliation(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    return sanitize_for_client(state.reconciler.status())


@router.post("/broker/reconciliation/run", summary="Reconcile internal vs broker positions")
def run_reconciliation(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    internal = {
        position.instrument_key: position.quantity
        for position in state.paper.open_positions()
    }
    broker = {}
    if state.broker.has_token:
        try:
            broker = {key: p.quantity for key, p in state.broker.positions.net_positions().items()}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    else:
        # Without a broker connection the ledger trivially matches itself, which
        # is reported honestly rather than silently passing.
        broker = dict(internal)
    report = state.reconciler.reconcile(internal, broker, mode=state.settings.effective_mode().value)
    return report.to_dict()


@router.post("/broker/reconciliation/clear", summary="Clear a mismatch after manual review")
def clear_reconciliation(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    state.reconciler.clear()
    return {"ok": True, "message": "reconciliation mismatch cleared manually"}


# --------------------------------------------------------------------------- #
# Static IP
# --------------------------------------------------------------------------- #
class StaticIpBody(BaseModel):
    primary_ip: str
    secondary_ip: Optional[str] = None


@router.get("/broker/static-ip", summary="Registered static IP information")
def static_ip(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    info = state.broker.broker_risk.get_registered_ips()
    info["outbound_check"] = state.broker.broker_risk.verify_outbound_ip()
    return sanitize_for_client(info)


@router.put("/broker/static-ip", summary="Register a static IP with Upstox")
def update_static_ip(body: StaticIpBody, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    if not state.broker.has_token:
        raise HTTPException(status_code=400, detail="connect Upstox before updating the static IP")
    result = state.broker.broker_risk.update_static_ips(body.primary_ip, body.secondary_ip)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "static IP update failed"))
    return sanitize_for_client(result)


# --------------------------------------------------------------------------- #
# Kill switch
# --------------------------------------------------------------------------- #
class KillSwitchBody(BaseModel):
    reason: str = "manual stop from the dashboard"
    use_broker_switch: bool = False
    segments: Optional[List[str]] = None


@router.post("/broker/kill-switch/engage", summary="STOP TRADING")
def engage_kill_switch(body: KillSwitchBody, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    from ...risk.killswitch import KillSwitchTrigger

    if body.use_broker_switch:
        result = state.kill_switch.engage_with_broker(
            KillSwitchTrigger.MANUAL_BUTTON,
            state.broker.broker_risk,
            reason=body.reason,
            segments=body.segments,
        )
    else:
        result = {"local": state.kill_switch.engage(
            KillSwitchTrigger.MANUAL_BUTTON, reason=body.reason, actor="dashboard"
        ).to_dict(), "broker": None}

    state.notifications.kill_switch(True, body.reason)
    log.error("kill switch engaged from the dashboard", context={"reason": body.reason})
    return sanitize_for_client({"ok": True, **result, "state": state.kill_switch.to_dict()})


@router.post("/broker/kill-switch/release", summary="Resume trading")
def release_kill_switch(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    result = state.kill_switch.release(actor="dashboard", broker_risk=state.broker.broker_risk)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("reason", "could not release"))
    state.notifications.kill_switch(False, "released from the dashboard")
    return sanitize_for_client(result)


@router.get("/broker/kill-switch", summary="Kill switch state")
def kill_switch_state(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    return sanitize_for_client(state.kill_switch.to_dict())


__all__ = ["router"]
