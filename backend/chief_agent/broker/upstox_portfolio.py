"""Portfolio: funds, margins, profile and capital/margin requirements.

Verified endpoints:
    GET  /v2/user/get-funds-and-margin       (?segment=SEC | COM)
    GET  /v2/user/profile
    POST /v2/portfolio/margin-required
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..logging_setup import get_logger
from ..settings import Settings, get_settings
from .upstox_client import Endpoints, UpstoxError, UpstoxHttpClient

log = get_logger(__name__, component="portfolio")


@dataclass
class Funds:
    """Normalised fund/margin snapshot."""

    available_cash: float = 0.0
    used_margin: float = 0.0
    total_equity: float = 0.0
    payin: float = 0.0
    spanning: float = 0.0
    collateral: float = 0.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0
    segment: str = "SEC"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def margin_utilised_pct(self) -> float:
        denom = self.total_equity or (self.available_cash + self.used_margin)
        if denom <= 0:
            return 0.0
        return self.used_margin / denom


def _num(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class UpstoxPortfolio:
    """Account-level reads: funds, profile and margin requirements."""

    def __init__(self, client: UpstoxHttpClient, settings: Optional[Settings] = None) -> None:
        self.client = client
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------- funds
    def funds(self, segment: str = "SEC") -> Funds:
        """``segment`` is ``SEC`` (equity) or ``COM`` (commodity)."""
        params = {"segment": segment} if segment else None
        try:
            response = self.client.get(Endpoints.FUNDS_AND_MARGIN, params=params, limit_class="standard")
        except UpstoxError as exc:
            log.warning("funds lookup failed", context={"segment": segment, "error": str(exc)})
            return Funds(segment=segment)

        data = response.data if isinstance(response.data, dict) else {}
        equity_raw = data.get("equity") if isinstance(data.get("equity"), dict) else data
        return self._parse_funds(equity_raw, segment)

    @staticmethod
    def _parse_funds(raw: Mapping[str, Any], segment: str) -> Funds:
        available = raw.get("available_margin")
        if isinstance(available, Mapping):
            available_cash = _num(available.get("cash") or available.get("adhoc_margin"))
        else:
            available_cash = _num(raw.get("available_margin") or raw.get("available_balance"))

        used = raw.get("used_margin")
        if isinstance(used, Mapping):
            used_margin = sum(_num(v) for v in used.values())
        else:
            used_margin = _num(used)

        return Funds(
            available_cash=available_cash,
            used_margin=used_margin,
            total_equity=_num(raw.get("net") or raw.get("total_equity")) or (available_cash + used_margin),
            payin=_num(raw.get("payin_amount")),
            spanning=_num(raw.get("span_margin")),
            collateral=_num(raw.get("collateral") or raw.get("collateral_amount")),
            unrealised_pnl=_num(raw.get("unrealised_profit") or raw.get("unrealised_mtm")),
            realised_pnl=_num(raw.get("realised_profit") or raw.get("realised_mtm")),
            segment=segment,
            raw=dict(raw),
        )

    # ----------------------------------------------------------------- profile
    def profile(self) -> Dict[str, Any]:
        try:
            response = self.client.get(Endpoints.PROFILE, limit_class="standard")
        except UpstoxError as exc:
            return {"ok": False, "error": str(exc)}
        data = response.data if isinstance(response.data, dict) else {}
        return {
            "ok": True,
            "user_id": data.get("user_id"),
            "user_name": data.get("user_name"),
            "email": data.get("email"),
            "broker": data.get("broker"),
            "exchanges": data.get("exchanges", []),
            "products": data.get("products", []),
            "order_types": data.get("order_types", []),
        }

    # ------------------------------------------------------- margin requirement
    def margin_required(self, orders: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Pre-trade margin check for one or more candidate orders."""
        payload = {"orders": [dict(order) for order in orders]}
        try:
            response = self.client.post(Endpoints.MARGIN_REQUIRED, json_body=payload, limit_class="standard")
        except UpstoxError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "data": response.data}

    # --------------------------------------------------------------- aggregate
    def snapshot(self) -> Dict[str, Any]:
        funds = self.funds("SEC")
        profile = self.profile()
        return {
            "funds": {
                "available_cash": round(funds.available_cash, 2),
                "used_margin": round(funds.used_margin, 2),
                "total_equity": round(funds.total_equity, 2),
                "collateral": round(funds.collateral, 2),
                "unrealised_pnl": round(funds.unrealised_pnl, 2),
                "realised_pnl": round(funds.realised_pnl, 2),
                "margin_utilised_pct": round(funds.margin_utilised_pct, 6),
            },
            "profile": profile,
        }


__all__ = ["UpstoxPortfolio", "Funds"]
