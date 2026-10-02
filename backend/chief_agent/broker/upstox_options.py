"""Option chain module.

Verified endpoints:
    GET /v2/option/contract?instrument_key=...&expiry_date=...
    GET /v2/option/chain?instrument_key=...&expiry_date=...
    GET /v2/option/greek?instrument_key=...

``expiry_date`` accepts either ``YYYY-MM-DD`` or a relative keyword:
    weekly  : current_week, next_week, far_week
    monthly : current_month, next_month, far_month
Keywords roll over automatically after each expiry, so research jobs can use
them without maintaining an expiry calendar.

Chain rows carry: expiry, pcr, strike_price, underlying_key,
underlying_spot_price, call_options{instrument_key, market_data, option_greeks},
put_options{...}.

This module is BUILT NOW but advanced options *trading* stays disabled in the
first release. It is used for context only: support/resistance from OI
concentration, PCR, and the volatility regime. No option order is ever placed by
this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..logging_setup import get_logger
from ..settings import Settings, get_settings
from .upstox_client import Endpoints, UpstoxError, UpstoxHttpClient

log = get_logger(__name__, component="options")

EXPIRY_KEYWORDS = {
    "current_week", "next_week", "far_week",
    "current_month", "next_month", "far_month",
}


@dataclass
class OptionLeg:
    instrument_key: str
    ltp: Optional[float] = None
    close_price: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    previous_open_interest: Optional[float] = None
    change_in_open_interest: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionChainRow:
    strike_price: float
    expiry: str = ""
    pcr: Optional[float] = None
    underlying_key: str = ""
    underlying_spot_price: Optional[float] = None
    call: Optional[OptionLeg] = None
    put: Optional[OptionLeg] = None


@dataclass
class OptionChainSummary:
    underlying_key: str
    expiry: str
    spot_price: Optional[float]
    rows: List[OptionChainRow]
    pcr: Optional[float] = None
    total_call_oi: float = 0.0
    total_put_oi: float = 0.0
    max_call_oi_strike: Optional[float] = None
    max_put_oi_strike: Optional[float] = None
    max_pain: Optional[float] = None
    atm_strike: Optional[float] = None
    atm_iv: Optional[float] = None
    change_in_call_oi: float = 0.0
    change_in_put_oi: float = 0.0

    def to_dict(self, limit: int = 25) -> Dict[str, Any]:
        return {
            "underlying_key": self.underlying_key,
            "expiry": self.expiry,
            "spot_price": self.spot_price,
            "pcr": self.pcr,
            "total_call_oi": self.total_call_oi,
            "total_put_oi": self.total_put_oi,
            "change_in_call_oi": self.change_in_call_oi,
            "change_in_put_oi": self.change_in_put_oi,
            "max_call_oi_strike": self.max_call_oi_strike,
            "max_put_oi_strike": self.max_put_oi_strike,
            "max_pain": self.max_pain,
            "atm_strike": self.atm_strike,
            "atm_iv": self.atm_iv,
            "strikes": len(self.rows),
            "sample": [_row_dict(row) for row in self.rows[:limit]],
        }


def _row_dict(row: OptionChainRow) -> Dict[str, Any]:
    def leg(option: Optional[OptionLeg]) -> Optional[Dict[str, Any]]:
        if option is None:
            return None
        return {
            "instrument_key": option.instrument_key,
            "ltp": option.ltp,
            "oi": option.open_interest,
            "oi_change": option.change_in_open_interest,
            "volume": option.volume,
            "iv": option.iv,
            "delta": option.delta,
        }

    return {
        "strike": row.strike_price,
        "expiry": row.expiry,
        "pcr": row.pcr,
        "call": leg(row.call),
        "put": leg(row.put),
    }


def _f(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class UpstoxOptions:
    """Option chain reads. Trading of options is NOT enabled by this module."""

    def __init__(self, client: UpstoxHttpClient, settings: Optional[Settings] = None) -> None:
        self.client = client
        self.settings = settings or get_settings()

    # ------------------------------------------------------------- contracts
    def contracts(self, underlying_instrument_key: str, expiry_date: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"instrument_key": underlying_instrument_key}
        if expiry_date:
            _validate_expiry(expiry_date)
            params["expiry_date"] = expiry_date
        try:
            response = self.client.get(Endpoints.OPTION_CONTRACT, params=params, limit_class="standard")
        except UpstoxError as exc:
            log.warning("option contracts fetch failed", context={"underlying": underlying_instrument_key, "error": str(exc)})
            return []
        data = response.data
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    # ----------------------------------------------------------------- chain
    def chain(self, underlying_instrument_key: str, expiry_date: str = "current_week") -> List[OptionChainRow]:
        _validate_expiry(expiry_date)
        try:
            response = self.client.get(
                Endpoints.OPTION_CHAIN,
                params={"instrument_key": underlying_instrument_key, "expiry_date": expiry_date},
                limit_class="standard",
            )
        except UpstoxError as exc:
            log.warning("option chain fetch failed", context={"underlying": underlying_instrument_key, "error": str(exc)})
            return []

        data = response.data
        rows: List[OptionChainRow] = []
        if not isinstance(data, list):
            return rows
        for row in data:
            if not isinstance(row, dict):
                continue
            rows.append(
                OptionChainRow(
                    strike_price=float(_f(row.get("strike_price")) or 0.0),
                    expiry=str(row.get("expiry") or ""),
                    pcr=_f(row.get("pcr")),
                    underlying_key=str(row.get("underlying_key") or underlying_instrument_key),
                    underlying_spot_price=_f(row.get("underlying_spot_price")),
                    call=self._leg(row.get("call_options")),
                    put=self._leg(row.get("put_options")),
                )
            )
        rows.sort(key=lambda r: r.strike_price)
        return rows

    @staticmethod
    def _leg(node: Any) -> Optional[OptionLeg]:
        if not isinstance(node, dict):
            return None
        market = node.get("market_data") if isinstance(node.get("market_data"), dict) else {}
        greeks = node.get("option_greeks") if isinstance(node.get("option_greeks"), dict) else {}
        depth = market.get("depth") if isinstance(market.get("depth"), dict) else {}
        bid = ask = None
        if isinstance(depth.get("buy"), list) and depth["buy"]:
            bid = _f(depth["buy"][0].get("price"))
        if isinstance(depth.get("sell"), list) and depth["sell"]:
            ask = _f(depth["sell"][0].get("price"))
        return OptionLeg(
            instrument_key=str(node.get("instrument_key") or ""),
            ltp=_f(market.get("ltp") or market.get("last_price")),
            close_price=_f(market.get("close_price")),
            volume=_f(market.get("volume")),
            open_interest=_f(market.get("oi") or market.get("open_interest")),
            previous_open_interest=_f(market.get("previous_oi") or market.get("prev_oi")),
            change_in_open_interest=_f(market.get("oi_day_change") or market.get("change_in_oi")),
            bid=bid,
            ask=ask,
            iv=_f(greeks.get("iv")),
            delta=_f(greeks.get("delta")),
            gamma=_f(greeks.get("gamma")),
            theta=_f(greeks.get("theta")),
            vega=_f(greeks.get("vega")),
            raw=node,
        )

    def greeks(self, instrument_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        if not instrument_keys:
            return {}
        try:
            response = self.client.get(
                Endpoints.OPTION_GREEK,
                params={"instrument_key": ",".join(instrument_keys)},
                limit_class="standard",
            )
        except UpstoxError as exc:
            log.warning("option greeks fetch failed", context={"error": str(exc)})
            return {}
        data = response.data
        return data if isinstance(data, dict) else {}

    # --------------------------------------------------------------- analytics
    def summarise(self, underlying_instrument_key: str, expiry_date: str = "current_week") -> OptionChainSummary:
        rows = self.chain(underlying_instrument_key, expiry_date)
        spot = next((r.underlying_spot_price for r in rows if r.underlying_spot_price), None)
        call_oi = sum(r.call.open_interest or 0 for r in rows if r.call)
        put_oi = sum(r.put.open_interest or 0 for r in rows if r.put)
        call_change = sum(r.call.change_in_open_interest or 0 for r in rows if r.call)
        put_change = sum(r.put.change_in_open_interest or 0 for r in rows if r.put)

        max_call = max(
            (r for r in rows if r.call and r.call.open_interest),
            key=lambda r: r.call.open_interest or 0,
            default=None,
        )
        max_put = max(
            (r for r in rows if r.put and r.put.open_interest),
            key=lambda r: r.put.open_interest or 0,
            default=None,
        )
        atm = min(rows, key=lambda r: abs(r.strike_price - spot)) if (rows and spot) else None
        atm_iv = None
        if atm:
            values = [leg.iv for leg in (atm.call, atm.put) if leg and leg.iv is not None]
            if values:
                atm_iv = sum(values) / len(values)

        return OptionChainSummary(
            underlying_key=underlying_instrument_key,
            expiry=str(rows[0].expiry) if rows else expiry_date,
            spot_price=spot,
            rows=rows,
            pcr=(put_oi / call_oi) if call_oi else None,
            total_call_oi=call_oi,
            total_put_oi=put_oi,
            change_in_call_oi=call_change,
            change_in_put_oi=put_change,
            max_call_oi_strike=max_call.strike_price if max_call else None,
            max_put_oi_strike=max_put.strike_price if max_put else None,
            max_pain=_max_pain(rows),
            atm_strike=atm.strike_price if atm else None,
            atm_iv=atm_iv,
        )


def _max_pain(rows: Iterable[OptionChainRow]) -> Optional[float]:
    """Strike at which total option-writer payout is minimised.

    Computed only from OI that the API actually returned - never estimated.
    """
    rows = [r for r in rows if r.call or r.put]
    if not rows:
        return None
    strikes = [r.strike_price for r in rows]
    best_strike = None
    best_pain = None
    for candidate in strikes:
        pain = 0.0
        for row in rows:
            if row.call and row.call.open_interest:
                pain += max(0.0, candidate - row.strike_price) * row.call.open_interest
            if row.put and row.put.open_interest:
                pain += max(0.0, row.strike_price - candidate) * row.put.open_interest
        if best_pain is None or pain < best_pain:
            best_pain = pain
            best_strike = candidate
    return best_strike


def _validate_expiry(expiry: str) -> None:
    if not expiry:
        raise ValueError("expiry_date is required (YYYY-MM-DD or a relative keyword)")
    if expiry in EXPIRY_KEYWORDS:
        return
    import datetime as dt

    try:
        dt.date.fromisoformat(expiry)
    except ValueError as exc:
        raise ValueError(
            f"expiry_date must be YYYY-MM-DD or one of {sorted(EXPIRY_KEYWORDS)}"
        ) from exc


__all__ = [
    "UpstoxOptions",
    "OptionChainRow",
    "OptionChainSummary",
    "OptionLeg",
    "EXPIRY_KEYWORDS",
]
