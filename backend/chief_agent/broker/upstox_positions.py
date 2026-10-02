"""Positions and holdings, plus reconciliation helpers.

Verified endpoints:
    GET /v2/portfolio/short-term-positions   (intraday / MIS positions)
    GET /v2/portfolio/long-term-holdings     (delivery holdings)
    PUT /v2/portfolio/convert-position       (I <-> D product conversion)

Reconciliation is the safety-critical part: any mismatch between our internal
position ledger and the broker's view must stop new entries. The system never
"auto-heals" a position it does not recognise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ..logging_setup import get_logger
from ..settings import Settings, get_settings
from ..timeutil import now_ist
from .upstox_client import Endpoints, UpstoxError, UpstoxHttpClient

log = get_logger(__name__, component="positions")


@dataclass
class BrokerPosition:
    """A normalised broker position."""

    instrument_key: str
    symbol: str
    quantity: int
    average_price: float
    last_price: Optional[float] = None
    product: str = "I"
    exchange: str = "NSE"
    pnl: Optional[float] = None
    unrealised: Optional[float] = None
    realised: Optional[float] = None
    multiplier: float = 1.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        if self.quantity > 0:
            return "LONG"
        if self.quantity < 0:
            return "SHORT"
        return "FLAT"

    @property
    def notional(self) -> float:
        price = self.last_price or self.average_price or 0.0
        return abs(self.quantity) * price * self.multiplier


def _to_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class UpstoxPositions:
    """Read positions/holdings and reconcile them against the internal ledger."""

    def __init__(self, client: UpstoxHttpClient, settings: Optional[Settings] = None) -> None:
        self.client = client
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ reads
    def positions(self) -> List[BrokerPosition]:
        response = self.client.get(Endpoints.POSITIONS, limit_class="standard")
        data = response.data
        if not isinstance(data, list):
            return []
        return [self._normalise(row) for row in data if isinstance(row, dict)]

    def holdings(self) -> List[BrokerPosition]:
        response = self.client.get(Endpoints.HOLDINGS, limit_class="standard")
        data = response.data
        if not isinstance(data, list):
            return []
        return [self._normalise(row, product="D") for row in data if isinstance(row, dict)]

    def net_positions(self) -> Dict[str, BrokerPosition]:
        """All non-flat positions keyed by instrument_key (positions win over holdings)."""
        out: Dict[str, BrokerPosition] = {}
        for position in self.holdings():
            if position.quantity:
                out[position.instrument_key] = position
        for position in self.positions():
            if position.quantity:
                out[position.instrument_key] = position
        return out

    def _normalise(self, row: Mapping[str, Any], product: Optional[str] = None) -> BrokerPosition:
        instrument_key = str(
            row.get("instrument_token") or row.get("instrument_key") or row.get("trading_symbol") or ""
        )
        quantity = _to_int(row.get("quantity"))
        # Upstox marks the side in `quantity`: long positions are positive.
        if quantity == 0:
            buy_qty = _to_int(row.get("buy_quantity") or row.get("day_buy_quantity"))
            sell_qty = _to_int(row.get("sell_quantity") or row.get("day_sell_quantity"))
            quantity = buy_qty - sell_qty
        return BrokerPosition(
            instrument_key=instrument_key,
            symbol=str(row.get("trading_symbol") or row.get("tradingsymbol") or instrument_key),
            quantity=quantity,
            average_price=float(_to_float(row.get("average_price") or row.get("buy_price")) or 0.0),
            last_price=_to_float(row.get("last_price") or row.get("ltp")),
            product=str(product or row.get("product") or "I"),
            exchange=str(row.get("exchange") or "NSE"),
            pnl=_to_float(row.get("pnl")),
            unrealised=_to_float(row.get("unrealised_profit") or row.get("unrealised")),
            realised=_to_float(row.get("realised_profit") or row.get("realised")),
            multiplier=float(_to_float(row.get("multiplier")) or 1.0),
            raw=dict(row),
        )

    # ---------------------------------------------------------- convert product
    def convert_position(
        self,
        instrument_key: str,
        from_product: str,
        to_product: str,
        quantity: int,
        transaction_type: str,
    ) -> bool:
        payload = {
            "instrument_token": instrument_key,
            "from_product": from_product,
            "to_product": to_product,
            "quantity": int(quantity),
            "transaction_type": transaction_type,
        }
        try:
            response = self.client.put(Endpoints.CONVERT_POSITION, json_body=payload, limit_class="standard")
            return response.ok
        except UpstoxError as exc:
            log.error("position conversion failed", context={"instrument": instrument_key, "error": str(exc)})
            return False

    # ------------------------------------------------------------ reconciliation
    def reconcile(
        self,
        internal_positions: Mapping[str, int],
        *,
        quantity_tolerance: int = 0,
        mode: str = "LIVE",
    ) -> "ReconciliationResult":
        """Compare the broker's net positions with our internal ledger.

        ``internal_positions`` maps ``instrument_key -> signed quantity``.
        """
        broker = self.net_positions()
        broker_map = {key: position.quantity for key, position in broker.items()}
        mismatches: List[Dict[str, Any]] = []

        for key, expected in internal_positions.items():
            actual = broker_map.get(key, 0)
            if abs(actual - int(expected)) > quantity_tolerance:
                mismatches.append(
                    {
                        "type": "QUANTITY_MISMATCH" if key in broker_map else "MISSING_AT_BROKER",
                        "instrument_key": key,
                        "internal": int(expected),
                        "broker": actual,
                        "delta": actual - int(expected),
                    }
                )

        for key, actual in broker_map.items():
            if key not in internal_positions and actual != 0:
                mismatches.append(
                    {
                        "type": "UNKNOWN_BROKER_POSITION",
                        "instrument_key": key,
                        "internal": 0,
                        "broker": actual,
                        "delta": actual,
                        "note": "Not present in the internal ledger. Never auto-healed - requires human review.",
                    }
                )

        result = ReconciliationResult(
            matched=not mismatches,
            internal=dict(internal_positions),
            broker=broker_map,
            mismatches=mismatches,
            mode=mode,
            checked_at=now_ist(),
        )
        if mismatches:
            log.error(
                "position reconciliation mismatch",
                context={"count": len(mismatches), "types": sorted({m['type'] for m in mismatches})},
            )
        return result


@dataclass
class ReconciliationResult:
    matched: bool
    internal: Dict[str, int]
    broker: Dict[str, int]
    mismatches: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = "LIVE"
    checked_at: Any = None

    @property
    def action(self) -> str:
        if self.matched:
            return "NONE"
        # Fail closed: stop opening new positions, keep managing what we know.
        return "HALT_NEW_ENTRIES"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "internal_count": len(self.internal),
            "broker_count": len(self.broker),
            "mismatches": self.mismatches,
            "action": self.action,
            "mode": self.mode,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
        }


__all__ = ["UpstoxPositions", "BrokerPosition", "ReconciliationResult"]
