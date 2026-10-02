"""Orders: Place Order V3, Modify Order V3, Cancel Order V3, GTT V3, order book
and trade book.

Verified against the current official documentation:

    POST   /v3/order/place                  (api-hft host)  + sandbox enabled
    PUT    /v3/order/modify                                 + sandbox enabled
    DELETE /v3/order/cancel                                 + sandbox enabled
    POST   /v2/order/multi/place
    DELETE /v2/order/multi/cancel
    POST   /v2/order/positions/exit         exit all positions
    GET    /v2/order/retrieve-all           order book
    GET    /v2/order/history                order history (by order_id or tag)
    GET    /v2/order/details                order status
    GET    /v2/order/trades/get-trades-for-day
    POST   /v3/order/gtt/place              GTT (incl. trailing stop-loss gap)
    PUT    /v3/order/gtt/modify
    DELETE /v3/order/gtt/cancel
    GET    /v3/order/gtt/list

Order payload fields (V3): quantity, product (I/D/MTF), validity (DAY/IOC),
price, tag, instrument_token, order_type (MARKET/LIMIT/SL/SL-M),
transaction_type (BUY/SELL), disclosed_quantity, trigger_price, is_amo, slice,
market_protection.

Important operational notes honoured by this module:
  * ``is_amo`` is ignored by Upstox - AMO is inferred from the market session.
  * ``market_protection=0`` causes market orders from the API to be rejected, so
    we default to ``-1`` (automatic protection per guidelines).
  * The response returns ``data.order_ids`` (a LIST, because of auto-slicing).

FAIL-CLOSED: order placement is never retried automatically. A transport
failure raises an error with ``outcome_uncertain = True`` so the execution
engine reconciles against the order book before doing anything else.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..logging_setup import get_logger
from ..settings import OperatingMode, Settings, get_config_store, get_settings
from ..timeutil import IST, ensure_ist, ist_date, now_ist
from .upstox_client import ApiResponse, Endpoints, UpstoxError, UpstoxHttpClient

log = get_logger(__name__, component="upstox_orders")

VALID_ORDER_TYPES = {"MARKET", "LIMIT", "SL", "SL-M"}
VALID_PRODUCTS = {"I", "D", "MTF"}
VALID_VALIDITY = {"DAY", "IOC"}
VALID_TRANSACTION = {"BUY", "SELL"}

# Documented order statuses (appendix/order-status)
OPEN_STATUSES = {
    "open",
    "open pending",
    "trigger pending",
    "validation pending",
    "put order req received",
    "modify pending",
    "modify validation pending",
    "cancel pending",
    "amo req received",
    "after market order req received",
    "waiting for approval",
}
FILLED_STATUSES = {"complete", "partially filled"}
FAILED_STATUSES = {"rejected", "cancelled", "cancelled after market order", "not cancelled", "invalid"}


@dataclass
class OrderRequest:
    """A validated order request, ready for the wire."""

    instrument_token: str
    transaction_type: str
    quantity: int
    order_type: str = "LIMIT"
    product: str = "I"
    validity: str = "DAY"
    price: float = 0.0
    trigger_price: float = 0.0
    disclosed_quantity: int = 0
    tag: str = ""
    is_amo: bool = False
    slice_order: bool = False
    market_protection: int = -1

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.instrument_token:
            errors.append("instrument_token is required")
        if self.transaction_type not in VALID_TRANSACTION:
            errors.append(f"transaction_type must be one of {sorted(VALID_TRANSACTION)}")
        if self.order_type not in VALID_ORDER_TYPES:
            errors.append(f"order_type must be one of {sorted(VALID_ORDER_TYPES)}")
        if self.product not in VALID_PRODUCTS:
            errors.append(f"product must be one of {sorted(VALID_PRODUCTS)}")
        if self.validity not in VALID_VALIDITY:
            errors.append(f"validity must be one of {sorted(VALID_VALIDITY)}")
        if int(self.quantity) <= 0:
            errors.append("quantity must be a positive integer")
        if self.order_type in ("LIMIT", "SL") and float(self.price) <= 0:
            errors.append(f"{self.order_type} orders require a positive price")
        if self.order_type in ("SL", "SL-M") and float(self.trigger_price) <= 0:
            errors.append(f"{self.order_type} orders require a positive trigger_price")
        if self.market_protection == 0 and self.order_type in ("MARKET", "SL-M"):
            errors.append("market_protection=0 would be rejected by the exchange for MARKET/SL-M orders")
        if self.tag and len(self.tag) > 40:
            errors.append("tag must be at most 40 characters")
        return errors

    def to_payload(self) -> Dict[str, Any]:
        return {
            "quantity": int(self.quantity),
            "product": self.product,
            "validity": self.validity,
            "price": float(self.price),
            "tag": self.tag or None,
            "instrument_token": self.instrument_token,
            "order_type": self.order_type,
            "transaction_type": self.transaction_type,
            "disclosed_quantity": int(self.disclosed_quantity),
            "trigger_price": float(self.trigger_price),
            "is_amo": bool(self.is_amo),
            "slice": bool(self.slice_order),
            "market_protection": int(self.market_protection),
        }

    def sanitized(self) -> Dict[str, Any]:
        """Payload without None values - what we actually send and log."""
        return {k: v for k, v in self.to_payload().items() if v is not None and v != ""}


@dataclass
class OrderResult:
    """Outcome of a place/modify/cancel call."""

    ok: bool
    order_ids: List[str] = field(default_factory=list)
    latency_ms: Optional[float] = None
    message: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    preflight_blocked: bool = False

    @property
    def primary_order_id(self) -> Optional[str]:
        return self.order_ids[0] if self.order_ids else None


class UpstoxOrders:
    """Order placement and management.

    In PAPER mode the HTTP transport is never used for orders at all - the
    execution engine routes to the paper simulator instead. In SANDBOX mode the
    calls go to ``https://sandbox.upstox.com``. Only in LIVE mode do they reach
    the production order host.
    """

    SANDBOX_SUPPORTED = {"place", "modify", "cancel"}

    def __init__(self, client: UpstoxHttpClient, settings: Optional[Settings] = None) -> None:
        self.client = client
        self.settings = settings or get_settings()
        config = get_config_store().load("broker")
        self._orders_cfg: Dict[str, Any] = config.get("orders", {}) or {}
        self._sim_cfg: Dict[str, Any] = config.get("simulation", {}) or {}

    # ------------------------------------------------------------------ guards
    @property
    def mode(self) -> OperatingMode:
        return self.settings.effective_mode()

    def _guard_live(self) -> Optional[OrderResult]:
        """Return a blocking result if live trading is not permitted."""
        if self.settings.operating_mode is not OperatingMode.LIVE:
            return None
        reasons = self.settings.live_blocked_reasons
        if reasons:
            log.error("live order blocked by preflight gates", context={"reasons": reasons})
            return OrderResult(
                ok=False,
                message="live trading blocked: " + "; ".join(reasons),
                preflight_blocked=True,
            )
        return None

    # ------------------------------------------------------------ place / modify
    def place_order(self, request: OrderRequest, *, raise_on_error: bool = True) -> OrderResult:
        blocked = self._guard_live()
        if blocked is not None:
            if raise_on_error:
                raise UpstoxError(blocked.message)
            return blocked

        errors = request.validate()
        if errors:
            raise ValueError("invalid order request: " + "; ".join(errors))

        payload = request.sanitized()
        headers = self._algo_headers()
        log.info(
            "placing order",
            context={
                "mode": self.mode.value,
                "instrument": request.instrument_token,
                "side": request.transaction_type,
                "qty": request.quantity,
                "type": request.order_type,
                "price": request.price,
                "trigger": request.trigger_price,
                "tag": request.tag,
            },
        )
        try:
            response: ApiResponse = self.client.request(
                "POST",
                Endpoints.ORDER_PLACE_V3,
                json_body=payload,
                headers=headers,
                limit_class="order_placement",
                use_hft=True,
                idempotent=False,          # NEVER blind-retry an order
                max_retries=0,
            )
        except UpstoxError as exc:
            log.error(
                "order placement failed",
                context={"tag": request.tag, "error": str(exc), "uncertain": exc.outcome_uncertain},
            )
            if raise_on_error:
                raise
            return OrderResult(ok=False, message=str(exc), raw={"error": str(exc)})

        data = response.data if isinstance(response.data, dict) else {}
        order_ids = data.get("order_ids") or ([data["order_id"]] if data.get("order_id") else [])
        meta = response.payload.get("metadata") if isinstance(response.payload, dict) else None
        latency = meta.get("latency") if isinstance(meta, dict) else response.latency_ms
        result = OrderResult(
            ok=True,
            order_ids=[str(x) for x in order_ids],
            latency_ms=float(latency) if isinstance(latency, (int, float)) else response.latency_ms,
            message="order accepted",
            raw=response.payload if isinstance(response.payload, dict) else {},
        )
        log.info("order accepted", context={"order_ids": result.order_ids, "tag": request.tag})
        return result

    def modify_order(
        self,
        order_id: str,
        *,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[str] = None,
        validity: Optional[str] = None,
        disclosed_quantity: Optional[int] = None,
        raise_on_error: bool = True,
    ) -> OrderResult:
        blocked = self._guard_live()
        if blocked is not None:
            if raise_on_error:
                raise UpstoxError(blocked.message)
            return blocked

        payload: Dict[str, Any] = {"order_id": order_id}
        for key, value in (
            ("quantity", quantity),
            ("price", price),
            ("trigger_price", trigger_price),
            ("order_type", order_type),
            ("validity", validity),
            ("disclosed_quantity", disclosed_quantity),
        ):
            if value is not None:
                payload[key] = value
        if not order_id:
            raise ValueError("order_id is required")

        try:
            response = self.client.request(
                "PUT",
                Endpoints.ORDER_MODIFY_V3,
                json_body=payload,
                headers=self._algo_headers(),
                limit_class="order_placement",
                use_hft=True,
                idempotent=False,
                max_retries=0,
            )
        except UpstoxError as exc:
            log.error("order modify failed", context={"order_id": order_id, "error": str(exc)})
            if raise_on_error:
                raise
            return OrderResult(ok=False, message=str(exc))

        data = response.data if isinstance(response.data, dict) else {}
        order_ids = data.get("order_ids") or ([data["order_id"]] if data.get("order_id") else [order_id])
        return OrderResult(ok=True, order_ids=[str(x) for x in order_ids], latency_ms=response.latency_ms, message="modified")

    def cancel_order(self, order_id: str, *, raise_on_error: bool = True) -> OrderResult:
        blocked = self._guard_live()
        if blocked is not None:
            if raise_on_error:
                raise UpstoxError(blocked.message)
            return blocked
        if not order_id:
            raise ValueError("order_id is required")
        try:
            response = self.client.request(
                "DELETE",
                Endpoints.ORDER_CANCEL_V3,
                params={"order_id": order_id},
                headers=self._algo_headers(),
                limit_class="order_placement",
                use_hft=True,
                idempotent=False,
                max_retries=0,
            )
        except UpstoxError as exc:
            log.error("order cancel failed", context={"order_id": order_id, "error": str(exc)})
            if raise_on_error:
                raise
            return OrderResult(ok=False, message=str(exc))

        data = response.data if isinstance(response.data, dict) else {}
        order_ids = data.get("order_ids") or ([data["order_id"]] if data.get("order_id") else [order_id])
        return OrderResult(ok=True, order_ids=[str(x) for x in order_ids], message="cancelled")

    def cancel_multi(self, segment: Optional[str] = None, tag: Optional[str] = None) -> OrderResult:
        params: Dict[str, Any] = {}
        if segment:
            params["segment"] = segment
        if tag:
            params["tag"] = tag
        try:
            response = self.client.delete(
                Endpoints.ORDER_MULTI_CANCEL,
                params=params,
                headers=self._algo_headers(),
                limit_class="order_placement",
                use_hft=True,
                idempotent=False,
                max_retries=0,
            )
        except UpstoxError as exc:
            return OrderResult(ok=False, message=str(exc))
        return OrderResult(ok=True, message="multi-cancel submitted", raw=response.payload or {})

    def exit_all_positions(self, segment: Optional[str] = None, tag: Optional[str] = None) -> OrderResult:
        """Exit all open positions (documented as POST /v2/order/positions/exit)."""
        params: Dict[str, Any] = {}
        if segment:
            params["segment"] = segment
        if tag:
            params["tag"] = tag
        try:
            response = self.client.post(
                Endpoints.ORDER_EXIT_ALL,
                params=params,
                headers=self._algo_headers(),
                limit_class="order_placement",
                idempotent=False,
                max_retries=0,
            )
        except UpstoxError as exc:
            log.error("exit-all-positions failed", context={"error": str(exc)})
            return OrderResult(ok=False, message=str(exc))
        return OrderResult(ok=True, message="exit-all submitted", raw=response.payload or {})

    # ------------------------------------------------------------------- reads
    def order_book(self) -> List[Dict[str, Any]]:
        response = self.client.get(Endpoints.ORDER_BOOK, limit_class="standard")
        data = response.data
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    def order_history(self, order_id: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if order_id:
            params["order_id"] = order_id
        if tag:
            params["tag"] = tag
        if not params:
            raise ValueError("order_id or tag is required")
        response = self.client.get(Endpoints.ORDER_HISTORY, params=params, limit_class="standard")
        data = response.data
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    def order_status(self, order_id: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if order_id:
            params["order_id"] = order_id
        if tag:
            params["tag"] = tag
        response = self.client.get(Endpoints.ORDER_STATUS, params=params, limit_class="standard")
        data = response.data
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    def trades_for_day(self) -> List[Dict[str, Any]]:
        response = self.client.get(Endpoints.TRADES_FOR_DAY, limit_class="standard")
        data = response.data
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    def trades_by_order(self, order_id: str) -> List[Dict[str, Any]]:
        response = self.client.get(Endpoints.TRADES_BY_ORDER, params={"order_id": order_id}, limit_class="standard")
        data = response.data
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    # --------------------------------------------------------------------- GTT
    def place_gtt(
        self,
        *,
        instrument_token: str,
        transaction_type: str,
        quantity: int,
        rules: Sequence[Mapping[str, Any]],
        gtt_type: str = "SINGLE",
        product: str = "D",
        raise_on_error: bool = True,
    ) -> OrderResult:
        """Place a GTT order (entry / target / stop-loss, with optional trailing gap).

        Rule shape: ``{"strategy": "ENTRY"|"TARGET"|"STOPLOSS",
                       "trigger_type": "ABOVE"|"BELOW"|"IMMEDIATE",
                       "trigger_price": float,
                       "trailing_gap": float (STOPLOSS only)}``
        """
        blocked = self._guard_live()
        if blocked is not None:
            if raise_on_error:
                raise UpstoxError(blocked.message)
            return blocked

        payload = {
            "type": gtt_type,
            "quantity": int(quantity),
            "product": product,
            "instrument_token": instrument_token,
            "transaction_type": transaction_type,
            "rules": [dict(rule) for rule in rules],
        }
        try:
            response = self.client.request(
                "POST",
                Endpoints.GTT_PLACE,
                json_body=payload,
                headers=self._algo_headers(),
                limit_class="order_placement",
                idempotent=False,
                max_retries=0,
            )
        except UpstoxError as exc:
            log.error("GTT place failed", context={"error": str(exc)})
            if raise_on_error:
                raise
            return OrderResult(ok=False, message=str(exc))
        data = response.data if isinstance(response.data, dict) else {}
        gtt_id = data.get("gtt_order_id") or data.get("id")
        return OrderResult(ok=True, order_ids=[str(gtt_id)] if gtt_id else [], message="gtt placed", raw=data)

    def modify_gtt(self, gtt_order_id: str, rules: Sequence[Mapping[str, Any]], quantity: Optional[int] = None,
                   gtt_type: str = "SINGLE") -> OrderResult:
        payload: Dict[str, Any] = {"gtt_order_id": gtt_order_id, "type": gtt_type, "rules": [dict(r) for r in rules]}
        if quantity is not None:
            payload["quantity"] = int(quantity)
        try:
            response = self.client.put(Endpoints.GTT_MODIFY, json_body=payload, limit_class="order_placement",
                                       idempotent=False, max_retries=0)
        except UpstoxError as exc:
            return OrderResult(ok=False, message=str(exc))
        return OrderResult(ok=True, message="gtt modified", raw=response.payload or {})

    def cancel_gtt(self, gtt_order_id: str) -> OrderResult:
        try:
            response = self.client.delete(Endpoints.GTT_CANCEL, params={"gtt_order_id": gtt_order_id},
                                          limit_class="order_placement", idempotent=False, max_retries=0)
        except UpstoxError as exc:
            return OrderResult(ok=False, message=str(exc))
        return OrderResult(ok=True, message="gtt cancelled", raw=response.payload or {})

    def list_gtt(self) -> List[Dict[str, Any]]:
        response = self.client.get(Endpoints.GTT_LIST, limit_class="standard")
        data = response.data
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    # ------------------------------------------------------------------ helpers
    def _algo_headers(self) -> Dict[str, str]:
        """The ``X-Algo-Name`` header is optional and only used with an
        exchange-approved algo name configured on the app."""
        headers: Dict[str, str] = {}
        name = self.settings.upstox_algo_name
        if name:
            headers["X-Algo-Name"] = name
        return headers

    # ------------------------------------------------------------- interpretation
    @staticmethod
    def classify_status(status: str) -> str:
        normalised = (status or "").strip().lower()
        if normalised in FILLED_STATUSES:
            return "FILLED"
        if normalised in FAILED_STATUSES:
            return "FAILED"
        if normalised in OPEN_STATUSES:
            return "OPEN"
        return "UNKNOWN"

    @staticmethod
    def is_terminal(status: str) -> bool:
        return UpstoxOrders.classify_status(status) in ("FILLED", "FAILED")

    @staticmethod
    def to_fill_price(order_row: Mapping[str, Any]) -> Optional[float]:
        for key in ("average_price", "avg_price", "filled_price", "price"):
            value = order_row.get(key)
            try:
                if value is not None and float(value) > 0:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def order_timestamp(order_row: Mapping[str, Any]) -> Optional[dt.datetime]:
        for key in ("order_timestamp", "exchange_timestamp", "status_message_timestamp"):
            value = order_row.get(key)
            if value:
                try:
                    return ensure_ist(value)
                except Exception:
                    continue
        return None


__all__ = [
    "UpstoxOrders",
    "OrderRequest",
    "OrderResult",
    "VALID_ORDER_TYPES",
    "VALID_PRODUCTS",
    "VALID_VALIDITY",
    "OPEN_STATUSES",
    "FILLED_STATUSES",
    "FAILED_STATUSES",
]
