"""Indian equity transaction cost model.

Backtesting without realistic costs is prohibited, so every simulated trade runs
through this module. All rates are CONFIGURABLE in ``config/execution.yaml``
(``costs:``) because charges, caps and STT rates change over time. The defaults
below follow the commonly published NSE equity-intraday schedule and are
documented inline so they can be re-verified against the current exchange and
broker schedules before any live deployment.

Charge components
-----------------
* brokerage                - percentage with a per-order cap (typical discount broker)
* STT / CTT                - securities transaction tax, side- and product-specific
* exchange transaction fee - NSE/BSE turnover charge
* SEBI turnover fee        - flat turnover levy
* GST                      - on (brokerage + exchange fee + SEBI fee)
* stamp duty               - on the buy side only
* DP charges               - delivery sell only
* slippage                 - model-based, volatility-scaled, applied per fill

The model is deliberately explicit: :meth:`CostModel.compute` returns a full
breakdown, which is persisted with every fill so a P&L number can always be
reconciled against its charges.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..logging_setup import get_logger
from ..settings import get_config_store

log = get_logger(__name__, component="costs")


class Product(str, enum.Enum):
    INTRADAY = "I"
    DELIVERY = "D"
    MTF = "MTF"


class InstrumentClass(str, enum.Enum):
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


@dataclass
class ChargeBreakdown:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange_txn: float = 0.0
    sebi_fee: float = 0.0
    gst: float = 0.0
    stamp_duty: float = 0.0
    dp_charges: float = 0.0
    slippage: float = 0.0
    other: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi_fee
            + self.gst
            + self.stamp_duty
            + self.dp_charges
            + self.slippage
            + self.other
        )

    @property
    def charges_only(self) -> float:
        """Everything except slippage (which is an execution cost, not a levy)."""
        return self.total - self.slippage

    def to_dict(self) -> Dict[str, float]:
        return {
            "brokerage": round(self.brokerage, 4),
            "stt": round(self.stt, 4),
            "exchange_txn": round(self.exchange_txn, 4),
            "sebi_fee": round(self.sebi_fee, 4),
            "gst": round(self.gst, 4),
            "stamp_duty": round(self.stamp_duty, 4),
            "dp_charges": round(self.dp_charges, 4),
            "slippage": round(self.slippage, 4),
            "other": round(self.other, 4),
            "charges_only": round(self.charges_only, 4),
            "total": round(self.total, 4),
        }

    def scale(self, factor: float) -> "ChargeBreakdown":
        """Return a copy with every component multiplied (used by cost stress tests)."""
        return ChargeBreakdown(
            brokerage=self.brokerage * factor,
            stt=self.stt * factor,
            exchange_txn=self.exchange_txn * factor,
            sebi_fee=self.sebi_fee * factor,
            gst=self.gst * factor,
            stamp_duty=self.stamp_duty * factor,
            dp_charges=self.dp_charges * factor,
            slippage=self.slippage * factor,
            other=self.other * factor,
        )


DEFAULT_COSTS: Dict[str, Any] = {
    "brokerage": {
        "model": "percentage_or_cap",
        "pct": 0.0003,
        "cap_per_order": 20.0,
        "delivery_pct": 0.0,
        "delivery_cap_per_order": 20.0,
    },
    "stt": {
        "intraday_sell_pct": 0.00025,
        "delivery_buy_and_sell_pct": 0.001,
        "futures_sell_pct": 0.000125,
        "options_sell_pct": 0.000625,
    },
    "exchange_transaction_charge": {
        "nse_equity_pct": 0.0000297,
        "nse_futures_pct": 0.0000173,
        "nse_options_pct": 0.0003503,
        "bse_equity_pct": 0.0000375,
    },
    "sebi_turnover_fee_pct": 0.000001,
    "gst_pct": 0.18,
    "stamp_duty": {
        "equity_intraday_buy_pct": 0.00003,
        "equity_delivery_buy_pct": 0.00015,
        "futures_buy_pct": 0.00002,
        "options_buy_pct": 0.00003,
    },
    "dp_charges": {"delivery_sell_per_scrip": 13.5},
    "slippage": {
        "default_bps": 4,
        "stop_exit_bps": 10,
        "volatility_scaled": True,
        "volatility_scaling_factor": 0.6,
    },
    "apply_costs_in_backtest": True,
}


class CostModel:
    """Computes the full charge breakdown for a single order/fill."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(DEFAULT_COSTS)
        supplied = config if config is not None else (get_config_store().load("execution").get("costs") or {})
        for key, value in supplied.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                merged = dict(cfg[key])
                merged.update(value)
                cfg[key] = merged
            else:
                cfg[key] = value
        self.cfg = cfg

    # ------------------------------------------------------------------ access
    def _get(self, *path: str, default: Any = 0.0) -> Any:
        node: Any = self.cfg
        for part in path:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node if node is not None else default

    # --------------------------------------------------------------- slippage
    def slippage_bps(
        self,
        *,
        is_stop_exit: bool = False,
        atr_pct: Optional[float] = None,
        base_bps: Optional[float] = None,
    ) -> float:
        """Slippage in basis points, optionally scaled by intraday volatility."""
        bps = float(
            base_bps
            if base_bps is not None
            else self._get("slippage", "stop_exit_bps" if is_stop_exit else "default_bps", default=4.0)
        )
        if self._get("slippage", "volatility_scaled", default=True) and atr_pct:
            factor = float(self._get("slippage", "volatility_scaling_factor", default=0.6))
            # 1% ATR is the neutral point; higher volatility widens the impact.
            multiplier = max(0.5, min(3.0, 1.0 + factor * (float(atr_pct) / 0.01 - 1.0)))
            bps *= multiplier
        return bps

    def slippage_amount(
        self,
        price: float,
        quantity: int,
        *,
        is_stop_exit: bool = False,
        atr_pct: Optional[float] = None,
        base_bps: Optional[float] = None,
        multiplier: float = 1.0,
    ) -> float:
        notional = abs(float(price) * int(quantity))
        bps = self.slippage_bps(is_stop_exit=is_stop_exit, atr_pct=atr_pct, base_bps=base_bps)
        return notional * bps / 10_000.0 * float(multiplier)

    # ----------------------------------------------------------------- compute
    def compute(
        self,
        *,
        price: float,
        quantity: int,
        side: str,                                  # BUY | SELL
        product: str = Product.INTRADAY.value,
        instrument_class: str = InstrumentClass.EQUITY.value,
        exchange: str = "NSE",
        is_stop_exit: bool = False,
        atr_pct: Optional[float] = None,
        slippage_bps_override: Optional[float] = None,
        slippage_multiplier: float = 1.0,
        fee_multiplier: float = 1.0,
        include_slippage: bool = True,
    ) -> ChargeBreakdown:
        """Full charge breakdown for one fill."""
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        quantity = int(quantity)
        if quantity <= 0:
            return ChargeBreakdown()
        price = float(price)
        notional = price * quantity
        intraday = product in (Product.INTRADAY.value, Product.MTF.value)

        breakdown = ChargeBreakdown()

        # --- brokerage -----------------------------------------------------
        if intraday:
            brokerage = notional * float(self._get("brokerage", "pct", default=0.0003))
            cap = self._get("brokerage", "cap_per_order", default=None)
            if cap:
                brokerage = min(brokerage, float(cap))
        else:
            brokerage = notional * float(self._get("brokerage", "delivery_pct", default=0.0))
            cap = self._get("brokerage", "delivery_cap_per_order", default=None)
            if cap:
                brokerage = min(brokerage, float(cap))
        breakdown.brokerage = brokerage

        # --- STT -----------------------------------------------------------
        if instrument_class == InstrumentClass.EQUITY.value:
            if intraday:
                if side == "SELL":
                    breakdown.stt = notional * float(self._get("stt", "intraday_sell_pct", default=0.00025))
            else:
                breakdown.stt = notional * float(self._get("stt", "delivery_buy_and_sell_pct", default=0.001))
        elif instrument_class == InstrumentClass.FUTURES.value:
            if side == "SELL":
                breakdown.stt = notional * float(self._get("stt", "futures_sell_pct", default=0.000125))
        else:
            if side == "SELL":
                breakdown.stt = notional * float(self._get("stt", "options_sell_pct", default=0.000625))

        # --- exchange transaction charge -----------------------------------
        if instrument_class == InstrumentClass.EQUITY.value:
            key = "nse_equity_pct" if exchange.upper() == "NSE" else "bse_equity_pct"
        elif instrument_class == InstrumentClass.FUTURES.value:
            key = "nse_futures_pct"
        else:
            key = "nse_options_pct"
        breakdown.exchange_txn = notional * float(self._get("exchange_transaction_charge", key, default=0.0000297))

        # --- SEBI turnover fee ---------------------------------------------
        breakdown.sebi_fee = notional * float(self._get("sebi_turnover_fee_pct", default=0.000001))

        # --- GST on brokerage + exchange fee + SEBI fee ---------------------
        gst_rate = float(self._get("gst_pct", default=0.18))
        breakdown.gst = (breakdown.brokerage + breakdown.exchange_txn + breakdown.sebi_fee) * gst_rate

        # --- stamp duty (buy side only) ------------------------------------
        if side == "BUY":
            if instrument_class == InstrumentClass.EQUITY.value:
                key = "equity_intraday_buy_pct" if intraday else "equity_delivery_buy_pct"
            elif instrument_class == InstrumentClass.FUTURES.value:
                key = "futures_buy_pct"
            else:
                key = "options_buy_pct"
            breakdown.stamp_duty = notional * float(self._get("stamp_duty", key, default=0.00003))

        # --- DP charges (delivery sell) -------------------------------------
        if not intraday and side == "SELL" and instrument_class == InstrumentClass.EQUITY.value:
            breakdown.dp_charges = float(self._get("dp_charges", "delivery_sell_per_scrip", default=13.5))

        # --- slippage --------------------------------------------------------
        if include_slippage:
            if slippage_bps_override is not None:
                breakdown.slippage = notional * float(slippage_bps_override) / 10_000.0 * slippage_multiplier
            else:
                breakdown.slippage = self.slippage_amount(
                    price,
                    quantity,
                    is_stop_exit=is_stop_exit,
                    atr_pct=atr_pct,
                    multiplier=slippage_multiplier,
                )

        if fee_multiplier != 1.0:
            # Stress-testing knob: scales every statutory/exchange charge.
            scaled = breakdown.scale(fee_multiplier)
            scaled.slippage = breakdown.slippage  # slippage handled separately
            breakdown = scaled

        return breakdown

    def round_trip(
        self,
        *,
        entry_price: float,
        exit_price: float,
        quantity: int,
        product: str = Product.INTRADAY.value,
        instrument_class: str = InstrumentClass.EQUITY.value,
        exchange: str = "NSE",
        atr_pct: Optional[float] = None,
        fee_multiplier: float = 1.0,
        slippage_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """Total cost of a completed round trip, with a combined breakdown."""
        entry = self.compute(
            price=entry_price,
            quantity=quantity,
            side="BUY",
            product=product,
            instrument_class=instrument_class,
            exchange=exchange,
            atr_pct=atr_pct,
            fee_multiplier=fee_multiplier,
            slippage_multiplier=slippage_multiplier,
        )
        exit_leg = self.compute(
            price=exit_price,
            quantity=quantity,
            side="SELL",
            product=product,
            instrument_class=instrument_class,
            exchange=exchange,
            atr_pct=atr_pct,
            fee_multiplier=fee_multiplier,
            slippage_multiplier=slippage_multiplier,
        )
        combined = ChargeBreakdown(
            brokerage=entry.brokerage + exit_leg.brokerage,
            stt=entry.stt + exit_leg.stt,
            exchange_txn=entry.exchange_txn + exit_leg.exchange_txn,
            sebi_fee=entry.sebi_fee + exit_leg.sebi_fee,
            gst=entry.gst + exit_leg.gst,
            stamp_duty=entry.stamp_duty + exit_leg.stamp_duty,
            dp_charges=entry.dp_charges + exit_leg.dp_charges,
            slippage=entry.slippage + exit_leg.slippage,
            other=entry.other + exit_leg.other,
        )
        return {
            "entry": entry.to_dict(),
            "exit": exit_leg.to_dict(),
            "combined": combined.to_dict(),
            "total_cost": combined.total,
        }

    def describe(self) -> Dict[str, Any]:
        return dict(self.cfg)


def cost_model_from_config(override: Optional[Dict[str, Any]] = None) -> CostModel:
    return CostModel(override)


__all__ = [
    "CostModel",
    "ChargeBreakdown",
    "Product",
    "InstrumentClass",
    "DEFAULT_COSTS",
    "cost_model_from_config",
]
