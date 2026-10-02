"""The baseline strategy family:

    REGIME-FILTERED, SECTOR-CONFIRMED
    OPENING-RANGE BREAKOUT + OPTIONAL RETEST
    WITH VWAP AND VOLUME CONFIRMATION

Every gate is a named, auditable check that produces a human-readable reason.
The strategy is deliberately interpretable: no black boxes, no hidden weights.
Each blocked gate also produces a structured *rejection reason*, which is what
lets the research engine ask "how many trades were we prevented from taking, and
what would have happened?" - pure false-negative evidence.

Implementation notes
--------------------
* Only **closed** candles are evaluated: the breakout must be confirmed by the
  close of a completed candle, never by an intrabar tick. This removes an entire
  class of look-ahead bugs.
* Stop and target come from an explicit model (structural / ATR / swing / pct)
  and are always rounded to a valid tick.
* The strategy never sizes a position and never places anything.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..indicators import core as ind
from ..indicators.opening_range import OpeningRange, opening_range_state, retest_succeeded
from ..logging_setup import get_logger
from ..timeutil import IST
from .base import ImmutableConfig, StrategyContext, TradeCandidate, deep_get

log = get_logger(__name__, component="strategy.orb_vwap")


@dataclass
class GateResult:
    passed: bool
    name: str
    detail: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.passed


class GateEvaluator:
    """Collects named pass/fail gates and their reasons."""

    def __init__(self) -> None:
        self.gates: List[GateResult] = []
        self.reasons: List[str] = []
        self.risks: List[str] = []

    def check(self, name: str, condition: bool, pass_reason: str = "", fail_reason: str = "") -> bool:
        self.gates.append(GateResult(passed=bool(condition), name=name, detail=pass_reason if condition else fail_reason))
        if condition and pass_reason:
            self.reasons.append(pass_reason)
        if not condition and fail_reason:
            self.risks.append(fail_reason)
        return bool(condition)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def failed_gates(self) -> List[str]:
        return [g.name for g in self.gates if not g.passed]

    def first_failure(self) -> Optional[GateResult]:
        return next((g for g in self.gates if not g.passed), None)


class ORBVWAPStrategy:
    """Regime-filtered ORB + retest with VWAP and volume confirmation."""

    family = "orb_vwap_retest"

    def __init__(self, config: Dict[str, Any], version: str = "ORB_v1.0.0") -> None:
        self.config = ImmutableConfig(config)
        self.version = version
        self.config_hash = self.config.hash

    # ------------------------------------------------------------------ facade
    def evaluate(self, context: StrategyContext) -> Optional[TradeCandidate]:
        features = context.features
        if features.opening_range is None:
            return None
        if context.position_open:
            return None

        for direction in ("LONG", "SHORT"):
            if not self.config.get(f"{direction.lower()}.enabled", direction == "LONG"):
                continue
            candidate = self._evaluate_direction(direction, context)
            if candidate is not None:
                return candidate
        return None

    # ------------------------------------------------------------- directions
    def _evaluate_direction(self, direction: str, context: StrategyContext) -> Optional[TradeCandidate]:
        prefix = direction.lower()
        features = context.features
        or_range: OpeningRange = features.opening_range
        gates = GateEvaluator()
        # Filters and the breakout margin are measured against the SESSION-scaled
        # ATR ("daily ATR"), which is the horizon the thresholds are calibrated on.
        atr_value = features.atr_daily or features.atr or 0.0

        # 1. --- opening range must have formed and be sane -------------------
        if not gates.check(
            "opening_range_formed",
            or_range.complete and or_range.or_bar_count > 0,
            f"{or_range.duration_minutes}m opening range {or_range.or_low:.2f}-{or_range.or_high:.2f}",
            or_range.reason_incomplete or "opening range not complete",
        ):
            return self._rejected(context, direction, gates)

        min_or_frac = float(self.config.get("opening_range.min_or_width_atr_fraction", 0.0))
        max_or_frac = float(self.config.get("opening_range.max_or_width_atr_fraction", 99.0))
        if or_range.or_width_atr_fraction is not None and atr_value > 0:
            if not gates.check(
                "opening_range_width_min",
                or_range.or_width_atr_fraction >= min_or_frac,
                f"opening range is {or_range.or_width_atr_fraction:.2f}x ATR wide",
                f"opening range too narrow ({or_range.or_width_atr_fraction:.2f}x ATR, need >= {min_or_frac:.2f})",
            ):
                return self._rejected(context, direction, gates)
            if not gates.check(
                "opening_range_width_max",
                or_range.or_width_atr_fraction <= max_or_frac,
                f"opening range within {max_or_frac:.2f}x ATR",
                f"opening range too wide ({or_range.or_width_atr_fraction:.2f}x ATR, max {max_or_frac:.2f})",
            ):
                return self._rejected(context, direction, gates)

        # 2. --- entry window ---------------------------------------------------
        start_min = float(self.config.get(f"{prefix}.entry_window_start_minutes", 0))
        end_min = float(self.config.get(f"{prefix}.entry_window_end_minutes", 375))
        mins = features.minutes_from_open
        if not gates.check(
            "entry_window",
            start_min <= mins <= end_min,
            f"within the {start_min:.0f}-{end_min:.0f} minute entry window",
            f"outside the entry window ({mins:.0f} min from open; allowed {start_min:.0f}-{end_min:.0f})",
        ):
            return self._rejected(context, direction, gates)

        # 3. --- regime ---------------------------------------------------------
        allowed_regimes = self.config.get(f"{prefix}.allowed_regimes", None) or []
        if allowed_regimes:
            regime = features.regime
            if not gates.check(
                "regime_allowed",
                regime in allowed_regimes,
                f"market regime {regime} is tradable for this setup",
                f"market regime {regime} is not in the allowed set {allowed_regimes}",
            ):
                return self._rejected(context, direction, gates)

        # 4. --- volatility / gap sanity ---------------------------------------
        # The ATR filters are written against the SESSION-scaled ATR% ("daily
        # ATR"), not the per-bar figure, because that is the volatility horizon
        # the thresholds are calibrated on (see config/strategy.yaml).
        atr_for_filter = features.atr_daily_pct if features.atr_daily_pct is not None else features.atr_pct
        min_atr_pct = float(self.config.get(f"{prefix}.min_atr_pct", 0.0))
        max_atr_pct = float(self.config.get(f"{prefix}.max_atr_pct", 1.0))
        if atr_for_filter is not None:
            if not gates.check(
                "atr_adequate",
                atr_for_filter >= min_atr_pct,
                f"daily ATR {atr_for_filter:.2%} is adequate",
                f"daily ATR {atr_for_filter:.2%} is below the {min_atr_pct:.2%} floor (too quiet)",
            ):
                return self._rejected(context, direction, gates)
            if not gates.check(
                "atr_not_extreme",
                atr_for_filter <= max_atr_pct,
                f"daily ATR {atr_for_filter:.2%} is not extreme",
                f"daily ATR {atr_for_filter:.2%} exceeds the {max_atr_pct:.2%} ceiling (too wild)",
            ):
                return self._rejected(context, direction, gates)

        max_gap = float(self.config.get(f"{prefix}.max_gap_pct", 1.0))
        if features.gap_pct is not None:
            if not gates.check(
                "gap_acceptable",
                abs(features.gap_pct) <= max_gap,
                f"gap {features.gap_pct:+.2%} is manageable",
                f"opening gap {features.gap_pct:+.2%} exceeds the {max_gap:.2%} limit",
            ):
                return self._rejected(context, direction, gates)

        # 5. --- breakout -------------------------------------------------------
        margin_frac = float(self.config.get(f"{prefix}.breakout_margin_atr_fraction", 0.0))
        state = opening_range_state(
            or_range,
            context.recent_candles[-1] if context.recent_candles else _candle_from_features(features),
            direction=direction,
            margin_atr_fraction=margin_frac,
            atr_value=atr_value,
        )
        require_close = bool(self.config.get(f"{prefix}.require_close_above_or_high", True)) if direction == "LONG" else bool(
            self.config.get(f"{prefix}.require_close_below_or_low", True)
        )
        level = or_range.or_high if direction == "LONG" else or_range.or_low
        if not gates.check(
            "orb_breakout",
            state["broken"],
            f"broke the opening range {'high' if direction == 'LONG' else 'low'} at {level:.2f}",
            f"price has not broken {level:.2f} yet",
        ):
            return self._rejected(context, direction, gates)
        if require_close:
            if not gates.check(
                "breakout_close_confirmation",
                state["close_confirmed"],
                "candle closed beyond the breakout level",
                "breakout not confirmed by a candle close (possible false break)",
            ):
                return self._rejected(context, direction, gates)

        # 6. --- VWAP -----------------------------------------------------------
        if bool(self.config.get(f"{prefix}.require_above_vwap", True)) if direction == "LONG" else bool(
            self.config.get(f"{prefix}.require_below_vwap", True)
        ):
            min_vwap_distance = float(self.config.get(f"{prefix}.min_vwap_distance_pct", 0.0))
            vwap_ok = (
                features.above_vwap is True and (features.vwap_distance_pct or 0) >= min_vwap_distance
                if direction == "LONG"
                else features.above_vwap is False and abs(features.vwap_distance_pct or 0) >= min_vwap_distance
            )
            if features.vwap:
                vwap_detail = (
                    f"price {features.close:.2f} is {'above' if direction == 'LONG' else 'below'} VWAP {features.vwap:.2f}"
                    if vwap_ok
                    else f"price {features.close:.2f} is on the wrong side of VWAP {features.vwap:.2f}"
                )
            else:
                vwap_detail = "VWAP is not available for this bar"
            if not gates.check("vwap_side", bool(vwap_ok), vwap_detail, vwap_detail):
                return self._rejected(context, direction, gates)

        # 7. --- relative volume -------------------------------------------------
        min_rvol = float(self.config.get(f"{prefix}.min_rvol", 1.0))
        rvol_value = features.rvol
        rvol_pass = rvol_value is not None and rvol_value >= min_rvol
        if rvol_value is not None:
            rvol_detail = (
                f"relative volume {rvol_value:.2f}x confirms participation"
                if rvol_pass
                else f"relative volume {rvol_value:.2f}x is below the {min_rvol:.2f}x threshold"
            )
        else:
            rvol_detail = "relative volume is not available yet (needs prior sessions to compare against)"
        if not gates.check("relative_volume", rvol_pass, rvol_detail, rvol_detail):
            return self._rejected(context, direction, gates)

        # 8. --- relative strength vs the index ----------------------------------
        rs_floor = float(self.config.get(f"{prefix}.min_relative_strength_vs_index", -1.0)) if direction == "LONG" else -1.0
        rs_ceiling = float(self.config.get(f"{prefix}.max_relative_strength_vs_index", 1.0)) if direction == "SHORT" else 1.0
        if features.relative_strength is not None:
            if direction == "LONG":
                if not gates.check(
                    "relative_strength",
                    features.relative_strength >= rs_floor,
                    f"outperforming NIFTY by {features.relative_strength:+.2%}",
                    f"underperforming NIFTY by {features.relative_strength:+.2%}",
                ):
                    return self._rejected(context, direction, gates)
            else:
                if not gates.check(
                    "relative_strength",
                    features.relative_strength <= rs_ceiling,
                    f"underperforming NIFTY by {features.relative_strength:+.2%}",
                    f"outperforming NIFTY by {features.relative_strength:+.2%}",
                ):
                    return self._rejected(context, direction, gates)

        # 9. --- index VWAP filter (candidate research lever, off by default) ----
        if bool(self.config.get(f"{prefix}.require_index_above_vwap", False)) and direction == "LONG":
            index_above = context.extras.get("index_above_vwap")
            if index_above is not None and not gates.check(
                "index_vwap_filter",
                bool(index_above),
                "NIFTY is above its VWAP",
                "NIFTY is below its VWAP - long setups are unreliable",
            ):
                return self._rejected(context, direction, gates)

        # 10. --- sector ---------------------------------------------------------
        sector = context.sector
        snapshot = context.sector_snapshot
        min_rank = self.config.get(f"{prefix}.require_sector_rank_at_least", None)
        require_sector_vwap = bool(self.config.get(f"{prefix}.require_sector_above_vwap", False))
        if snapshot is not None and sector:
            state_sector = snapshot.state_of(sector)
            if state_sector is not None:
                if require_sector_vwap and direction == "LONG":
                    if not gates.check(
                        "sector_vwap",
                        state_sector.above_vwap is not False,
                        f"{sector} sector is above its VWAP",
                        f"{sector} sector index is below its VWAP",
                    ):
                        return self._rejected(context, direction, gates)
                if min_rank is not None and state_sector.rank is not None:
                    if not gates.check(
                        "sector_rank",
                        state_sector.rank <= int(min_rank),
                        f"{sector} is the #{state_sector.rank} strongest sector",
                        f"{sector} ranks #{state_sector.rank} of {len(snapshot.sectors)} (needs top {int(min_rank)})",
                    ):
                        return self._rejected(context, direction, gates)

        # 11. --- spread / liquidity --------------------------------------------
        max_spread = float(self.config.get(f"{prefix}.max_spread_pct", 1.0))
        if features.spread_pct is not None:
            if not gates.check(
                "spread_acceptable",
                features.spread_pct <= max_spread,
                f"spread {features.spread_pct:.3%} is tight",
                f"spread {features.spread_pct:.3%} exceeds {max_spread:.3%}",
            ):
                return self._rejected(context, direction, gates)

        # 12. --- optional retest ------------------------------------------------
        require_retest = bool(self.config.get(f"{prefix}.require_retest", False))
        if require_retest:
            retest_ok, retest_ts = retest_succeeded(
                [c for c in context.recent_candles if c.ts > or_range.formed_at] if or_range.formed_at else [],
                level,
                direction,
            )
            if not gates.check(
                "retest",
                retest_ok,
                f"level {level:.2f} was retested and held",
                f"no successful retest of {level:.2f} yet",
            ):
                return self._rejected(context, direction, gates)

        # 13. --- confirmation candle -------------------------------------------
        if bool(self.config.get(f"{prefix}.require_confirmation_candle", False)) and context.recent_candles:
            last = context.recent_candles[-1]
            body = ind.candle_body_pct(last.open, last.high, last.low, last.close)
            if direction == "LONG":
                confirmed = last.close >= last.open and body >= 0.35
                detail = f"breakout candle body {body:.0%} of range"
            else:
                confirmed = last.close <= last.open and body >= 0.35
                detail = f"breakdown candle body {body:.0%} of range"
            if not gates.check(
                "confirmation_candle",
                confirmed,
                detail,
                "breakout candle quality is poor (small or against the direction)",
            ):
                return self._rejected(context, direction, gates)

        # ------------------------------------------------------------------ exit plan
        entry_price = self._entry_price(features, direction)
        stop_price, stop_note = self._stop_price(features, or_range, direction, entry_price, context)
        target_1, target_2 = self._targets(entry_price, stop_price, direction)
        risk = abs(entry_price - stop_price)
        min_rr = float(self.config.get("targets.min_risk_reward_for_entry", 1.0))
        rr = abs(target_1 - entry_price) / risk if risk > 0 else 0.0
        if not gates.check(
            "risk_reward",
            rr >= min_rr,
            f"risk/reward 1:{rr:.2f} meets the 1:{min_rr:.2f} minimum",
            f"risk/reward 1:{rr:.2f} is below the 1:{min_rr:.2f} minimum",
        ):
            return self._rejected(context, direction, gates)

        # ------------------------------------------------------------------ score
        components = self._score_components(features, or_range, direction, rr, context)
        score = sum(components.values())

        candidate = TradeCandidate(
            instrument_key=features.instrument_key,
            symbol=context.extras.get("symbol") or features.instrument_key,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_1=target_1,
            target_2=target_2,
            strategy_family=self.family,
            strategy_version=self.version,
            reasons=list(gates.reasons),
            risks=list(gates.risks),
            features={**features.to_dict(), "gates_passed": [g.name for g in gates.gates]},
            score_components=components,
            score=score,
            regime=features.regime,
            sector=context.sector,
            sector_rank=(context.sector_snapshot.rank_of(context.sector) if context.sector_snapshot else None),
            ts=context.ts,
            tags=[stop_note],
        )
        return candidate

    # ------------------------------------------------------------------ helpers
    def _rejected(self, context: StrategyContext, direction: str, gates: GateEvaluator) -> None:
        """Record a rejection (evidence of trades NOT taken) and return None."""
        failure = gates.first_failure()
        if failure is not None:
            context.extras.setdefault("rejections", []).append(
                {
                    "direction": direction,
                    "gate": failure.name,
                    "detail": failure.detail,
                    "ts": context.ts.isoformat() if hasattr(context.ts, "isoformat") else str(context.ts),
                    "instrument_key": context.features.instrument_key,
                }
            )
        return None

    def _entry_price(self, features: Any, direction: str) -> float:
        """Entry is the close of the confirming candle - never a future price."""
        return float(features.close)

    def _stop_price(
        self,
        features: Any,
        or_range: OpeningRange,
        direction: str,
        entry_price: float,
        context: Optional[StrategyContext] = None,
    ) -> Tuple[float, str]:
        model = str(self.config.get("stops.model", "orb_structural"))
        atr_value = float(features.atr or 0.0)
        tick = 0.05

        if model == "orb_structural":
            buffer_fraction = float(self.config.get("stops.structural_buffer_atr", 0.25))
            buffer = atr_value * buffer_fraction
            raw = (or_range.or_low - buffer) if direction == "LONG" else (or_range.or_high + buffer)
            note = "structural stop beyond the opening range"
        elif model == "atr":
            multiplier = float(self.config.get("stops.atr_multiplier", 1.5))
            raw = entry_price - multiplier * atr_value if direction == "LONG" else entry_price + multiplier * atr_value
            note = f"ATR stop ({multiplier:.1f} x ATR)"
        elif model == "swing":
            lookback = int(self.config.get("stops.swing_lookback", 10))
            recent = list(context.recent_candles) if context and context.recent_candles else []
            if direction == "LONG":
                lows = [c.low for c in recent] or [features.low]
                swing = ind.swing_low(lows, len(lows) - 1, min(lookback, len(lows))) or features.low
                raw = swing - atr_value * 0.1
            else:
                highs = [c.high for c in recent] or [features.high]
                swing = ind.swing_high(highs, len(highs) - 1, min(lookback, len(highs))) or features.high
                raw = swing + atr_value * 0.1
            note = f"swing stop ({lookback} bars)"
        else:  # pct
            pct = float(self.config.get("stops.pct_stop", 0.012))
            raw = entry_price * (1 - pct) if direction == "LONG" else entry_price * (1 + pct)
            note = f"{pct:.2%} percentage stop"

        # Clamp the stop distance so risk is always meaningful and bounded.
        min_distance = float(self.config.get("stops.min_stop_distance_pct", 0.001)) * entry_price
        max_distance = float(self.config.get("stops.max_stop_distance_pct", 0.10)) * entry_price
        distance = abs(entry_price - raw)
        if distance < min_distance:
            raw = entry_price - min_distance if direction == "LONG" else entry_price + min_distance
            note += " (widened to the minimum distance)"
        elif distance > max_distance:
            raw = entry_price - max_distance if direction == "LONG" else entry_price + max_distance
            note += " (capped at the maximum distance)"

        return ind.round_to_tick(raw, tick, "down" if direction == "LONG" else "up"), note

    def _targets(self, entry_price: float, stop_price: float, direction: str) -> Tuple[float, float]:
        risk = abs(entry_price - stop_price)
        r1 = float(self.config.get("targets.r_multiple_t1", 1.5))
        r2 = float(self.config.get("targets.r_multiple_t2", 2.5))
        if direction == "LONG":
            return (
                ind.round_to_tick(entry_price + r1 * risk, 0.05, "down"),
                ind.round_to_tick(entry_price + r2 * risk, 0.05, "down"),
            )
        return (
            ind.round_to_tick(entry_price - r1 * risk, 0.05, "up"),
            ind.round_to_tick(entry_price - r2 * risk, 0.05, "up"),
        )

    # ------------------------------------------------------------------- score
    def _score_components(
        self,
        features: Any,
        or_range: OpeningRange,
        direction: str,
        risk_reward: float,
        context: StrategyContext,
    ) -> Dict[str, float]:
        """Deterministic, fully transparent opportunity score (0-100).

        Weights live in ``config/strategy.yaml -> opportunity_score.weights`` and
        can only be changed through the controlled research process.
        """
        weights = self.config.get("opportunity_score.weights", {}) or {}
        scale = float(self.config.get("opportunity_score.scale", 100))

        def clamp01(value: float) -> float:
            return max(0.0, min(1.0, value))

        # trend quality: EMA spread, ADX and range position
        ema_spread = features.ema_spread_pct or 0.0
        adx_value = features.adx or 0.0
        trend_quality = clamp01(
            0.45 * clamp01(abs(ema_spread) / 0.006)
            + 0.35 * clamp01((adx_value - 12.0) / 25.0)
            + 0.20 * clamp01((features.range_position - 0.5) * 2 if direction == "LONG" else (0.5 - features.range_position) * 2)
        )

        # ORB quality: the breakout distance relative to ATR, plus range sanity
        breakout_distance = (
            (features.close - or_range.or_high) if direction == "LONG" else (or_range.or_low - features.close)
        )
        atr_value = features.atr or 0.0
        distance_fraction = (breakout_distance / atr_value) if atr_value else 0.0
        orb_quality = clamp01(
            0.6 * clamp01(distance_fraction / 0.5)
            + 0.4 * (1.0 - clamp01(abs((or_range.or_width_atr_fraction or 0.0) - 0.8) / 1.5))
        )

        vwap_confirmation = clamp01(abs(features.vwap_distance_pct or 0.0) / 0.005)
        volume_confirmation = clamp01(((features.rvol or 0.0) - 1.0) / 1.0)

        sector_strength = 0.5
        if context.sector_snapshot is not None and context.sector:
            state = context.sector_snapshot.state_of(context.sector)
            if state is not None and state.rank and context.sector_snapshot.sectors:
                total = len(context.sector_snapshot.sectors)
                normalised = 1.0 - (state.rank - 1) / max(1, total - 1)
                sector_strength = normalised if direction == "LONG" else 1.0 - normalised

        market_regime = 0.5
        if features.regime:
            if direction == "LONG":
                market_regime = {"TREND_UP": 1.0, "GAP_UP": 0.85, "RANGE": 0.55, "LOW_VOLATILITY": 0.6}.get(
                    features.regime, 0.25
                )
            else:
                market_regime = {"TREND_DOWN": 1.0, "GAP_DOWN": 0.85, "RANGE": 0.55, "LOW_VOLATILITY": 0.6}.get(
                    features.regime, 0.25
                )

        relative_strength = 0.5
        if features.relative_strength is not None:
            relative_strength = clamp01(
                0.5 + (features.relative_strength / 0.02) * (1 if direction == "LONG" else -1)
            )

        liquidity = 0.5
        if features.average_daily_volume:
            liquidity = clamp01(features.average_daily_volume / 2_000_000)

        spread_quality = clamp01(1.0 - ((features.spread_pct or 0.001) / 0.004))
        rr_quality = clamp01((risk_reward - 1.0) / 2.0)

        raw = {
            "trend_quality": trend_quality,
            "orb_quality": orb_quality,
            "vwap_confirmation": vwap_confirmation,
            "volume_confirmation": volume_confirmation,
            "sector_strength": sector_strength,
            "market_regime": market_regime,
            "relative_strength": relative_strength,
            "liquidity": liquidity,
            "risk_reward": rr_quality,
            "spread_quality": spread_quality,
        }
        total_weight = sum(float(weights.get(k, 0.0)) for k in raw) or 1.0
        return {k: (v * float(weights.get(k, 0.0)) / total_weight) * scale for k, v in raw.items()}


def _candle_from_features(features: Any) -> Any:
    """Reconstruct a minimal candle-like object from features (used when the
    caller does not supply ``recent_candles``). Supplies only observed values."""
    from ..broker.upstox_market_data import Candle

    return Candle(
        ts=features.ts,
        open=features.open,
        high=features.high,
        low=features.low,
        close=features.close,
        volume=features.volume,
        instrument_key=features.instrument_key,
    )


__all__ = ["ORBVWAPStrategy", "GateEvaluator", "GateResult"]
