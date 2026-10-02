"""Signal explanation in SIMPLE and ADVANCED modes.

SIMPLE MODE is written for someone who does not read quantitative detail:

    "Reliance has broken above its morning range with strong volume, is above
     VWAP, and its sector is strong. Risk: Rs 250. Potential first target: Rs 520."

ADVANCED MODE exposes every input, weight and gate.

The explanation is built **deterministically** from the data that produced the
signal. An LLM may rewrite it into friendlier prose, but it is never allowed to
invent a number, and the deterministic version is always available as the
authoritative text. Under no circumstance does an explanation claim a probability
of profit or a guaranteed outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..logging_setup import get_logger
from ..timeutil import now_ist

log = get_logger(__name__, component="explainer")

#: Statements this layer must never make, regardless of what a model suggests.
BANNED_PHRASES = (
    "guaranteed",
    "guarantee",
    "risk-free",
    "risk free",
    "sure shot",
    "1% per day",
    "assured return",
    "assured profit",
    "will definitely",
    "cannot lose",
    "90% win",
    "100% accuracy",
)


def contains_banned_claim(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            return phrase
    return None


@dataclass
class Explanation:
    symbol: str
    direction: str
    simple: str
    advanced: Dict[str, Any] = field(default_factory=dict)
    bullets: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    generated_by: str = "deterministic"
    generated_at: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "simple": self.simple,
            "bullets": self.bullets,
            "risks": self.risks,
            "advanced": self.advanced,
            "generated_by": self.generated_by,
            "generated_at": (self.generated_at or now_ist()).isoformat(),
            "disclaimer": (
                "This is a description of a rule-based setup, not a prediction. "
                "No outcome is guaranteed."
            ),
        }


def explain_opportunity(opportunity: Mapping[str, Any], *, account_equity: Optional[float] = None) -> Explanation:
    """Build both explanation modes from a scanner opportunity."""
    symbol = str(opportunity.get("symbol", ""))
    direction = str(opportunity.get("direction", "LONG"))
    entry = float(opportunity.get("entry_price", 0.0) or 0.0)
    stop = float(opportunity.get("stop_price", 0.0) or 0.0)
    target_1 = float(opportunity.get("target_1", 0.0) or 0.0)
    risk_per_share = abs(entry - stop)
    quantity = int(opportunity.get("quantity_hint", 0) or 0)
    risk_amount = float(opportunity.get("risk_amount_hint", 0.0) or 0.0)
    if risk_amount <= 0 and quantity > 0:
        risk_amount = risk_per_share * quantity
    potential = abs(target_1 - entry) * quantity if quantity else abs(target_1 - entry)

    bullets = list(opportunity.get("reasons") or [])
    risks = list(opportunity.get("risks") or [])

    simple_lines: List[str] = []
    simple_lines.append(
        f"{symbol} looks like a {direction.lower()} setup."
    )
    if opportunity.get("plain_english"):
        simple_lines.append(str(opportunity["plain_english"]))
    simple_lines.append(
        f"Entry around Rs {entry:,.2f}, stop at Rs {stop:,.2f} "
        f"(that is Rs {risk_per_share:,.2f} of risk per share)."
    )
    if quantity:
        simple_lines.append(
            f"Suggested size {quantity} shares = about Rs {risk_amount:,.0f} at risk"
            + (f" ({risk_amount / account_equity:.2%} of your capital)." if account_equity else ".")
        )
    simple_lines.append(f"First target Rs {target_1:,.2f} (about Rs {potential:,.0f} if it gets there).")
    if risks:
        simple_lines.append("Things that could go wrong: " + "; ".join(risks[:3]) + ".")

    advanced = {
        "score": opportunity.get("score"),
        "score_components": opportunity.get("score_components") or opportunity.get("opportunity_score_components"),
        "entry": entry,
        "stop": stop,
        "target_1": target_1,
        "target_2": opportunity.get("target_2"),
        "risk_per_share": round(risk_per_share, 4),
        "risk_reward": opportunity.get("risk_reward"),
        "regime": opportunity.get("regime"),
        "sector": opportunity.get("sector"),
        "sector_rank": opportunity.get("sector_rank"),
        "vwap": opportunity.get("vwap"),
        "vwap_state": opportunity.get("vwap_state"),
        "relative_volume": opportunity.get("rvol"),
        "daily_atr_pct": opportunity.get("atr_daily_pct"),
        "spread_pct": opportunity.get("spread_pct"),
        "strategy_version": opportunity.get("strategy_version"),
        "sizing": {
            "quantity": quantity,
            "risk_amount": round(risk_amount, 2),
            "risk_pct_of_equity": round(risk_amount / account_equity, 6) if account_equity else None,
        },
        **{k: v for k, v in (opportunity.get("advanced") or {}).items()},
    }

    return Explanation(
        symbol=symbol,
        direction=direction,
        simple=" ".join(simple_lines),
        bullets=bullets,
        risks=risks,
        advanced=advanced,
    )


def explain_trade(trade: Mapping[str, Any]) -> Explanation:
    """Explain a completed trade - what the thesis was and what actually happened."""
    symbol = str(trade.get("symbol", ""))
    direction = str(trade.get("direction", "LONG"))
    net = float(trade.get("net_pnl", 0) or 0.0)
    r_multiple = float(trade.get("r_multiple", 0) or 0.0)
    reason = str(trade.get("exit_reason", "")).replace("_", " ").lower()
    outcome_word = "made money" if net > 0 else "lost money" if net < 0 else "finished flat"

    lines = [
        f"{symbol} was a {direction.lower()} trade entered at Rs {float(trade.get('entry_price', 0)):,.2f} "
        f"and closed at Rs {float(trade.get('exit_price', 0)):,.2f} ({reason}).",
        f"It {outcome_word}: Rs {net:,.2f} ({r_multiple:+.2f} times the risk taken).",
    ]
    if trade.get("regime_at_entry"):
        lines.append(f"The market regime at entry was {str(trade['regime_at_entry']).replace('_', ' ').lower()}.")
    if trade.get("sector"):
        lines.append(f"Sector: {trade['sector']}.")
    mfe = trade.get("mfe_r")
    mae = trade.get("mae_r")
    if mfe is not None and mae is not None:
        lines.append(
            f"At its best it was {float(mfe):+.2f}R and at its worst {float(mae):+.2f}R - "
            f"that shows how much of the move was available and how close the stop came."
        )
    if trade.get("lesson"):
        lines.append(str(trade["lesson"]))

    advanced = {
        key: trade.get(key)
        for key in (
            "entry_price",
            "exit_price",
            "stop_price",
            "target_price",
            "quantity",
            "gross_pnl",
            "fees",
            "slippage_cost",
            "net_pnl",
            "initial_risk",
            "r_multiple",
            "mae_r",
            "mfe_r",
            "holding_minutes",
            "exit_reason",
            "regime_at_entry",
            "sector",
            "strategy_version",
            "process_quality",
        )
    }
    return Explanation(symbol=symbol, direction=direction, simple=" ".join(lines), advanced=advanced)


def daily_summary(
    *,
    trading_date: Any,
    trades: Sequence[Mapping[str, Any]],
    equity_start: float,
    equity_end: float,
    regime: Optional[str],
    data_incidents: int = 0,
    risk_events: int = 0,
) -> Dict[str, Any]:
    """A plain-English daily review plus the numbers behind it."""
    net = equity_end - equity_start
    return_pct = (net / equity_start) if equity_start else 0.0
    wins = [t for t in trades if float(t.get("net_pnl", 0) or 0) > 0]
    losses = [t for t in trades if float(t.get("net_pnl", 0) or 0) < 0]
    best = max(trades, key=lambda t: float(t.get("net_pnl", 0) or 0), default=None)
    worst = min(trades, key=lambda t: float(t.get("net_pnl", 0) or 0), default=None)

    simple = [
        f"On {trading_date} the account moved from Rs {equity_start:,.0f} to Rs {equity_end:,.0f} "
        f"({return_pct:+.2%}).",
        f"There were {len(trades)} trades: {len(wins)} winners and {len(losses)} losers.",
    ]
    if best is not None and float(best.get("net_pnl", 0) or 0) > 0:
        simple.append(f"Best: {best.get('symbol')} +Rs {float(best.get('net_pnl', 0)):,.0f}.")
    if worst is not None and float(worst.get("net_pnl", 0) or 0) < 0:
        simple.append(f"Worst: {worst.get('symbol')} -Rs {abs(float(worst.get('net_pnl', 0))):,.0f}.")
    if regime:
        simple.append(f"The market was in a {regime.replace('_', ' ').lower()} state.")
    if data_incidents:
        simple.append(
            f"{data_incidents} market-data issue(s) were logged; trading pauses automatically when data is unreliable."
        )
    if not trades:
        simple.append("No trades were taken - the filters did not produce an acceptable setup.")

    return {
        "trading_date": str(trading_date),
        "simple": " ".join(simple),
        "net_pnl": round(net, 2),
        "return_pct": round(return_pct, 6),
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "best_trade": {"symbol": best.get("symbol"), "net_pnl": float(best.get("net_pnl", 0))} if best else None,
        "worst_trade": {"symbol": worst.get("symbol"), "net_pnl": float(worst.get("net_pnl", 0))} if worst else None,
        "regime": regime,
        "data_incidents": data_incidents,
        "risk_events": risk_events,
    }


__all__ = [
    "explain_opportunity",
    "explain_trade",
    "daily_summary",
    "Explanation",
    "contains_banned_claim",
    "BANNED_PHRASES",
]
