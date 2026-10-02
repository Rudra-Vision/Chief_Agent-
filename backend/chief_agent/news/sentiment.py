"""News classification.

Two responsibilities:

1. **Deterministic classification** (always on): a transparent keyword/lexicon
   classifier producing ``positive | negative | neutral | uncertain`` plus an
   event type (earnings, management, order win, regulatory, legal, M&A, macro,
   geopolitical, corporate action, analyst action, other).

2. **Optional LLM refinement**: if an LLM is configured it may *add* a label, but
   the deterministic label is always kept, disagreements are recorded, and an LLM
   label **never** on its own changes a trading decision. The brief is explicit:
   never blindly trade because an LLM called an article positive.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..logging_setup import get_logger
from ..settings import get_settings
from ..timeutil import now_ist

log = get_logger(__name__, component="news")

SENTIMENTS = ("positive", "negative", "neutral", "uncertain")

EVENT_TYPES = (
    "earnings",
    "management",
    "order_win",
    "regulatory",
    "legal",
    "ma",
    "macro",
    "geopolitical",
    "corporate_action",
    "analyst_action",
    "other",
)

#: Deliberately small, auditable lexicons. Keywords are matched case-insensitively
#: on word boundaries so "order" does not match "border".
POSITIVE_TERMS: Dict[str, float] = {
    "beats estimates": 1.0, "beat estimates": 1.0, "record profit": 1.0, "record revenue": 0.9,
    "profit rises": 0.9, "profit jumps": 1.0, "profit surges": 1.0, "revenue rises": 0.8,
    "margin expansion": 0.8, "order win": 0.9, "wins order": 0.9, "bags order": 0.9,
    "contract win": 0.8, "upgrade": 0.7, "raised target": 0.7, "raises guidance": 0.9,
    "strong growth": 0.8, "expansion plan": 0.5, "dividend": 0.5, "bonus issue": 0.5,
    "buyback": 0.7, "approval": 0.5, "partnership": 0.4, "acquisition": 0.3,
    "all-time high": 0.8, "outperform": 0.7, "positive outlook": 0.7, "debt reduction": 0.6,
    "capacity expansion": 0.5, "new plant": 0.4, "fda approval": 0.9, "order book strong": 0.8,
}

NEGATIVE_TERMS: Dict[str, float] = {
    "misses estimates": 1.0, "miss estimates": 1.0, "profit falls": 0.9, "profit drops": 0.9,
    "profit plunges": 1.0, "revenue falls": 0.8, "loss widens": 1.0, "net loss": 0.9,
    "margin pressure": 0.8, "downgrade": 0.7, "cut target": 0.7, "lowers guidance": 0.9,
    "guidance cut": 0.9, "regulatory action": 0.8, "penalty": 0.8, "fine imposed": 0.8,
    "fraud": 1.0, "investigation": 0.9, "probe": 0.8, "raid": 1.0, "lawsuit": 0.8,
    "litigation": 0.7, "default": 1.0, "insolvency": 1.0, "bankruptcy": 1.0,
    "resignation": 0.6, "ceo resigns": 0.9, "cfo resigns": 0.9, "layoffs": 0.6,
    "recall": 0.7, "strike": 0.5, "shutdown": 0.6, "plant closure": 0.7,
    "order cancellation": 0.8, "contract terminated": 0.8, "sebi": 0.4, "income tax raid": 1.0,
    "bribery": 1.0, "scam": 1.0, "negative outlook": 0.7, "downturn": 0.6,
    "growth slows": 0.6, "underperform": 0.7, "delay": 0.4, "shortfall": 0.7,
}

UNCERTAIN_TERMS = (
    "reportedly", "may", "might", "could", "expected to", "likely to",
    "sources say", "speculation", "rumour", "rumor", "unnamed",
)

EVENT_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("earnings", ("q1", "q2", "q3", "q4", "quarterly", "earnings", "net profit", "revenue", "ebitda", "results")),
    ("management", ("ceo", "cfo", "managing director", "board", "resign", "appointed", "chairman", "md ") ),
    ("order_win", ("order win", "wins order", "bags order", "contract", "tender", "order book", "purchase order")),
    ("regulatory", ("sebi", "rbi", "regulator", "approval", "compliance", "circular", "penalty", "nclt")),
    ("legal", ("court", "lawsuit", "litigation", "tribunal", "case filed", "verdict", "arbitration")),
    ("ma", ("acquisition", "merger", "amalgamation", "stake sale", "demerger", "buyout", "takeover", "stake buy")),
    ("macro", ("inflation", "gdp", "interest rate", "repo rate", "crude", "rupee", "fed", "fiscal", "budget")),
    ("geopolitical", ("tariff", "sanction", "war", "conflict", "border", "geopolit", "import duty", "trade war")),
    ("corporate_action", ("dividend", "bonus", "split", "rights issue", "buyback", "record date", "ex-date")),
    ("analyst_action", ("upgrade", "downgrade", "target price", "brokerage", "rating", "initiates coverage")),
)


@dataclass
class Classification:
    sentiment: str
    sentiment_score: float
    event_type: str
    matched_terms: List[str] = field(default_factory=list)
    uncertainty: float = 0.0
    classifier: str = "lexicon"
    llm_sentiment: Optional[str] = None
    agreement: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    @property
    def is_adverse(self) -> bool:
        return self.sentiment == "negative" and self.sentiment_score <= -0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentiment": self.sentiment,
            "sentiment_score": round(self.sentiment_score, 4),
            "event_type": self.event_type,
            "matched_terms": self.matched_terms[:20],
            "uncertainty": round(self.uncertainty, 3),
            "classifier": self.classifier,
            "llm_sentiment": self.llm_sentiment,
            "llm_agrees": self.agreement,
            "notes": self.notes,
            "adverse_for_long": self.is_adverse,
        }


def _matches(text: str, term: str) -> bool:
    # Word-boundary match so "order" does not fire inside "border".
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


class NewsClassifier:
    """Deterministic news classifier with an optional LLM second opinion."""

    def __init__(self, llm: Optional[Any] = None, use_llm: bool = False) -> None:
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    def classify(self, heading: str, summary: str = "") -> Classification:
        text = f"{heading or ''} {summary or ''}".lower()

        positive_hits: List[Tuple[str, float]] = []
        negative_hits: List[Tuple[str, float]] = []
        for term, weight in POSITIVE_TERMS.items():
            if _matches(text, term):
                positive_hits.append((term, weight))
        for term, weight in NEGATIVE_TERMS.items():
            if _matches(text, term):
                negative_hits.append((term, weight))

        positive_score = sum(w for _, w in positive_hits)
        negative_score = sum(w for _, w in negative_hits)
        total = positive_score + negative_score

        uncertainty_hits = [term for term in UNCERTAIN_TERMS if _matches(text, term)]
        uncertainty = min(1.0, len(uncertainty_hits) * 0.25)

        if total == 0:
            sentiment = "uncertain" if uncertainty > 0.4 else "neutral"
            score = 0.0
        else:
            score = (positive_score - negative_score) / max(total, 1e-9)
            score *= max(0.3, 1.0 - uncertainty * 0.5)
            if uncertainty >= 0.5 and abs(score) < 0.5:
                sentiment = "uncertain"
            elif score > 0.15:
                sentiment = "positive"
            elif score < -0.15:
                sentiment = "negative"
            else:
                sentiment = "neutral"

        event_type = self._event_type(text)
        classification = Classification(
            sentiment=sentiment,
            sentiment_score=score,
            event_type=event_type,
            matched_terms=[f"+{t}" for t, _ in positive_hits] + [f"-{t}" for t, _ in negative_hits],
            uncertainty=uncertainty,
            notes=[f"uncertainty markers: {', '.join(uncertainty_hits)}"] if uncertainty_hits else [],
        )

        if self.use_llm:
            self._apply_llm(classification, heading, summary)
        return classification

    @staticmethod
    def _event_type(text: str) -> str:
        for event_type, patterns in EVENT_PATTERNS:
            for pattern in patterns:
                if _matches(text, pattern):
                    return event_type
        return "other"

    def _apply_llm(self, classification: Classification, heading: str, summary: str) -> None:
        """Ask the LLM for a second opinion. It can never override the lexicon.

        Disagreements are recorded so they can be reviewed later; the
        deterministic label remains the one stored and acted upon.
        """
        try:
            prompt = (
                "Classify the sentiment of this Indian market news item as exactly one of "
                "positive, negative, neutral or uncertain. Answer with the single word only.\n\n"
                f"Headline: {heading}\nSummary: {summary[:600]}"
            )
            raw = str(self.llm.complete(prompt)).strip().lower()
            for candidate in SENTIMENTS:
                if candidate in raw:
                    classification.llm_sentiment = candidate
                    classification.agreement = candidate == classification.sentiment
                    if not classification.agreement:
                        classification.notes.append(
                            f"the LLM disagreed ({candidate}); the deterministic label is authoritative"
                        )
                    return
        except Exception as exc:
            classification.notes.append(f"LLM classification unavailable: {exc}")

    # ------------------------------------------------------------- aggregation
    def market_context(self, articles: Sequence[Mapping[str, Any]], limit: int = 10) -> Dict[str, Any]:
        """Summarise a batch of classified articles into a market-context block."""
        if not articles:
            return {"articles": 0, "sentiment": "neutral", "score": 0.0, "event_mix": {}, "adverse": []}

        scores: List[float] = []
        event_mix: Dict[str, int] = {}
        adverse: List[Dict[str, Any]] = []
        for article in articles:
            classification = article.get("classification") or {}
            if not classification:
                classification = self.classify(
                    str(article.get("heading", "")), str(article.get("summary", ""))
                ).to_dict()
            scores.append(float(classification.get("sentiment_score", 0.0)))
            event = str(classification.get("event_type", "other"))
            event_mix[event] = event_mix.get(event, 0) + 1
            if classification.get("adverse_for_long"):
                adverse.append(
                    {
                        "heading": str(article.get("heading", ""))[:200],
                        "instrument_key": article.get("instrument_key"),
                        "sentiment_score": classification.get("sentiment_score"),
                        "event_type": event,
                    }
                )

        average = sum(scores) / len(scores) if scores else 0.0
        return {
            "articles": len(articles),
            "sentiment": "positive" if average > 0.15 else "negative" if average < -0.15 else "neutral",
            "score": round(average, 4),
            "event_mix": event_mix,
            "adverse": adverse[:limit],
            "note": (
                "News context NEVER overrides the risk engine and is never a standalone reason "
                "to enter a trade. It can only add context, or veto an entry via the "
                "'adverse news' no-trade rule."
            ),
        }


__all__ = ["NewsClassifier", "Classification", "SENTIMENTS", "EVENT_TYPES"]
