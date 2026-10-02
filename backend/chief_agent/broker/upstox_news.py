"""Upstox News API.

Verified endpoint:   GET /v2/news
    Query parameters:
        category          required: instrument_keys | positions | holdings
        instrument_keys   comma-separated, MAXIMUM 30 per request
                          (required when category=instrument_keys)
        page_number       1..100 (default 1)
        page_size         1..100 (default 100)
    Returns news published in the PAST 7 DAYS.

Response rows: heading, summary, thumbnail, article_link, published_time (epoch ms).

The broker module only fetches and normalises. Classification lives in
``chief_agent.news.sentiment`` and is deterministic by default - an LLM label is
never on its own a reason to trade.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..logging_setup import get_logger
from ..settings import Settings, get_settings
from ..timeutil import ensure_ist, now_ist
from .upstox_client import Endpoints, UpstoxError, UpstoxHttpClient

log = get_logger(__name__, component="news")

MAX_INSTRUMENT_KEYS_PER_REQUEST = 30
VALID_CATEGORIES = {"instrument_keys", "positions", "holdings"}


@dataclass
class NewsItem:
    news_id: str
    heading: str
    summary: str = ""
    article_link: str = ""
    thumbnail: str = ""
    published_time: Any = None
    instrument_key: Optional[str] = None
    symbol: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def published_iso(self) -> Optional[str]:
        return self.published_time.isoformat() if self.published_time else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "news_id": self.news_id,
            "heading": self.heading,
            "summary": self.summary,
            "article_link": self.article_link,
            "thumbnail": self.thumbnail,
            "published_time": self.published_iso,
            "instrument_key": self.instrument_key,
            "symbol": self.symbol,
        }


def _make_news_id(instrument_key: Optional[str], heading: str, published_ms: Any) -> str:
    digest = hashlib.sha1(f"{instrument_key}|{heading}|{published_ms}".encode("utf-8")).hexdigest()
    return digest[:24]


class UpstoxNews:
    """Fetches and normalises news. Read-only."""

    def __init__(self, client: UpstoxHttpClient, settings: Optional[Settings] = None) -> None:
        self.client = client
        self.settings = settings or get_settings()
        self._cache: Dict[str, List[NewsItem]] = {}
        self._cache_ts: Optional[Any] = None

    def _cache_fresh(self) -> bool:
        if self._cache_ts is None:
            return False
        return (now_ist() - self._cache_ts).total_seconds() < 300

    def by_instruments(self, instrument_keys: Sequence[str], page_size: int = 100) -> List[NewsItem]:
        """Fetch news for up to 30 instrument keys per call (chunked automatically)."""
        keys = [k for k in instrument_keys if k]
        if not keys:
            return []
        out: List[NewsItem] = []
        for start in range(0, len(keys), MAX_INSTRUMENT_KEYS_PER_REQUEST):
            chunk = keys[start : start + MAX_INSTRUMENT_KEYS_PER_REQUEST]
            out.extend(self._fetch(category="instrument_keys", instrument_keys=chunk, page_size=page_size))
        return out

    def for_positions(self, page_size: int = 100) -> List[NewsItem]:
        return self._fetch(category="positions", page_size=page_size)

    def for_holdings(self, page_size: int = 100) -> List[NewsItem]:
        return self._fetch(category="holdings", page_size=page_size)

    def _fetch(
        self,
        category: str,
        instrument_keys: Optional[Sequence[str]] = None,
        page_number: int = 1,
        page_size: int = 100,
    ) -> List[NewsItem]:
        if category not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}")
        params: Dict[str, Any] = {
            "category": category,
            "page_number": max(1, min(100, page_number)),
            "page_size": max(1, min(100, page_size)),
        }
        if category == "instrument_keys":
            keys = [k for k in (instrument_keys or []) if k]
            if not keys:
                raise ValueError("instrument_keys is required when category=instrument_keys")
            if len(keys) > MAX_INSTRUMENT_KEYS_PER_REQUEST:
                raise ValueError(f"news supports at most {MAX_INSTRUMENT_KEYS_PER_REQUEST} instrument keys per request")
            params["instrument_keys"] = ",".join(keys)

        try:
            response = self.client.get(Endpoints.NEWS, params=params, limit_class="standard")
        except UpstoxError as exc:
            log.warning("news fetch failed", context={"category": category, "error": str(exc)})
            return []

        data = response.data
        items: List[NewsItem] = []
        if isinstance(data, dict):
            for instrument_key, articles in data.items():
                if not isinstance(articles, list):
                    continue
                for article in articles:
                    if isinstance(article, dict):
                        items.append(self._normalise(article, instrument_key if instrument_key != "all" else None))
        items.sort(key=lambda item: item.published_time or now_ist(), reverse=True)
        return items

    @staticmethod
    def _normalise(article: Mapping[str, Any], instrument_key: Optional[str]) -> NewsItem:
        raw_time = article.get("published_time")
        try:
            published = ensure_ist(raw_time) if raw_time is not None else None
        except Exception:
            published = None
        heading = str(article.get("heading") or "")[:2000]
        return NewsItem(
            news_id=_make_news_id(instrument_key, heading, raw_time),
            heading=heading,
            summary=str(article.get("summary") or "")[:4000],
            article_link=str(article.get("article_link") or ""),
            thumbnail=str(article.get("thumbnail") or ""),
            published_time=published,
            instrument_key=instrument_key,
            raw=dict(article),
        )


__all__ = ["UpstoxNews", "NewsItem", "MAX_INSTRUMENT_KEYS_PER_REQUEST", "VALID_CATEGORIES"]
