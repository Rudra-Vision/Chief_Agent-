"""News routes: fetch, classify and contextualise market news."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from ...data.db import session_scope
from ...data.schema import NewsArticle
from ...logging_setup import get_logger
from ...news.sentiment import NewsClassifier
from ...timeutil import now_ist
from ..state import AppState, get_app_state

log = get_logger(__name__, component="api.news")
router = APIRouter()


@router.get("/news", summary="Latest market news with deterministic classification")
def news(
    instrument_keys: str = Query("", description="Comma-separated instrument keys (max 30)"),
    limit: int = Query(30, ge=1, le=200),
    refresh: bool = Query(False),
    state: AppState = Depends(get_app_state),
) -> Dict[str, Any]:
    classifier = NewsClassifier()
    items: List[Dict[str, Any]] = []

    if refresh and state.broker.has_token and instrument_keys:
        keys = [k.strip() for k in instrument_keys.split(",") if k.strip()][:30]
        try:
            fetched = state.broker.news.by_instruments(keys, page_size=100)
        except Exception as exc:
            log.warning("news fetch failed", context={"error": str(exc)})
            fetched = []
        for item in fetched:
            classification = classifier.classify(item.heading, item.summary)
            payload = item.to_dict()
            payload["classification"] = classification.to_dict()
            items.append(payload)
            try:
                with session_scope() as session:
                    exists = session.execute(
                        select(NewsArticle).where(NewsArticle.news_id == item.news_id)
                    ).scalar_one_or_none()
                    if exists is None:
                        session.add(
                            NewsArticle(
                                news_id=item.news_id,
                                instrument_key=item.instrument_key,
                                heading=item.heading,
                                summary=item.summary,
                                article_link=item.article_link,
                                thumbnail=item.thumbnail,
                                published_time=item.published_time or now_ist(),
                                sentiment=classification.sentiment,
                                sentiment_score=classification.sentiment_score,
                                event_type=classification.event_type,
                                classifier="lexicon",
                                is_adverse_for_open_position=classification.is_adverse,
                            )
                        )
                        session.commit()
            except Exception as exc:  # storage is best-effort
                log.warning("could not persist news", context={"error": str(exc)})

    if not items:
        with session_scope() as session:
            rows = (
                session.execute(
                    select(NewsArticle).order_by(NewsArticle.published_time.desc()).limit(limit)
                )
                .scalars()
                .all()
            )
        items = [
            {
                "news_id": row.news_id,
                "instrument_key": row.instrument_key,
                "heading": row.heading,
                "summary": row.summary,
                "article_link": row.article_link,
                "published_time": row.published_time.isoformat() if row.published_time else None,
                "classification": {
                    "sentiment": row.sentiment,
                    "sentiment_score": row.sentiment_score,
                    "event_type": row.event_type,
                    "classifier": row.classifier,
                    "adverse_for_long": row.is_adverse_for_open_position,
                },
            }
            for row in rows
        ]

    return {
        "count": len(items),
        "items": items[:limit],
        "market_context": classifier.market_context(items),
        "llm_used": False,
        "note": (
            "News is classified by a transparent keyword lexicon. No trading decision is ever made "
            "because of a sentiment label; news can only add context or veto an entry through the "
            "'adverse news' no-trade rule. The Upstox News API returns articles from the past 7 days."
        ),
    }


@router.get("/news/context", summary="News context for the current watchlist / positions")
def context(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    if not state.broker.has_token:
        return {
            "available": False,
            "reason": "no Upstox token configured - the News API needs authentication",
            "market_context": {"articles": 0, "sentiment": "neutral", "score": 0.0},
        }
    keys = [position.instrument_key for position in state.paper.open_positions()][:30]
    if not keys:
        watchlist = state.broker.watchlist
        watchlist.load()
        resolved = watchlist.resolved(state.broker.instruments if state.broker.instruments.is_loaded else None)
        keys = [row["instrument_key"] for row in resolved if row.get("instrument_key")][:30]
    if not keys:
        return {"available": False, "reason": "no instrument keys to look up"}

    try:
        items = state.broker.news.by_instruments(keys)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    classifier = NewsClassifier()
    payloads = []
    for item in items:
        classification = classifier.classify(item.heading, item.summary)
        payload = item.to_dict()
        payload["classification"] = classification.to_dict()
        payloads.append(payload)

    return {
        "available": True,
        "instruments_queried": len(keys),
        "items": payloads[:50],
        "market_context": classifier.market_context(payloads),
    }


__all__ = ["router"]
