#!/usr/bin/env python
"""Download and cache historical candles.

Cache-first: sessions already held locally are never re-requested.

    python scripts/download_data.py --years 1 --timeframe 1m --max-instruments 60
    python scripts/download_data.py --symbols RELIANCE,INFY --start 2025-01-01 --end 2025-06-30
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Chief Agent historical downloader")
    parser.add_argument("--symbols", default="", help="comma separated; default = the whole watchlist")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--years", type=int, default=1)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-instruments", type=int, default=60)
    parser.add_argument("--force", action="store_true", help="re-fetch even if cached")
    parser.add_argument("--build-higher-timeframes", action="store_true")
    args = parser.parse_args()

    from chief_agent.data.db import init_db, session_scope
    from chief_agent.data.historical_downloader import HistoricalDownloader
    from chief_agent.data.provider import MarketDataProvider
    from chief_agent.logging_setup import configure_logging

    configure_logging(level="INFO")
    init_db()
    provider = MarketDataProvider()
    print(f"Data source: {provider.data_source}")

    from chief_agent.broker.upstox_instruments import load_instruments_into_db
    from chief_agent.data.db import session_scope as scope

    # Load the official instrument master if it is already cached, so real
    # instrument_key values are used where available.
    with scope() as session:
        from chief_agent.broker.upstox_instruments import InstrumentMaster, Watchlist

        master = InstrumentMaster()
        try:
            if master.cache_path("nse").exists():
                master.load("nse", auto_download=False)
        except Exception as exc:
            print(f"  (instrument master not loaded: {exc})")
        watchlist = Watchlist(master=master)
        watchlist.load()
        if master.is_loaded:
            info = load_instruments_into_db(session, master, watchlist)
            session.commit()
            print(f"  instrument master: {info}")
        resolved = watchlist.resolved(master if master.is_loaded else None)

    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        resolved = [row for row in resolved if row["symbol"] in wanted]

    instruments = []
    for row in resolved:
        instruments.append(
            {"instrument_key": row.get("instrument_key") or f"SIM|{row['symbol']}", "symbol": row["symbol"]}
        )

    # Benchmarks power the regime engine and relative strength.
    instruments.insert(0, {"instrument_key": "__index__India VIX", "symbol": "India VIX"})
    instruments.insert(0, {"instrument_key": "__index__NIFTY 50", "symbol": "NIFTY 50"})

    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    start = dt.date.fromisoformat(args.start) if args.start else end - dt.timedelta(days=365 * args.years)

    def progress(index: int, total: int, symbol: str, inserted: int) -> None:
        if index % 10 == 0 or index == total:
            print(f"  [{index}/{total}] {symbol:<16} +{inserted} candles")

    with session_scope() as session:
        downloader = HistoricalDownloader(provider, session)
        report = downloader.download(
            instruments[: args.max_instruments + 2],
            args.timeframe,
            start_date=start,
            end_date=end,
            force=args.force,
            progress=progress,
        )
        if args.build_higher_timeframes:
            keys = [row["instrument_key"] for row in instruments[: args.max_instruments + 2] if row.get("instrument_key")]
            written = downloader.build_higher_timeframes(keys, ("5m", "15m", "30m", "60m"), start_date=start, end_date=end)
            print(f"  higher timeframes: {len(written)} series written")
        session.commit()

    payload = report.to_dict()
    print()
    print(f"Inserted      {payload['candles_inserted']:,} candles")
    print(f"Already had   {payload['candles_duplicate']:,} (skipped, never re-requested)")
    print(f"Invalid       {payload['candles_invalid']:,}")
    print(f"Instruments   {payload['instruments_fetched']}/{payload['instruments_requested']} fetched, "
          f"{payload['instruments_skipped_cached']} fully cached")
    print(f"Duration      {payload['duration_seconds']}s")
    if payload["failures"]:
        print(f"Failures      {payload['failure_count']}")
        for failure in payload["failures"][:5]:
            print(f"  {failure.get('symbol')}: {failure.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
