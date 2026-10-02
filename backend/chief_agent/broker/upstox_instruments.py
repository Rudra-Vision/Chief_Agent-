"""Instruments: the BOD (beginning-of-day) master.

Source (official):
    https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz
    https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz
    https://assets.upstox.com/market-quote/instruments/exchange/global.json.gz
    https://assets.upstox.com/market-quote/instruments/exchange/suspended-instrument.json.gz

Upstox guidance followed here:
  * ``instrument_key`` is the canonical identifier - ``exchange_token`` may be
    REUSED by the exchange for a different instrument after expiry, so it is
    never used as a durable key.
  * JSON is preferred over CSV because the structure is more robust.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

from ..logging_setup import get_logger
from ..settings import CACHE_DIR, REPO_ROOT, Settings, get_settings
from ..timeutil import ensure_ist, ist_date, now_ist
from .upstox_client import Endpoints, UpstoxError, UpstoxHttpClient

log = get_logger(__name__, component="instruments")

ASSET_URLS = {
    "nse": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    "complete": "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz",
    "global": "https://assets.upstox.com/market-quote/instruments/exchange/global.json.gz",
    "suspended": "https://assets.upstox.com/market-quote/instruments/exchange/suspended-instrument.json.gz",
    "nse_mis": "https://assets.upstox.com/market-quote/instruments/exchange/NSE_MIS.json.gz",
}

# Sectors that NSE publishes as indices; used to map a stock to a tradeable
# sector index instrument_key for relative-strength work.
SECTOR_INDEX_KEYS: Dict[str, str] = {
    "Financial Services": "NSE_INDEX|Nifty Bank",
    "Information Technology": "NSE_INDEX|Nifty IT",
    "Automobile": "NSE_INDEX|Nifty Auto",
    "Fast Moving Consumer Goods": "NSE_INDEX|Nifty FMCG",
    "Healthcare": "NSE_INDEX|Nifty Pharma",
    "Metals": "NSE_INDEX|Nifty Metal",
    "Energy": "NSE_INDEX|Nifty Energy",
    "Power": "NSE_INDEX|Nifty Energy",
    "Realty": "NSE_INDEX|Nifty Realty",
    "Media": "NSE_INDEX|Nifty Media",
    "Cement": "NSE_INDEX|Nifty Infra",
    "Construction": "NSE_INDEX|Nifty Infra",
    "Telecommunication": "NSE_INDEX|Nifty India Digital",
    "Chemicals": "NSE_INDEX|Nifty Commodities",
    "Textiles": "NSE_INDEX|Nifty Commodities",
    "Industrial Manufacturing": "NSE_INDEX|Nifty India Manufacturing",
    "Consumer Durables": "NSE_INDEX|Nifty Consumer Durables",
    "Consumer Services": "NSE_INDEX|Nifty Consumption",
    "Logistics": "NSE_INDEX|Nifty Infra",
}

# Broad-market indices + India VIX, per the V3 docs (available for historical
# candles and the market feed).
BENCHMARK_KEYS = {
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "NIFTY BANK": "NSE_INDEX|Nifty Bank",
    "NIFTY IT": "NSE_INDEX|Nifty IT",
    "NIFTY NEXT 50": "NSE_INDEX|Nifty Next 50",
    "INDIA VIX": "NSE_INDEX|India VIX",
}


@dataclass
class InstrumentRecord:
    instrument_key: str
    trading_symbol: str
    name: str = ""
    exchange: str = "NSE"
    segment: str = "NSE_EQ"
    instrument_type: str = "EQ"
    isin: str = ""
    lot_size: int = 1
    tick_size: float = 0.05
    freeze_quantity: Optional[float] = None
    security_type: str = "NORMAL"
    cas_eligible: bool = False
    exchange_token: str = ""
    expiry_ms: Optional[int] = None
    underlying_symbol: str = ""
    strike_price: Optional[float] = None
    sector: Optional[str] = None
    tier: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_bod(cls, row: Dict[str, Any]) -> "InstrumentRecord":
        def _f(key: str, default: Any = None) -> Any:
            value = row.get(key, default)
            return value if value is not None else default

        return cls(
            instrument_key=str(_f("instrument_key", "")),
            trading_symbol=str(_f("trading_symbol", "") or _f("short_name", "")),
            name=str(_f("name", "")),
            exchange=str(_f("exchange", "NSE")),
            segment=str(_f("segment", "NSE_EQ")),
            instrument_type=str(_f("instrument_type", "EQ")),
            isin=str(_f("isin", "") or ""),
            lot_size=int(_f("lot_size", 1) or 1),
            tick_size=float(_f("tick_size", 0.05) or 0.05),
            freeze_quantity=_f("freeze_quantity"),
            security_type=str(_f("security_type", "NORMAL") or "NORMAL"),
            cas_eligible=bool(_f("cas_eligible", False)),
            exchange_token=str(_f("exchange_token", "") or ""),
            expiry_ms=_f("expiry"),
            underlying_symbol=str(_f("underlying_symbol", "") or ""),
            strike_price=_f("strike_price"),
            raw=row,
        )

    @property
    def expiry_date(self):
        if not self.expiry_ms:
            return None
        return ist_date(self.expiry_ms)

    @property
    def is_equity(self) -> bool:
        return self.segment == "NSE_EQ" and self.instrument_type == "EQ"

    @property
    def is_index(self) -> bool:
        return self.segment.endswith("_INDEX")

    @property
    def is_derivative(self) -> bool:
        return self.instrument_type in ("FUT", "CE", "PE")


class InstrumentMaster:
    """Download, cache, index and query the Upstox instrument master."""

    CACHE_SUBDIR = "instruments"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = CACHE_DIR / self.CACHE_SUBDIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._by_key: Dict[str, InstrumentRecord] = {}
        self._by_symbol: Dict[str, InstrumentRecord] = {}
        self._loaded_from: Optional[str] = None

    # ------------------------------------------------------------------ paths
    def cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.json"

    def cache_age_hours(self, name: str = "nse") -> Optional[float]:
        path = self.cache_path(name)
        if not path.exists():
            return None
        return (now_ist().timestamp() - path.stat().st_mtime) / 3600.0

    # --------------------------------------------------------------- download
    def download(self, name: str = "nse", force: bool = False) -> Path:
        """Download a BOD instrument file (gzipped JSON) into the local cache."""
        url = ASSET_URLS.get(name)
        if not url:
            raise ValueError(f"unknown instrument file: {name}")
        target = self.cache_path(name)
        if target.exists() and not force and (self.cache_age_hours(name) or 0) < 12:
            return target

        log.info("downloading instrument master", context={"name": name, "url": url})
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.get(url, headers={"Accept": "*/*"})
            response.raise_for_status()
            raw = response.content

        if url.endswith(".gz"):
            try:
                text = gzip.decompress(raw).decode("utf-8")
            except (OSError, UnicodeDecodeError):
                text = raw.decode("utf-8")
        else:
            text = raw.decode("utf-8")

        payload = json.loads(text)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)
        log.info("instrument master cached", context={"name": name, "count": len(payload)})
        return target

    # ------------------------------------------------------------------- load
    def load(self, name: str = "nse", auto_download: bool = True, force: bool = False) -> int:
        path = self.cache_path(name)
        if force or not path.exists():
            if not auto_download:
                raise FileNotFoundError(f"instrument cache not present at {path}")
            path = self.download(name, force=force)
        elif (self.cache_age_hours(name) or 0) > 20 and auto_download:
            try:
                path = self.download(name, force=False)
            except Exception as exc:  # network is optional - stale cache still works
                log.warning("instrument refresh failed; using cache", context={"error": str(exc)})

        rows = json.loads(path.read_text(encoding="utf-8"))
        self._by_key.clear()
        self._by_symbol.clear()
        for row in rows:
            if not isinstance(row, dict):
                continue
            record = InstrumentRecord.from_bod(row)
            if not record.instrument_key:
                continue
            self._by_key[record.instrument_key] = record
            if record.segment == "NSE_EQ" and record.instrument_type == "EQ":
                symbol = record.trading_symbol.upper()
                # Prefer the canonical series; keep the first EQ seen per symbol.
                self._by_symbol.setdefault(symbol, record)
        self._loaded_from = name
        log.info(
            "instrument master loaded",
            context={"source": name, "instruments": len(self._by_key), "equities": len(self._by_symbol)},
        )
        return len(self._by_key)

    @property
    def is_loaded(self) -> bool:
        return bool(self._by_key)

    @property
    def size(self) -> int:
        return len(self._by_key)

    # ------------------------------------------------------------------ query
    def get(self, instrument_key: str) -> Optional[InstrumentRecord]:
        return self._by_key.get(instrument_key)

    def by_symbol(self, symbol: str) -> Optional[InstrumentRecord]:
        return self._by_symbol.get((symbol or "").strip().upper())

    def search(self, query: str, limit: int = 25) -> List[InstrumentRecord]:
        q = (query or "").strip().upper()
        if not q:
            return []
        out: List[InstrumentRecord] = []
        for record in self._by_key.values():
            if q in record.trading_symbol.upper() or q in (record.name or "").upper() or q in record.instrument_key.upper():
                out.append(record)
                if len(out) >= limit:
                    break
        return out

    def equities(self) -> List[InstrumentRecord]:
        return list(self._by_symbol.values())

    def resolve_sector_index(self, sector: Optional[str]) -> Optional[str]:
        if not sector:
            return None
        return SECTOR_INDEX_KEYS.get(sector)

    def all_sector_index_keys(self) -> List[str]:
        return sorted(set(SECTOR_INDEX_KEYS.values()) | set(BENCHMARK_KEYS.values()))

    def benchmark_key(self, symbol: str) -> Optional[str]:
        return BENCHMARK_KEYS.get((symbol or "").strip().upper())


class Watchlist:
    """The user-editable trading universe (``config/universe_nifty.csv``).

    The CSV intentionally contains SYMBOLS only. ``instrument_key`` values are
    resolved against the official instrument master so nothing is hard-coded.
    """

    def __init__(self, settings: Optional[Settings] = None, master: Optional[InstrumentMaster] = None) -> None:
        self.settings = settings or get_settings()
        self.master = master
        self.rows: List[Dict[str, Any]] = []
        self.path = REPO_ROOT / "config" / "universe_nifty.csv"

    def load(self, path: Optional[Path] = None) -> List[Dict[str, Any]]:
        target = Path(path or self.path)
        if not target.exists():
            self.rows = []
            return self.rows
        rows: List[Dict[str, Any]] = []
        with target.open("r", encoding="utf-8", newline="") as handle:
            # Skip leading comment lines, then parse the CSV normally.
            lines = [line for line in handle if line.strip() and not line.startswith("#")]
        reader = csv.DictReader(io.StringIO("".join(lines)))
        for row in reader:
            symbol = (row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "name": (row.get("name") or "").strip(),
                    "sector": (row.get("sector") or "Unknown").strip(),
                    "tier": int((row.get("tier") or "2").strip() or 2),
                }
            )
        self.rows = rows
        log.info("watchlist loaded", context={"count": len(rows), "path": str(target)})
        return rows

    def resolved(self, master: Optional[InstrumentMaster] = None) -> List[Dict[str, Any]]:
        """Attach instrument_key / isin / tick_size by joining the master."""
        master = master or self.master
        out: List[Dict[str, Any]] = []
        unresolved: List[str] = []
        for row in self.rows:
            if master and master.is_loaded:
                record = master.by_symbol(row["symbol"])
                if record is None:
                    unresolved.append(row["symbol"])
                    continue
                merged = dict(row)
                merged.update(
                    {
                        "instrument_key": record.instrument_key,
                        "isin": record.isin,
                        "tick_size": record.tick_size,
                        "lot_size": record.lot_size,
                        "exchange_token": record.exchange_token,
                        "resolved": True,
                    }
                )
                out.append(merged)
            else:
                merged = dict(row)
                merged.update({"instrument_key": None, "resolved": False})
                out.append(merged)
        if unresolved:
            log.warning(
                "watchlist symbols not found in the instrument master",
                context={"count": len(unresolved), "symbols": unresolved[:20]},
            )
        return out


def load_instruments_into_db(session, master: InstrumentMaster, watchlist: Optional[Watchlist] = None, name: str = "nse") -> Dict[str, int]:
    """Persist the master (+ watchlist flags) into the ``instruments`` table.

    Uses ``instrument_key`` as the upsert key. Existing rows are updated in place;
    source market data is never touched by this function.
    """
    from sqlalchemy import select

    from ..data.schema import Instrument

    if not master.is_loaded:
        master.load(name)

    universe_map: Dict[str, Dict[str, Any]] = {}
    if watchlist is not None:
        if not watchlist.rows:
            watchlist.load()
        for row in watchlist.resolved(master):
            if row.get("instrument_key"):
                universe_map[row["instrument_key"]] = row

    suspended_keys: set[str] = set()
    try:
        suspended_path = master.cache_path("suspended")
        if suspended_path.exists():
            for entry in json.loads(suspended_path.read_text(encoding="utf-8")):
                key = entry.get("instrument_key") if isinstance(entry, dict) else None
                if key:
                    suspended_keys.add(key)
    except Exception as exc:  # pragma: no cover - optional data
        log.warning("could not read suspended instrument list", context={"error": str(exc)})

    created = 0
    updated = 0
    for key, record in master._by_key.items():  # noqa: SLF001 - same module family
        existing = session.execute(select(Instrument).where(Instrument.instrument_key == key)).scalar_one_or_none()
        meta = universe_map.get(key)
        values = dict(
            exchange_token=record.exchange_token,
            segment=record.segment,
            exchange=record.exchange,
            instrument_type=record.instrument_type,
            trading_symbol=record.trading_symbol,
            short_name=record.trading_symbol,
            name=record.name,
            isin=record.isin,
            lot_size=record.lot_size,
            freeze_quantity=record.freeze_quantity,
            tick_size=record.tick_size,
            security_type=record.security_type,
            cas_eligible=record.cas_eligible,
            underlying_symbol=record.underlying_symbol or None,
            strike_price=record.strike_price,
            is_suspended=key in suspended_keys,
            source=f"upstox_bod:{name}",
        )
        if meta:
            values.update(sector=meta.get("sector"), tier=meta.get("tier"), in_universe=True)
        if existing is None:
            session.add(Instrument(instrument_key=key, **values))
            created += 1
        else:
            for field_name, value in values.items():
                setattr(existing, field_name, value)
            if not meta:
                existing.in_universe = False
            updated += 1

    session.flush()
    result = {"created": created, "updated": updated, "universe": len(universe_map), "suspended": len(suspended_keys)}
    log.info("instruments persisted", context=result)
    return result


__all__ = [
    "InstrumentMaster",
    "InstrumentRecord",
    "Watchlist",
    "load_instruments_into_db",
    "ASSET_URLS",
    "SECTOR_INDEX_KEYS",
    "BENCHMARK_KEYS",
]
