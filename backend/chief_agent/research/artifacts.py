"""Human-readable mirrors of the research artefacts.

The database is authoritative. These files exist so the research history can be
inspected, diffed, reviewed offline and backed up as plain text:

    research_data/journal/       one JSON file per completed trade
    research_data/experiments/   one JSON file per experiment
    research_data/strategies/    the immutable published configuration per version
    research_data/candidates/    challengers awaiting validation
    research_data/promoted/      versions that became champion, with evidence
    research_data/rejected/      versions that failed the gate, with the reason

Nothing here is read back as an input to a trading decision.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from ..logging_setup import get_logger
from ..settings import RESEARCH_DIR
from ..timeutil import now_ist

log = get_logger(__name__, component="research.artifacts")

SUBDIRS = ("journal", "experiments", "strategies", "candidates", "rejected", "promoted")


def _default(value: Any) -> Any:
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


class ArtifactStore:
    """Writes JSON mirrors of research artefacts."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root or RESEARCH_DIR)
        for name in SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ write
    def _write(self, subdir: str, filename: str, payload: Mapping[str, Any]) -> Optional[Path]:
        try:
            directory = self.root / subdir
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / filename
            path.write_text(
                json.dumps(payload, indent=2, default=_default, ensure_ascii=False), encoding="utf-8"
            )
            return path
        except Exception as exc:  # artifact mirrors must never break trading
            log.warning("could not write a research artifact", context={"subdir": subdir, "error": str(exc)})
            return None

    def trade(self, trade: Mapping[str, Any]) -> Optional[Path]:
        trade_id = str(trade.get("trade_id") or f"trade-{now_ist().timestamp()}")
        return self._write("journal", f"{trade_id}.json", {"written_at": now_ist(), **dict(trade)})

    def strategy(self, version: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
        payload = {"written_at": now_ist(), **dict(version)}
        if config is not None:
            payload["config"] = dict(config)
        return self._write("strategies", f"{version.get('version')}.json", payload)

    def experiment(self, outcome: Mapping[str, Any]) -> Optional[Path]:
        experiment_id = str(outcome.get("experiment_id") or f"exp-{now_ist().timestamp()}")
        version = str(outcome.get("challenger_version") or "")
        status = str(outcome.get("status") or "")
        payload = {"written_at": now_ist(), **dict(outcome)}
        path = self._write("experiments", f"{experiment_id}.json", payload)

        # A copy into the state-specific folder makes the history browsable.
        if status in ("FAILED", "FAILED_GATE"):
            self._write("rejected", f"{experiment_id}-{version}.json", payload)
        elif status == "PROMOTED":
            self._write("promoted", f"{experiment_id}-{version}.json", payload)
        elif status in ("CREATED", "TESTING", "WALK_FORWARD", "HOLDOUT", "PAPER_VALIDATION"):
            self._write("candidates", f"{experiment_id}-{version}.json", payload)
        elif status in ("INSUFFICIENT_SAMPLE", "REJECTED"):
            self._write("rejected", f"{experiment_id}-{version}.json", payload)
        return path

    def hypothesis(self, hypothesis: Mapping[str, Any]) -> Optional[Path]:
        hypothesis_id = str(hypothesis.get("hypothesis_id") or f"h-{now_ist().timestamp()}")
        return self._write("experiments", f"{hypothesis_id}-hypothesis.json", {"written_at": now_ist(), **dict(hypothesis)})

    # ------------------------------------------------------------------- read
    def list(self, subdir: str) -> list:
        directory = self.root / subdir
        if not directory.exists():
            return []
        return sorted(path.name for path in directory.glob("*.json"))

    def summary(self) -> Dict[str, Any]:
        return {
            "root": str(self.root),
            "counts": {name: len(self.list(name)) for name in SUBDIRS},
        }


_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    global _store
    if _store is None:
        _store = ArtifactStore()
    return _store


def reset_artifact_store() -> None:
    global _store
    _store = None


__all__ = ["ArtifactStore", "get_artifact_store", "reset_artifact_store", "SUBDIRS"]
