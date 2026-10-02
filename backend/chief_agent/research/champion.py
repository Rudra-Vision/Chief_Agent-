"""Champion / Challenger registry.

At all times there is exactly ONE production champion. Anything the research
engine proposes is a CHALLENGER - a new immutable strategy version with exactly
one primary variable changed.

Rules enforced here (not by convention):

* A published version is **immutable** - its configuration is hashed and any
  later attempt to change that hash raises.
* The champion is only replaced by an explicit promotion call from the
  promotion gate. Nothing else may write ``status = CHAMPION``.
* Every version records its parent, the variable changed, the old/new values,
  the reason and the evidence attached to it.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..data.schema import StrategyVersion
from ..logging_setup import get_logger
from ..settings import get_config_store
from ..strategies.base import ImmutableConfig, bump_version, config_hash, deep_get
from ..timeutil import now_ist

log = get_logger(__name__, component="champion")

STATUS_FLOW = [
    "CANDIDATE",
    "TESTING",
    "WALK_FORWARD",
    "HOLDOUT",
    "PAPER",
    "ELIGIBLE",
    "CHAMPION",
    "REJECTED",
    "SUSPENDED",
    "RETIRED",
]

#: Statuses a version may transition to from each state. Anything else is a bug.
ALLOWED_TRANSITIONS: Dict[str, set] = {
    "CANDIDATE": {"TESTING", "REJECTED"},
    "TESTING": {"WALK_FORWARD", "REJECTED", "CANDIDATE"},
    "WALK_FORWARD": {"HOLDOUT", "REJECTED", "TESTING"},
    "HOLDOUT": {"PAPER", "REJECTED", "WALK_FORWARD"},
    "PAPER": {"ELIGIBLE", "REJECTED", "HOLDOUT"},
    "ELIGIBLE": {"CHAMPION", "REJECTED", "PAPER"},
    "CHAMPION": {"SUSPENDED", "RETIRED"},
    "SUSPENDED": {"CHAMPION", "RETIRED"},
    "RETIRED": set(),
    "REJECTED": set(),
}


@dataclass
class StrategyVersionInfo:
    version: str
    family: str
    status: str
    config: Dict[str, Any]
    config_hash: str
    parent_version: Optional[str] = None
    variable_changed: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    reason_for_change: Optional[str] = None
    created_at: Optional[dt.datetime] = None
    promoted_at: Optional[dt.datetime] = None
    retired_at: Optional[dt.datetime] = None
    backtest_summary: Optional[Dict[str, Any]] = None
    walk_forward_summary: Optional[Dict[str, Any]] = None
    holdout_summary: Optional[Dict[str, Any]] = None
    paper_summary: Optional[Dict[str, Any]] = None

    def to_dict(self, include_config: bool = False) -> Dict[str, Any]:
        payload = {
            "version": self.version,
            "family": self.family,
            "status": self.status,
            "config_hash": self.config_hash,
            "parent_version": self.parent_version,
            "variable_changed": self.variable_changed,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason_for_change": self.reason_for_change,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "backtest_summary": self.backtest_summary,
            "walk_forward_summary": self.walk_forward_summary,
            "holdout_summary": self.holdout_summary,
            "paper_summary": self.paper_summary,
        }
        if include_config:
            payload["config"] = self.config
        return payload


class ImmutableVersionError(RuntimeError):
    """Raised when something tries to modify an already-published version."""


class InvalidTransitionError(RuntimeError):
    """Raised when a status transition is not permitted by the state machine."""


class ChampionRegistry:
    """Database-backed registry of strategy versions."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- queries
    def get(self, version: str) -> Optional[StrategyVersion]:
        return self.session.execute(
            select(StrategyVersion).where(StrategyVersion.version == version)
        ).scalar_one_or_none()

    def info(self, version: str) -> Optional[StrategyVersionInfo]:
        row = self.get(version)
        return self._to_info(row) if row else None

    def champion(self, family: Optional[str] = None) -> Optional[StrategyVersionInfo]:
        stmt = select(StrategyVersion).where(StrategyVersion.status == "CHAMPION")
        if family:
            stmt = stmt.where(StrategyVersion.family == family)
        row = self.session.execute(stmt.order_by(StrategyVersion.promoted_at.desc())).scalars().first()
        return self._to_info(row) if row else None

    def list_versions(
        self,
        *,
        family: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[StrategyVersionInfo]:
        stmt = select(StrategyVersion)
        if family:
            stmt = stmt.where(StrategyVersion.family == family)
        if status:
            stmt = stmt.where(StrategyVersion.status == status)
        rows = self.session.execute(
            stmt.order_by(StrategyVersion.created_at.desc()).limit(limit)
        ).scalars().all()
        return [self._to_info(row) for row in rows]

    def config_for(self, version: Optional[str]) -> Dict[str, Any]:
        """The configuration to run. Falls back to ``config/strategy.yaml``."""
        if version:
            row = self.get(version)
            if row and row.config:
                return copy.deepcopy(row.config)
        return get_config_store().load("strategy")

    @staticmethod
    def _to_info(row: StrategyVersion) -> StrategyVersionInfo:
        return StrategyVersionInfo(
            version=row.version,
            family=row.family,
            status=row.status,
            config=copy.deepcopy(row.config or {}),
            config_hash=row.config_hash,
            parent_version=row.parent_version,
            variable_changed=row.variable_changed,
            old_value=row.old_value,
            new_value=row.new_value,
            reason_for_change=row.reason_for_change,
            created_at=row.created_at,
            promoted_at=row.promoted_at,
            retired_at=row.retired_at,
            backtest_summary=row.backtest_summary,
            walk_forward_summary=row.walk_forward_summary,
            holdout_summary=row.holdout_summary,
            paper_summary=row.paper_summary,
        )

    # -------------------------------------------------------------- publishing
    def publish(
        self,
        config: Dict[str, Any],
        *,
        version: Optional[str] = None,
        family: Optional[str] = None,
        parent_version: Optional[str] = None,
        variable_changed: Optional[str] = None,
        old_value: Any = None,
        new_value: Any = None,
        reason: str = "",
        bump: str = "patch",
        status: str = "CANDIDATE",
        git_commit: Optional[str] = None,
    ) -> StrategyVersionInfo:
        """Publish an immutable strategy version."""
        with self._lock:
            family = family or str(deep_get(config, "champion.strategy_family", "orb_vwap_retest"))
            if version is None:
                base = parent_version or str(deep_get(config, "champion.version", "ORB_v1.0.0"))
                version = bump_version(base, bump, family="ORB")
            if self.get(version) is not None:
                raise ImmutableVersionError(
                    f"strategy version {version} already exists and is immutable; "
                    f"create a new version instead of editing it"
                )
            frozen = ImmutableConfig(config)
            row = StrategyVersion(
                version=version,
                family=family,
                status=status,
                config=frozen.raw,
                config_hash=frozen.hash,
                parent_version=parent_version,
                variable_changed=variable_changed,
                old_value=_stringify(old_value),
                new_value=_stringify(new_value),
                reason_for_change=reason,
                git_commit=git_commit,
                published_at=now_ist(),
            )
            self.session.add(row)
            self.session.flush()
            from .artifacts import get_artifact_store

            get_artifact_store().strategy(
                {"version": version, "family": family, "status": status, "parent_version": parent_version,
                 "variable_changed": variable_changed, "old_value": old_value, "new_value": new_value,
                 "reason_for_change": reason, "config_hash": frozen.hash, "published_at": row.published_at},
                config=copy.deepcopy(config),
            )
            log.info(
                "strategy version published",
                context={
                    "version": version,
                    "parent": parent_version,
                    "variable": variable_changed,
                    "old": old_value,
                    "new": new_value,
                    "config_hash": frozen.hash,
                },
            )
            return self._to_info(row)

    def ensure_baseline_champion(self) -> StrategyVersionInfo:
        """Create the baseline champion from ``config/strategy.yaml`` if none exists."""
        existing = self.champion()
        if existing is not None:
            return existing
        config = get_config_store().load("strategy")
        version = str(deep_get(config, "champion.version", "ORB_v1.0.0"))
        row = self.get(version)
        if row is None:
            info = self.publish(
                config,
                version=version,
                family=str(deep_get(config, "champion.strategy_family", "orb_vwap_retest")),
                reason="Baseline interpretable strategy family, published as the first champion. "
                       "No optimisation was applied - this is the reference point.",
                status="CHAMPION",
            )
            row = self.get(version)
        stale = self.session.execute(
            select(StrategyVersion).where(
                StrategyVersion.status == "CHAMPION", StrategyVersion.version != version
            )
        ).scalars().all()
        for other in stale:
            other.status = "RETIRED"
            other.retired_at = now_ist()
        assert row is not None
        if row.promoted_at is None:
            row.promoted_at = now_ist()
        row.status = "CHAMPION"
        self.session.flush()
        return self._to_info(row)

    # -------------------------------------------------------------- transitions
    def transition(
        self,
        version: str,
        new_status: str,
        *,
        reason: str = "",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> StrategyVersionInfo:
        with self._lock:
            row = self.get(version)
            if row is None:
                raise KeyError(f"unknown strategy version {version}")
            if new_status not in STATUS_FLOW:
                raise ValueError(f"unknown status {new_status}")
            allowed = ALLOWED_TRANSITIONS.get(row.status, set())
            if new_status not in allowed and new_status != row.status:
                raise InvalidTransitionError(
                    f"cannot move {version} from {row.status} to {new_status} "
                    f"(allowed: {sorted(allowed) or 'none'})"
                )
            row.status = new_status
            if new_status == "CHAMPION":
                row.promoted_at = now_ist()
            if new_status == "RETIRED":
                row.retired_at = now_ist()
            if evidence:
                if "backtest" in evidence:
                    row.backtest_summary = evidence["backtest"]
                if "walk_forward" in evidence:
                    row.walk_forward_summary = evidence["walk_forward"]
                if "holdout" in evidence:
                    row.holdout_summary = evidence["holdout"]
                if "paper" in evidence:
                    row.paper_summary = evidence["paper"]
            if reason:
                row.notes = (row.notes or "") + f"\n[{row.status}] {now_ist().isoformat()} {reason}"
            self.session.flush()
            log.info("strategy version transition", context={"version": version, "status": new_status})
            return self._to_info(row)

    # ------------------------------------------------------------- challengers
    def create_challenger(
        self,
        *,
        variable: str,
        new_value: Any,
        reason: str,
        hypothesis_id: Optional[str] = None,
        bump: Optional[str] = None,
        parent_version: Optional[str] = None,
    ) -> tuple[StrategyVersionInfo, Dict[str, Any], Any]:
        """Clone the champion, change exactly ONE variable, publish a challenger.

        Returns ``(challenger_info, new_config, old_value)``. Raises if the parent
        is not found or the variable does not exist in the champion config.
        """
        parent = parent_version or (self.champion().version if self.champion() else None)
        if parent is None:
            raise KeyError("no champion strategy is registered")
        parent_row = self.get(parent)
        if parent_row is None:
            raise KeyError(f"unknown parent strategy {parent}")

        base_config = copy.deepcopy(parent_row.config or get_config_store().load("strategy"))
        old_value = deep_get(base_config, variable, None)
        if old_value is None and not _path_exists(base_config, variable):
            raise KeyError(
                f"variable '{variable}' does not exist in the champion configuration "
                f"- refusing to create a challenger that changes an unknown setting"
            )
        if old_value == new_value:
            raise ValueError(
                f"challenger would set '{variable}' to its current value ({new_value!r}); "
                f"a challenger must change exactly one variable"
            )

        config = copy.deepcopy(base_config)
        _set_path(config, variable, new_value)

        # Verify the diff is exactly one leaf.
        changes = _diff_leaves(base_config, config)
        if len(changes) != 1:
            raise ValueError(
                f"a challenger must change exactly one variable; this diff touches {len(changes)}: {changes}"
            )

        version = bump_version(parent, bump or _infer_bump(variable), family="ORB")
        if self.get(version) is not None:
            version = bump_version(version, "patch", family="ORB")

        info = self.publish(
            config,
            version=version,
            family=parent_row.family,
            parent_version=parent,
            variable_changed=variable,
            old_value=old_value,
            new_value=new_value,
            reason=reason + (f" (hypothesis {hypothesis_id})" if hypothesis_id else ""),
            status="CANDIDATE",
        )
        return info, config, old_value

    # ------------------------------------------------------------------ count
    def version_count(self) -> int:
        return int(self.session.execute(select(StrategyVersion)).scalars().all().__len__())


def _path_exists(config: Dict[str, Any], path: str) -> bool:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _set_path(config: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = config
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _diff_leaves(a: Any, b: Any, prefix: str = "") -> List[str]:
    """Leaf paths that differ between two nested structures."""
    out: List[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            out.extend(_diff_leaves(a.get(key), b.get(key), f"{prefix}.{key}" if prefix else key))
    elif isinstance(a, list) and isinstance(b, list):
        if a != b:
            out.append(prefix)
    else:
        if a != b:
            out.append(prefix)
    return out


def _infer_bump(variable: str) -> str:
    """patch = a value retune, minor = a new rule/filter, major = structure."""
    structural_prefixes = ("stops.model", "targets.model", "champion.strategy_family")
    if variable.startswith(structural_prefixes):
        return "major"
    rule_suffixes = ("enabled", "require_retest", "require_index_above_vwap", "allowed_regimes")
    if variable.endswith(rule_suffixes):
        return "minor"
    return "patch"


def _stringify(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text[:160]


__all__ = [
    "ChampionRegistry",
    "StrategyVersionInfo",
    "ImmutableVersionError",
    "InvalidTransitionError",
    "STATUS_FLOW",
    "ALLOWED_TRANSITIONS",
]
