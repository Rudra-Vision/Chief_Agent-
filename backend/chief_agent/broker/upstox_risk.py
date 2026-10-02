"""Broker-side risk controls: the Upstox Kill Switch and static-IP registration.

Verified against the current official documentation:

Kill Switch  -  POST /v2/user/kill-switch
    Body is an ARRAY of ``{"segment": ..., "action": "ENABLE"|"DISABLE"}``.
    Segments: BSE_EQ, NSE_EQ, NCD_FO, BCD_FO, NSE_FO, BSE_FO, MCX_FO, NSE_COM.

    Documented platform rules that this module RESPECTS and never works around:
      * all open positions in a segment must be closed before disabling it;
      * a 12-hour cooling period applies before a disabled segment can be
        re-enabled;
      * disabling a segment cancels all its open orders automatically;
      * you must regenerate the access token after calling it for the change to
        take effect for that token;
      * if any segment in the request fails, none are updated.

Static IP    -  PUT /v2/user/ip
    ``{"primary_ip": ..., "secondary_ip": ...}``. Can only be changed once per
    calendar week, and a successful update invalidates existing access tokens.
    When enforcement is active, orders may be rejected unless traffic originates
    from a registered IP.

Our own (local, instant) kill switch is always the first line of defence; the
broker-side switch is an additional account-level backstop used in LIVE mode.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..logging_setup import get_logger
from ..settings import Settings, get_config_store, get_settings
from ..timeutil import now_ist
from .upstox_client import Endpoints, UpstoxError, UpstoxHttpClient

log = get_logger(__name__, component="broker_risk")

VALID_SEGMENTS = {
    "BSE_EQ", "NSE_EQ", "NCD_FO", "BCD_FO", "NSE_FO", "BSE_FO", "MCX_FO", "NSE_COM",
}
VALID_ACTIONS = {"ENABLE", "DISABLE"}


@dataclass
class SegmentStatus:
    segment: str
    segment_status: str = "UNKNOWN"        # ACTIVE | INACTIVE, independent of the switch
    kill_switch_enabled: bool = False

    @property
    def trading_blocked(self) -> bool:
        return self.kill_switch_enabled or self.segment_status.upper() == "INACTIVE"


@dataclass
class KillSwitchResult:
    ok: bool
    engine: str = "upstox"
    segments: List[SegmentStatus] = field(default_factory=list)
    message: str = ""
    token_regeneration_required: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "engine": self.engine,
            "message": self.message,
            "token_regeneration_required": self.token_regeneration_required,
            "segments": [
                {
                    "segment": s.segment,
                    "segment_status": s.segment_status,
                    "kill_switch_enabled": s.kill_switch_enabled,
                    "trading_blocked": s.trading_blocked,
                }
                for s in self.segments
            ],
        }


class UpstoxRisk:
    """Kill-switch control and static-IP registration."""

    def __init__(self, client: UpstoxHttpClient, settings: Optional[Settings] = None) -> None:
        self.client = client
        self.settings = settings or get_settings()
        config = get_config_store().load("broker")
        self._ks_cfg: Dict[str, Any] = config.get("kill_switch", {}) or {}
        self._ip_cfg: Dict[str, Any] = config.get("static_ip", {}) or {}
        self._last_result: Optional[KillSwitchResult] = None

    # -------------------------------------------------------------- kill switch
    @property
    def default_segments(self) -> List[str]:
        configured = self._ks_cfg.get("segments") or ["NSE_EQ"]
        return [s for s in configured if s in VALID_SEGMENTS] or ["NSE_EQ"]

    def engage(self, segments: Optional[Sequence[str]] = None, reason: str = "manual") -> KillSwitchResult:
        """DISABLE the given segments at the broker (blocks new orders, cancels open ones).

        Documented prerequisites are checked *locally* first so we do not send a
        request we know will fail (and to give the user an actionable message).
        """
        targets = [s for s in (segments or self.default_segments) if s in VALID_SEGMENTS]
        if not targets:
            return KillSwitchResult(ok=False, message="no valid segments supplied")

        warnings = self.check_preconditions(targets)
        if warnings:
            log.error("kill switch preconditions not met", context={"warnings": warnings})
            return KillSwitchResult(ok=False, message="; ".join(warnings))

        return self._update_segments(targets, "DISABLE", reason=reason)

    def release(self, segments: Optional[Sequence[str]] = None, reason: str = "manual") -> KillSwitchResult:
        """ENABLE the given segments again.

        The documented 12-hour cooling period means this may legitimately fail;
        we surface that as an informational result rather than an error.
        """
        targets = [s for s in (segments or self.default_segments) if s in VALID_SEGMENTS]
        result = self._update_segments(targets, "ENABLE", reason=reason)
        if not result.ok and "cooling" in result.message.lower():
            log.warning("kill switch cooling period still active", context={"message": result.message})
        return result

    def check_preconditions(self, segments: Sequence[str]) -> List[str]:
        """Return a list of reasons why DISABLE would be rejected."""
        problems: List[str] = []
        if self._ks_cfg.get("requires_all_positions_closed", True):
            try:
                from .upstox_positions import UpstoxPositions

                open_positions = [
                    p for p in UpstoxPositions(self.client, self.settings).positions() if p.quantity != 0
                ]
                if open_positions:
                    problems.append(
                        "all open positions must be closed before disabling a segment "
                        f"({len(open_positions)} still open)"
                    )
            except UpstoxError as exc:
                problems.append(f"could not verify open positions: {exc}")
        return problems

    def _update_segments(self, segments: Sequence[str], action: str, reason: str = "manual") -> KillSwitchResult:
        if action not in VALID_ACTIONS:
            raise ValueError(f"invalid action: {action}")
        payload = [{"segment": s, "action": action} for s in segments]
        log.warning(
            "calling Upstox kill switch",
            context={"action": action, "segments": list(segments), "reason": reason},
        )
        try:
            response = self.client.post(Endpoints.KILL_SWITCH, json_body=payload, limit_class="standard")
        except UpstoxError as exc:
            message = str(exc)
            result = KillSwitchResult(
                ok=False,
                message=message,
                segments=[],
                token_regeneration_required=False,
            )
            self._last_result = result
            return result

        data = response.data
        rows = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
        statuses = [
            SegmentStatus(
                segment=str(row.get("segment", "")),
                segment_status=str(row.get("segment_status", "UNKNOWN")),
                kill_switch_enabled=bool(row.get("kill_switch_enabled", False)),
            )
            for row in rows
        ]
        result = KillSwitchResult(
            ok=True,
            segments=statuses,
            message=f"kill switch {action.lower()}d for {', '.join(segments)}",
            token_regeneration_required=bool(self._ks_cfg.get("requires_token_regeneration", True)),
            raw=response.payload if isinstance(response.payload, dict) else {},
        )
        self._last_result = result
        log.warning(
            "kill switch state changed",
            context={"action": action, "segments": segments, "result": result.message},
        )
        return result

    @property
    def last_result(self) -> Optional[KillSwitchResult]:
        return self._last_result

    # ---------------------------------------------------------------- static IP
    def get_registered_ips(self) -> Dict[str, Any]:
        """The registered IPs come from configuration; Upstox exposes no GET for them.

        ``PUT /v2/user/ip`` returns the authoritative values after an update, so
        the dashboard shows the configured values and marks them unverified until
        an update (or a successful live order) confirms them.
        """
        return {
            "primary_ip": self.settings.upstox_static_ip_primary or None,
            "secondary_ip": self.settings.upstox_static_ip_secondary or None,
            "configured": bool(self.settings.upstox_static_ip_primary),
            "verified": False,
            "note": "Upstox does not expose a read endpoint for registered IPs; "
                    "values are taken from configuration and confirmed on update.",
        }

    def update_static_ips(self, primary_ip: str, secondary_ip: Optional[str] = None) -> Dict[str, Any]:
        """Register a primary (and optional secondary) static IP for the account.

        WARNING: this can only be done once per calendar week and it invalidates
        existing access tokens, so the caller must re-run the OAuth login
        afterwards. The dashboard requires explicit confirmation.
        """
        errors = _validate_ip(primary_ip, secondary_ip)
        if errors:
            return {"ok": False, "error": "; ".join(errors)}

        payload: Dict[str, Any] = {"primary_ip": primary_ip}
        if secondary_ip:
            payload["secondary_ip"] = secondary_ip
        try:
            response = self.client.put(Endpoints.STATIC_IPS, json_body=payload, limit_class="standard")
        except UpstoxError as exc:
            return {"ok": False, "error": str(exc)}

        data = response.data if isinstance(response.data, dict) else {}
        invalidated = bool(data.get("access_tokens_invalidated"))
        if invalidated:
            log.warning("static IP updated - existing access tokens were invalidated")
            self.client.clear_access_token()
        return {
            "ok": True,
            "primary_ip": data.get("primary_ip"),
            "secondary_ip": data.get("secondary_ip"),
            "primary_ip_updated_at": data.get("primary_ip_updated_at"),
            "secondary_ip_updated_at": data.get("secondary_ip_updated_at"),
            "access_tokens_invalidated": invalidated,
            "next_action": "Re-run the Upstox login to obtain a fresh access token." if invalidated else None,
        }

    # ------------------------------------------------------------ static IP check
    def verify_outbound_ip(self, expected_ip: Optional[str] = None) -> Dict[str, Any]:
        """Compare this host's public IP with the configured/registered IP.

        A mismatch is a hard blocker for LIVE order placement, so the preflight
        check reports it explicitly.
        """
        expected = (expected_ip or self.settings.upstox_static_ip_primary or "").strip()
        actual = self._discover_public_ip()
        if not expected:
            return {"ok": False, "reason": "no_static_ip_configured", "actual_ip": actual, "expected_ip": None}
        if not actual:
            return {"ok": False, "reason": "could_not_determine_public_ip", "actual_ip": None, "expected_ip": expected}
        match = actual == expected
        return {
            "ok": match,
            "reason": "match" if match else "ip_mismatch",
            "actual_ip": actual,
            "expected_ip": expected,
        }

    @staticmethod
    def _discover_public_ip() -> Optional[str]:
        import httpx

        for url in ("https://api.ipify.org", "https://ifconfig.me/ip", "https://icanhazip.com"):
            try:
                response = httpx.get(url, timeout=5.0, follow_redirects=True)
                if response.status_code == 200:
                    text = response.text.strip()
                    ipaddress.ip_address(text)
                    return text
            except Exception:
                continue
        return None


def _validate_ip(primary: str, secondary: Optional[str]) -> List[str]:
    errors: List[str] = []
    try:
        ipaddress.ip_address((primary or "").strip())
    except ValueError:
        errors.append("primary_ip must be a valid IPv4 or IPv6 address")
    if secondary:
        try:
            ipaddress.ip_address(secondary.strip())
        except ValueError:
            errors.append("secondary_ip must be a valid IPv4 or IPv6 address")
        else:
            if secondary.strip() == (primary or "").strip():
                errors.append("primary_ip and secondary_ip must be different")
    return errors


__all__ = ["UpstoxRisk", "KillSwitchResult", "SegmentStatus", "VALID_SEGMENTS", "VALID_ACTIONS"]
