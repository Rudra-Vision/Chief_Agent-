"""FastAPI application factory.

Serves:
    /api/*      the JSON API
    /health     the health endpoint the brief requires
    /*          the React dashboard (built assets from ``frontend/dist``)

Security: broker tokens stay server-side; state-changing dashboard requests
require a session (and a CSRF token when auth is enabled).
"""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..data.db import init_db
from ..logging_setup import configure_logging, get_logger
from ..settings import FRONTEND_DIST, VAR_DIR, get_settings
from ..timeutil import now_ist
from .security import get_auth
from .state import AppState, get_app_state

log = get_logger(__name__, component="api")

#: Requests that change state and therefore need CSRF protection.
STATEFUL_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def create_app(state: Optional[AppState] = None, *, start_services: bool = True) -> FastAPI:
    configure_logging()
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db()
        app.state.chief = state or get_app_state()
        if start_services:
            try:
                from ..scheduling.jobs import build_scheduler

                scheduler = build_scheduler(app.state.chief)
                scheduler.start()
                app.state.chief.attach_scheduler(scheduler)
                log.info("scheduler started")
            except Exception as exc:
                log.warning("scheduler could not be started", context={"error": str(exc)})
        mode = app.state.chief.mode_summary()
        log.warning(
            "Chief Agent API starting",
            context={
                "mode": mode["mode"],
                "banner": mode["banner"],
                "data_source": mode["data_source"],
                "live_permitted": mode["live_permitted"],
            },
        )
        yield
        scheduler = getattr(app.state.chief, "scheduler", None)
        if scheduler is not None:
            try:
                scheduler.shutdown()
            except Exception:  # pragma: no cover
                pass

    app = FastAPI(
        title="Chief Agent",
        description=(
            "Self-improving algorithmic trading system for the Indian market (Upstox). "
            "Trading decisions are deterministic; the research layer proposes and a "
            "deterministic validation engine decides."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-csrf-token"],
    )

    # ------------------------------------------------------------------ auth
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        path = request.url.path
        auth = get_auth()

        # Public endpoints and static assets pass straight through.
        if auth.is_public(path) or not path.startswith("/api"):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
            response.headers["Referrer-Policy"] = "same-origin"
            return response

        session = auth.current_session(request)
        if auth.auth_required and session is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "not authenticated", "login_required": True},
            )
        if request.method in STATEFUL_METHODS and auth.auth_required:
            try:
                auth.require_csrf(request, session or {})
            except Exception as exc:
                detail = getattr(exc, "detail", "invalid CSRF token")
                return JSONResponse(status_code=403, content={"detail": detail})

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        return response

    # --------------------------------------------------------------- routers
    from .routers import backtest, broker, data, journal, news, research, system, trading

    app.include_router(system.router, prefix="/api", tags=["system"])
    app.include_router(broker.router, prefix="/api", tags=["broker"])
    app.include_router(data.router, prefix="/api", tags=["data"])
    app.include_router(trading.router, prefix="/api", tags=["trading"])
    app.include_router(backtest.router, prefix="/api", tags=["backtest"])
    app.include_router(research.router, prefix="/api", tags=["research"])
    app.include_router(journal.router, prefix="/api", tags=["journal"])
    app.include_router(news.router, prefix="/api", tags=["news"])

    # ------------------------------------------------------------ health root
    @app.get("/health", tags=["system"], summary="System health (root)")
    def root_health() -> Dict[str, Any]:
        """The health endpoint, at the path the brief specifies."""
        try:
            from .state import get_app_state as _get_state

            return _get_state().health_report()
        except Exception as exc:  # pragma: no cover - never fail a health probe
            return {"status": "STOPPED", "error": str(exc)}

    @app.get("/api", include_in_schema=False)
    def api_root() -> Dict[str, Any]:
        return {"name": "Chief Agent", "version": __version__, "docs": "/docs"}

    # ---------------------------------------------------------------- frontend
    dist = Path(FRONTEND_DIST)
    if (dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str = "") -> Any:
        """Serve the built dashboard, falling back to a built-in page.

        The API lives under ``/api``, so any other path is a dashboard route.
        """
        if path.startswith("api/"):
            return JSONResponse(status_code=404, content={"detail": "unknown API route"})
        if path in ("health", "healthz", "api/health"):
            try:
                from .state import get_app_state as _get_state

                return _get_state().health_report()
            except Exception as exc:  # pragma: no cover
                return JSONResponse(status_code=503, content={"status": "STOPPED", "error": str(exc)})
        index = dist / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return HTMLResponse(_fallback_page(settings))

    return app


def _fallback_page(settings: Any) -> str:
    """A minimal, dependency-free dashboard used when the React build is absent.

    It is deliberately self-contained so the system is usable straight after a
    ``pip install`` with no Node toolchain.
    """
    dist = Path(FRONTEND_DIST)
    if (dist / "index.html").exists():
        try:
            return (dist / "index.html").read_text(encoding="utf-8")
        except OSError:
            pass
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Chief Agent</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0b1220;color:#e6edf7}
 header{padding:16px 24px;background:#111a2e;border-bottom:1px solid #1f2b45;display:flex;justify-content:space-between;align-items:center}
 h1{font-size:18px;margin:0}
 main{padding:24px;max-width:1200px;margin:0 auto}
 .card{background:#121c30;border:1px solid #1f2b45;border-radius:10px;padding:16px;margin-bottom:16px}
 .row{display:flex;gap:16px;flex-wrap:wrap}
 .col{flex:1;min-width:240px}
 .k{color:#8ea3c4;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
 .v{font-size:22px;font-weight:600;margin-top:4px}
 .good{color:#4ade80}.bad{color:#f87171}.warn{color:#fbbf24}
 button{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:10px 16px;font-weight:600;cursor:pointer}
 button.stop{background:#dc2626;font-size:16px;padding:14px 22px}
 table{width:100%;border-collapse:collapse;font-size:14px}
 th,td{text-align:left;padding:8px;border-bottom:1px solid #1f2b45}
 code{background:#0b1220;padding:2px 6px;border-radius:4px}
 .banner{padding:10px 24px;font-weight:600;background:#1d4ed8;color:#fff}
 .banner.live{background:#b91c1c}
 a{color:#7dd3fc}
</style></head>
<body>
<div class="banner" id="banner">Loading…</div>
<header><h1>Chief Agent</h1>
 <div><button id="stop" class="stop">STOP TRADING</button></div></header>
<main>
 <div class="card"><div class="k">Build the dashboard</div>
  <p>The React dashboard has not been built yet. From the repository root run:</p>
  <p><code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code></p>
  <p>This page still works as a live status view. The full JSON API is at
   <a href="/docs">/docs</a> and <a href="/api/system/status">/api/system/status</a>.</p>
 </div>
 <div class="card"><div class="k">Status</div><div id="status">loading…</div></div>
 <div class="card"><div class="k">Health</div><pre id="health" style="white-space:pre-wrap"></pre></div>
 <div class="card"><div class="k">Top opportunities</div><div id="opps">loading…</div></div>
</main>
<script>
const fmt = (v) => typeof v === 'number' ? v.toLocaleString(undefined,{maximumFractionDigits:2}) : (v ?? '—');
async function j(url, opts){const r=await fetch(url,opts); if(!r.ok) throw new Error(await r.text()); return r.json();}
async function refresh(){
  try{
    const s = await j('/api/system/status');
    document.getElementById('banner').textContent = s.mode.banner + '  •  data: ' + s.data.source + '  •  strategy ' + (s.strategy.version||'—');
    document.getElementById('banner').className = 'banner' + (s.mode.is_live?' live':'');
    const p = s.portfolio;
    document.getElementById('status').innerHTML = `<div class="row">
      <div class="col"><div class="k">Mode</div><div class="v">${s.mode.mode}</div></div>
      <div class="col"><div class="k">Equity</div><div class="v">₹${fmt(p.equity)}</div></div>
      <div class="col"><div class="k">Today's P&amp;L</div><div class="v ${p.realized_pnl_today>=0?'good':'bad'}">₹${fmt(p.realized_pnl_today)}</div></div>
      <div class="col"><div class="k">Open positions</div><div class="v">${p.open_position_count}</div></div>
      <div class="col"><div class="k">Drawdown</div><div class="v ${p.drawdown_pct<0?'warn':''}">${(p.drawdown_pct*100).toFixed(2)}%</div></div>
      <div class="col"><div class="k">Kill switch</div><div class="v ${s.kill_switch.engaged?'bad':'good'}">${s.kill_switch.engaged?'ENGAGED':'released'}</div></div>
    </div>`;
    const h = await j('/api/system/health');
    document.getElementById('health').textContent = h.status + '\\n' + h.checks.map(c=>` - ${c.name}: ${c.status} ${c.detail||''}`).join('\\n');
  }catch(e){ document.getElementById('status').textContent = 'error: '+e.message; }
  try{
    const o = await j('/api/trading/opportunities?top_n=10&refresh=false');
    const rows = (o.longs||[]).concat(o.shorts||[]).map(x=>`<tr><td>${x.rank}</td><td>${x.symbol}</td><td>${x.direction}</td><td>${x.score}</td><td>₹${fmt(x.entry_price)}</td><td>₹${fmt(x.stop_price)}</td><td>₹${fmt(x.target_1)}</td><td>${x.risk_reward}</td></tr>`).join('');
    document.getElementById('opps').innerHTML = rows ? `<table><tr><th>#</th><th>Symbol</th><th>Side</th><th>Score</th><th>Entry</th><th>Stop</th><th>Target</th><th>R:R</th></tr>${rows}</table>` : (o.error||'no opportunities yet — run a data download first');
  }catch(e){ document.getElementById('opps').textContent = 'error: '+e.message; }
}
document.getElementById('stop').onclick = async () => {
  if(!confirm('Stop all trading immediately?')) return;
  const status = await j('/api/auth/status');
  const headers = {'Content-Type':'application/json'};
  if(status.csrf_token) headers['X-CSRF-Token'] = status.csrf_token;
  await j('/api/broker/kill-switch/engage',{method:'POST',headers,body:JSON.stringify({reason:'dashboard STOP button'})});
  refresh();
};
refresh(); setInterval(refresh, 15000);
</script></body></html>"""


app = None  # populated by uvicorn factory below


def get_app() -> FastAPI:
    """uvicorn factory entry point (``uvicorn chief_agent.api.app:get_app --factory``)."""
    return create_app()


__all__ = ["create_app", "get_app"]
