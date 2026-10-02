import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import api, { setCsrfToken, SystemStatus } from './services/api'
import Overview from './pages/Overview'
import Opportunities from './pages/Opportunities'
import Positions from './pages/Positions'
const Backtest = lazy(() => import('./pages/Backtest'))
import Research from './pages/Research'
import Journal from './pages/Journal'
import DataPage from './pages/Data'
import Settings from './pages/Settings'

function fmtMoney(value: number | undefined | null, currency = '₹') {
  if (value === undefined || value === null || Number.isNaN(value)) return '—'
  return (
    currency +
    Math.abs(value).toLocaleString('en-IN', { maximumFractionDigits: 0, minimumFractionDigits: 0 }) +
    (value < 0 ? '' : '')
  )
}

function ModeBanner({ status }: { status: SystemStatus | null }) {
  if (!status) return <div className="banner">Connecting to Chief Agent…</div>
  const live = status.mode.is_live
  return (
    <div className={`banner ${live ? 'live' : ''}`}>
      <span>{live ? '🔴 LIVE TRADING — REAL MONEY IS AT RISK' : '🧪 SIMULATION — no real money is at risk'}</span>
      <span className="tag">MODE: {status.mode.mode}</span>
      <span className="tag">STAGE: {status.mode.deployment_stage.replace('STAGE_', 'stage ')}</span>
      <span className="tag">
        DATA: {status.data.source}
        {status.data.is_simulated ? ' (simulated)' : ''}
      </span>
      <span className="tag">STRATEGY: {status.strategy.version ?? '—'}</span>
      {status.kill_switch.engaged && <span className="tag">⛔ KILL SWITCH ENGAGED</span>}
      {status.data.data_safe_mode && <span className="tag">⚠ DATA SAFE MODE</span>}
    </div>
  )
}

export default function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [error, setError] = useState<string>('')
  const [authState, setAuthState] = useState<any>(null)
  const [stopping, setStopping] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const [s, auth] = await Promise.all([api.status(), api.authStatus()])
      setStatus(s)
      setAuthState(auth)
      setCsrfToken(auth?.csrf_token ?? null)
      setError('')
    } catch (e: any) {
      if (e?.status === 401) {
        setAuthState({ auth_required: true, authenticated: false })
      } else {
        setError(e?.message || String(e))
      }
    }
  }, [])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, 15000)
    return () => clearInterval(timer)
  }, [refresh])

  const stopTrading = useCallback(async () => {
    if (!window.confirm('STOP ALL TRADING immediately?\n\nThis blocks every new order right away.')) return
    setStopping(true)
    try {
      await api.engageKillSwitch('STOP button pressed on the dashboard')
      await refresh()
    } catch (e: any) {
      window.alert('Could not engage the kill switch: ' + (e?.message || e))
    } finally {
      setStopping(false)
    }
  }, [refresh])

  const resumeTrading = useCallback(async () => {
    if (!window.confirm('Resume trading? New orders will be allowed again.')) return
    try {
      await api.releaseKillSwitch()
      await refresh()
    } catch (e: any) {
      window.alert('Could not release the kill switch: ' + (e?.message || e))
    }
  }, [refresh])

  const needsLogin = useMemo(
    () => Boolean(authState?.auth_required && !authState?.authenticated),
    [authState],
  )

  if (needsLogin) {
    return (
      <div className="login-wrap">
        <div className="card login-box">
          <h2>Chief Agent</h2>
          <p className="muted">This dashboard is password protected.</p>
          <form
            onSubmit={async (e) => {
              e.preventDefault()
              const form = e.target as HTMLFormElement
              const username = (form.elements.namedItem('u') as HTMLInputElement).value
              const password = (form.elements.namedItem('p') as HTMLInputElement).value
              try {
                await api.login(username, password)
                await refresh()
              } catch (err: any) {
                setError(err?.message || 'login failed')
              }
            }}
          >
            <div className="field">
              <label>Username</label>
              <input name="u" defaultValue="owner" autoComplete="username" />
            </div>
            <div className="field">
              <label>Password</label>
              <input name="p" type="password" autoComplete="current-password" />
            </div>
            {error && <div className="error-box">{error}</div>}
            <button type="submit">Log in</button>
          </form>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <ModeBanner status={status} />
      <div className="topbar">
        <h1>
          Chief Agent <span>self-improving trading system · Upstox · Indian equities</span>
        </h1>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <div className="stat" style={{ textAlign: 'right' }}>
            <div className="k">Equity</div>
            <div className="v" style={{ fontSize: 16 }}>
              {fmtMoney(status?.portfolio?.equity)}
            </div>
          </div>
          <div className="stat" style={{ textAlign: 'right' }}>
            <div className="k">Today</div>
            <div className={`v ${(status?.portfolio?.realized_pnl_today ?? 0) >= 0 ? 'good' : 'bad'}`} style={{ fontSize: 16 }}>
              {fmtMoney(status?.portfolio?.realized_pnl_today)}
            </div>
          </div>
          {status?.kill_switch?.engaged ? (
            <button className="huge" onClick={resumeTrading}>
              ▶ RESUME TRADING
            </button>
          ) : (
            <button className="huge danger" onClick={stopTrading} disabled={stopping}>
              {stopping ? 'STOPPING…' : '⛔ STOP TRADING'}
            </button>
          )}
        </div>
      </div>

      <div className="layout">
        <nav className="sidebar">
          <NavLink to="/overview" className={({ isActive }) => (isActive ? 'active' : '')}>
            Overview
          </NavLink>
          <NavLink to="/opportunities" className={({ isActive }) => (isActive ? 'active' : '')}>
            Opportunities
          </NavLink>
          <NavLink to="/positions" className={({ isActive }) => (isActive ? 'active' : '')}>
            Positions
          </NavLink>
          <NavLink to="/backtest" className={({ isActive }) => (isActive ? 'active' : '')}>
            Backtest
          </NavLink>
          <NavLink to="/research" className={({ isActive }) => (isActive ? 'active' : '')}>
            Self-improvement
          </NavLink>
          <NavLink to="/journal" className={({ isActive }) => (isActive ? 'active' : '')}>
            Trade journal
          </NavLink>
          <NavLink to="/data" className={({ isActive }) => (isActive ? 'active' : '')}>
            Data &amp; universe
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => (isActive ? 'active' : '')}>
            Settings &amp; safety
          </NavLink>
        </nav>

        <main className="content">
          {error && <div className="error-box">{error}</div>}
          <Suspense fallback={<div className="spinner">Loading…</div>}>
          <Routes>
            <Route path="/" element={<Navigate to="/overview" replace />} />
            <Route path="/overview" element={<Overview status={status} refresh={refresh} />} />
            <Route path="/opportunities" element={<Opportunities />} />
            <Route path="/positions" element={<Positions />} />
            <Route path="/backtest" element={<Backtest />} />
            <Route path="/research" element={<Research />} />
            <Route path="/journal" element={<Journal />} />
            <Route path="/data" element={<DataPage />} />
            <Route path="/settings" element={<Settings status={status} refresh={refresh} />} />
            <Route path="*" element={<Navigate to="/overview" replace />} />
          </Routes>
          </Suspense>
        </main>
      </div>
    </div>
  )
}
