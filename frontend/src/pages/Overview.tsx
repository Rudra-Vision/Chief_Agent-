import { useEffect, useState } from 'react'
import api, { SystemStatus } from '../services/api'
import { Card, ErrorBox, Pill, SimulatedBanner, Stat, money, num, pct, useAsync } from '../components/common'

export default function Overview({ status, refresh }: { status: SystemStatus | null; refresh: () => void }) {
  const health = useAsync(() => api.health(), [])
  const notifications = useAsync(() => api.notifications(25), [])
  const [busy, setBusy] = useState('')

  useEffect(() => {
    const timer = setInterval(() => {
      health.reload()
      notifications.reload()
    }, 30000)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (!status) return <div className="spinner">Loading system status…</div>

  const p = status.portfolio

  const runCycle = async () => {
    setBusy('cycle')
    try {
      const result = await api.runCycle()
      window.alert(
        `Cycle complete.\n\nScanned: ${result.scanned ?? 0} instruments\nOpportunities: ${result.opportunities ?? 0}\n` +
          `Orders filled: ${(result.executed || []).length}\nRejected by risk: ${(result.rejected || []).length}`,
      )
      refresh()
    } catch (e: any) {
      window.alert('Cycle failed: ' + (e?.message || e))
    } finally {
      setBusy('')
    }
  }

  return (
    <div>
      <SimulatedBanner isSimulated={status.data.is_simulated} />

      <div className="grid c4" style={{ marginBottom: 16 }}>
        <Card>
          <Stat label="Mode" value={status.mode.mode} sub={status.mode.is_live ? 'REAL MONEY' : 'no real money at risk'} tone={status.mode.is_live ? 'bad' : ''} />
        </Card>
        <Card>
          <Stat label="Account equity" value={money(p.equity)} sub={`peak ${money(p.peak_equity)}`} />
        </Card>
        <Card>
          <Stat label="Today's P&L" value={money(p.realized_pnl_today)} sub={`${pct(p.daily_return_pct)} today`} tone={p.realized_pnl_today >= 0 ? 'good' : 'bad'} />
        </Card>
        <Card>
          <Stat label="Unrealised" value={money(p.unrealized_pnl)} sub={`${p.open_position_count} open`} tone={p.unrealized_pnl >= 0 ? 'good' : 'bad'} />
        </Card>
        <Card>
          <Stat label="Drawdown" value={pct(p.drawdown_pct)} sub="from peak equity" tone={p.drawdown_pct < -0.02 ? 'warn' : ''} />
        </Card>
        <Card>
          <Stat label="Daily risk used" value={pct(p.open_risk_pct)} sub={`limit ${pct(status.risk.max_total_open_risk_pct)}`} />
        </Card>
        <Card>
          <Stat label="Exposure" value={pct(p.gross_exposure_pct)} sub="gross, of equity" />
        </Card>
        <Card>
          <Stat label="Kill switch" value={status.kill_switch.engaged ? 'ENGAGED' : 'released'} sub={status.kill_switch.reason || 'armed'} tone={status.kill_switch.engaged ? 'bad' : 'good'} />
        </Card>
      </div>

      <div className="grid c2">
        <Card
          title="System health"
          actions={
            <button className="secondary" onClick={() => health.reload()}>
              Refresh
            </button>
          }
        >
          {health.loading && !health.data && <div className="spinner">Checking…</div>}
          {health.error && <ErrorBox message={health.error} />}
          {health.data && (
            <>
              <div style={{ marginBottom: 10 }}>
                Overall: <Pill value={health.data.status} />
              </div>
              <div className="scroll-x">
                <table>
                  <thead>
                    <tr>
                      <th>Component</th>
                      <th>Status</th>
                      <th>Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {health.data.checks.map((c: any) => (
                      <tr key={c.name}>
                        <td>{c.name.replace(/_/g, ' ')}</td>
                        <td>
                          <Pill value={c.status} />
                        </td>
                        <td className="muted">{(c.detail || '').slice(0, 110)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </Card>

        <Card title="Market & broker">
          <div className="kv">
            <div>Market status</div>
            <div>
              <Pill value={status.market?.exchange_status?.status || 'UNKNOWN'} /> {status.market?.phase}
            </div>
            <div>Trading day</div>
            <div>{status.market?.session?.is_trading_day ? 'yes' : 'no'}</div>
            <div>Upstox connection</div>
            <div>{status.broker.authenticated ? `connected (${status.broker.token_source})` : 'not connected'}</div>
            <div>Token valid today</div>
            <div>{status.broker.token_valid_today ? 'yes' : 'no'}</div>
            <div>Instrument master</div>
            <div>
              {status.broker.instrument_master_loaded ? `${status.broker.instrument_count} instruments` : 'not loaded'}
            </div>
            <div>Static IP registered</div>
            <div>{status.broker.static_ip_configured ? 'yes' : 'no'}</div>
            <div>Market data source</div>
            <div>
              {status.data.source} {status.data.is_simulated && <span className="badge-sim">SIMULATED</span>}
            </div>
            <div>Data safe mode</div>
            <div className={status.data.data_safe_mode ? 'bad' : 'good'}>{status.data.data_safe_mode ? 'ACTIVE' : 'off'}</div>
            <div>Position reconciliation</div>
            <div className={status.reconciliation.pending ? 'bad' : 'good'}>
              {status.reconciliation.pending ? 'MISMATCH — entries blocked' : 'matched'}
            </div>
            <div>Execution slippage (median)</div>
            <div>{num(status.slippage?.median_bps, 1)} bps <Pill value={status.slippage?.level} /></div>
            <div>Scheduler</div>
            <div>{health.data?.checks?.find((c: any) => c.name === 'scheduler')?.detail || '—'}</div>
          </div>
          <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
            <button onClick={runCycle} disabled={busy === 'cycle'}>
              {busy === 'cycle' ? 'Running…' : 'Run trading cycle now'}
            </button>
            <button className="secondary" onClick={refresh}>
              Refresh status
            </button>
          </div>
          <div className="note">
            A cycle scans the market, ranks opportunities, runs each through the risk engine and (in PAPER mode)
            simulates the orders. Nothing here can bypass the risk engine.
          </div>
        </Card>
      </div>

      <div className="grid c2">
        <Card title="Open positions">
          {p.open_positions.length === 0 ? (
            <div className="muted">No open positions.</div>
          ) : (
            <div className="scroll-x">
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th className="num">Qty</th>
                    <th className="num">Entry</th>
                    <th className="num">Stop</th>
                    <th className="num">LTP</th>
                    <th className="num">P&amp;L</th>
                    <th className="num">R</th>
                  </tr>
                </thead>
                <tbody>
                  {p.open_positions.map((o: any) => (
                    <tr key={o.position_id}>
                      <td>{o.symbol}</td>
                      <td>
                        <Pill value={o.direction} />
                      </td>
                      <td className="num">{o.quantity}</td>
                      <td className="num">{num(o.entry_price)}</td>
                      <td className="num">{num(o.current_stop)}</td>
                      <td className="num">{num(o.ltp)}</td>
                      <td className={`num ${o.unrealized_pnl >= 0 ? 'good' : 'bad'}`}>{money(o.unrealized_pnl)}</td>
                      <td className="num">{num(o.r_multiple)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title="Recent notifications" actions={<button className="secondary" onClick={() => notifications.reload()}>Refresh</button>}>
          {notifications.data?.items?.length ? (
            <div className="dd">
              <table>
                <thead>
                  <tr>
                    <th>When</th>
                    <th>Severity</th>
                    <th>Message</th>
                  </tr>
                </thead>
                <tbody>
                  {notifications.data.items.map((n: any, i: number) => (
                    <tr key={i}>
                      <td className="muted">{new Date(n.created_at).toLocaleTimeString()}</td>
                      <td>
                        <Pill value={n.severity} />
                      </td>
                      <td>
                        {n.title}
                        {n.body ? <div className="muted" style={{ fontSize: 12 }}>{n.body}</div> : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="muted">Nothing yet.</div>
          )}
        </Card>
      </div>
    </div>
  )
}
