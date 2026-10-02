import { useState } from 'react'
import api, { SystemStatus } from '../services/api'
import { Card, ErrorBox, Pill, money, num, pct, useAsync } from '../components/common'

export default function Settings({ status, refresh }: { status: SystemStatus | null; refresh: () => void }) {
  const broker = useAsync(() => api.brokerStatus(), [])
  const preflight = useAsync(() => api.preflight(true), [])
  const config = useAsync(() => api.config(), [])
  const news = useAsync(() => api.newsContext(), [])
  const slippage = useAsync(() => api.slippage(), [])
  const [msg, setMsg] = useState('')
  const [busy, setBusy] = useState('')
  const [token, setToken] = useState('')

  const connect = async () => {
    setBusy('connect')
    setMsg('')
    try {
      const info = await api.loginUrl()
      window.open(info.url, '_blank', 'noopener,noreferrer')
      setMsg('Opened the Upstox login page in a new tab. After authorising, return here — the token is stored server-side only.')
    } catch (e: any) {
      setMsg('Could not start the login: ' + (e?.message || e))
    } finally {
      setBusy('')
    }
  }

  const attach = async () => {
    setBusy('attach')
    setMsg('')
    try {
      const result = await api.attachToken(token.trim())
      setMsg(result.ok ? 'Token verified and attached.' : 'Token attached but verification failed — check the token scope.')
      setToken('')
      broker.reload()
      preflight.reload()
      refresh()
    } catch (e: any) {
      setMsg('Failed: ' + (e?.message || e))
    } finally {
      setBusy('')
    }
  }

  const logout = async () => {
    if (!window.confirm('Log out of Upstox and clear the stored token?')) return
    await api.logoutBroker()
    broker.reload()
    refresh()
  }

  const runPreflight = async () => {
    setBusy('preflight')
    await preflight.reload()
    setBusy('')
  }

  return (
    <div>
      <Card title="Upstox connection">
        {broker.data && (
          <>
            <div className="kv" style={{ maxWidth: 640 }}>
              <div>Operating mode</div>
              <div>
                <Pill value={broker.data.mode} />
              </div>
              <div>Authenticated</div>
              <div>{broker.data.auth?.authenticated ? 'yes' : 'no'}</div>
              <div>Token source</div>
              <div>{broker.data.auth?.token_source || '—'}</div>
              <div>Token valid today</div>
              <div>{broker.data.auth?.token_probably_valid_today ? 'yes' : 'no'}</div>
              <div>API key configured</div>
              <div>{broker.data.auth?.api_key_configured ? 'yes' : 'no'}</div>
              <div>Instrument master loaded</div>
              <div>{broker.data.instrument_master?.loaded ? `${broker.data.instrument_master.count} instruments` : 'no'}</div>
              <div>Static IP registered</div>
              <div>{broker.data.static_ip?.configured ? broker.data.static_ip.primary_ip : 'not configured'}</div>
            </div>
            <div style={{ display: 'flex', gap: 10, marginTop: 12, flexWrap: 'wrap' }}>
              <button onClick={connect} disabled={busy === 'connect'}>
                Connect Upstox (OAuth login)
              </button>
              <button className="secondary" onClick={logout}>
                Log out of Upstox
              </button>
            </div>
            <div className="field" style={{ marginTop: 14, maxWidth: 620 }}>
              <label>Or paste a daily access token (it is stored server-side and never returned to the browser)</label>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  type="password"
                  value={token}
                  placeholder="UPSTOX_ACCESS_TOKEN"
                  onChange={(e) => setToken(e.target.value)}
                />
                <button onClick={attach} disabled={!token.trim() || busy === 'attach'}>
                  Attach
                </button>
              </div>
            </div>
          </>
        )}
        {msg && <div className="note">{msg}</div>}
      </Card>

      <Card
        title="LIVE-mode compliance preflight"
        actions={
          <button onClick={runPreflight} disabled={busy === 'preflight'}>
            Re-run checks
          </button>
        }
      >
        {preflight.data && (
          <>
            <div style={{ marginBottom: 10 }}>
              Result: {preflight.data.passed ? <Pill value="PASS" /> : <Pill value="FAIL" />}{' '}
              <span className="muted">
                ran {preflight.data.age_seconds}s ago · deep checks {String(preflight.data.ran_deep)}
              </span>
            </div>
            <div className="scroll-x">
              <table>
                <thead>
                  <tr>
                    <th>Check</th>
                    <th>Status</th>
                    <th>Detail</th>
                    <th>Remediation</th>
                  </tr>
                </thead>
                <tbody>
                  {preflight.data.checks.map((c: any) => (
                    <tr key={c.name}>
                      <td>{c.name.replace(/_/g, ' ')}</td>
                      <td>
                        <Pill value={c.status} />
                      </td>
                      <td className="muted">{(c.detail || '').slice(0, 90)}</td>
                      <td className="muted">{c.status === 'FAIL' ? (c.remediation || '').slice(0, 110) : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="note warn">
              No live order is possible while any required check fails. The system fails closed.
            </div>
            {status && (
              <>
                <h3>Why LIVE is currently blocked</h3>
                <ul className="reasons">
                  {(status.mode.live_blocked_reasons || []).map((r: string, i: number) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </>
            )}
          </>
        )}
      </Card>

      <div className="grid c2">
        <Card title="Risk limits (enforced by the risk engine)">
          {status && (
            <>
              <div className="kv">
                <div>Risk per trade</div>
                <div>{pct(status.risk.risk_per_trade_pct, 2)}</div>
                <div>Hard ceiling per trade</div>
                <div>{pct(status.risk.max_risk_per_trade_pct, 2)}</div>
                <div>Maximum simultaneous positions</div>
                <div>{status.risk.max_simultaneous_positions}</div>
                <div>Maximum correlated-sector positions</div>
                <div>{status.risk.max_correlated_sector_positions}</div>
                <div>Maximum total open risk</div>
                <div>{pct(status.risk.max_total_open_risk_pct, 2)}</div>
                <div>Soft daily stop</div>
                <div className="warn">{pct(status.risk.soft_daily_stop_pct, 1)}</div>
                <div>Hard daily stop</div>
                <div className="bad">{pct(status.risk.hard_daily_stop_pct, 1)}</div>
                <div>Weekly drawdown warning</div>
                <div className="warn">{pct(status.risk.weekly_drawdown_warning_pct, 1)}</div>
                <div>Maximum consecutive losses</div>
                <div>{status.risk.max_consecutive_losses}</div>
                <div>Maximum trades per day</div>
                <div>{status.risk.max_trades_per_day}</div>
                <div>Strategy suspension drawdown</div>
                <div>{pct(status.risk.strategy_suspension_drawdown_pct, 1)}</div>
                <div>Averaging down</div>
                <div className="good">forbidden</div>
                <div>Martingale / doubling</div>
                <div className="good">forbidden</div>
                <div>Stop required before entry</div>
                <div className="good">yes</div>
              </div>
              <div className="note">
                These limits live in <code>config/risk.yaml</code> and have absolute authority. They cannot be relaxed
                through the API — only tightened. No AI component can override them.
              </div>
            </>
          )}
        </Card>

        <Card title="Kill switch">
          {status && (
            <>
              <div style={{ marginBottom: 10 }}>
                Status: {status.kill_switch.engaged ? <Pill value="FAIL" /> : <Pill value="PASS" />}{' '}
                <span className="muted">{status.kill_switch.reason || 'armed and released'}</span>
              </div>
              <div style={{ display: 'flex', gap: 10 }}>
                <button
                  className="danger"
                  onClick={async () => {
                    await api.engageKillSwitch('Engaged from Settings page')
                    refresh()
                  }}
                >
                  ⛔ STOP TRADING
                </button>
                <button
                  className="secondary"
                  onClick={async () => {
                    try {
                      await api.releaseKillSwitch()
                      refresh()
                    } catch (e: any) {
                      window.alert('Could not release: ' + (e?.message || e))
                    }
                  }}
                >
                  ▶ Resume trading
                </button>
              </div>
              <div className="note">
                The internal switch is instant and local, and works even when the broker is unreachable. In LIVE mode an
                additional account-level Upstox kill switch can be engaged (it cancels open orders and applies a 12-hour
                cooling period before it can be re-enabled).
              </div>
              <h3 style={{ marginTop: 12 }}>Slippage monitor</h3>
              {slippage.data && (
                <div className="kv">
                  <div>Median slippage</div>
                  <div>{num(slippage.data.summary.median_bps, 1)} bps</div>
                  <div>Status</div>
                  <div>
                    <Pill value={slippage.data.summary.level} />
                  </div>
                  <div>Total slippage cost</div>
                  <div>{money(slippage.data.summary.total_cost)}</div>
                </div>
              )}
            </>
          )}
        </Card>
      </div>

      <Card title="News context (Upstox News API)">
        {news.data && (
          <>
            {!news.data.available ? (
              <div className="note warn">{news.data.reason}</div>
            ) : (
              <>
                <div className="kv" style={{ maxWidth: 520 }}>
                  <div>Instruments queried</div>
                  <div>{news.data.instruments_queried}</div>
                  <div>Overall sentiment</div>
                  <div>{news.data.market_context?.sentiment} ({num(news.data.market_context?.score, 3)})</div>
                  <div>Articles</div>
                  <div>{news.data.market_context?.articles}</div>
                </div>
                <div className="scroll-x dd" style={{ marginTop: 10 }}>
                  <table>
                    <thead>
                      <tr><th>Published</th><th>Headline</th><th>Sentiment</th><th>Event</th></tr>
                    </thead>
                    <tbody>
                      {(news.data.items || []).slice(0, 25).map((item: any, i: number) => (
                        <tr key={i}>
                          <td className="muted">{item.published_time?.slice(0, 16).replace('T', ' ')}</td>
                          <td style={{ whiteSpace: 'normal', maxWidth: 480 }}>
                            <a href={item.article_link} target="_blank" rel="noopener noreferrer">
                              {item.heading}
                            </a>
                          </td>
                          <td>
                            <Pill value={item.classification?.sentiment === 'positive' ? 'PASS' : item.classification?.sentiment === 'negative' ? 'FAIL' : 'WARN'} />{' '}
                            <span className="muted">{item.classification?.sentiment}</span>
                          </td>
                          <td className="muted">{item.classification?.event_type}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
            <div className="note">{news.data.market_context?.note}</div>
          </>
        )}
      </Card>

      <Card title="Configuration (secrets redacted)">
        {config.data && (
          <>
            <div className="note">
              Settings come from environment variables and <code>config/*.yaml</code>. Secrets are never returned to the
              browser — only whether they are set.
            </div>
            <pre>{JSON.stringify(config.data.settings, null, 2)}</pre>
          </>
        )}
      </Card>
    </div>
  )
}
