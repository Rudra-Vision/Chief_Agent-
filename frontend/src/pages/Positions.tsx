import { useState } from 'react'
import api from '../services/api'
import { Card, ErrorBox, Pill, SimulatedBanner, Stat, money, num, pct, useAsync } from '../components/common'

export default function Positions() {
  const { data, loading, error, reload } = useAsync(() => api.positions(), [])
  const orders = useAsync(() => api.orders(), [])
  const slippage = useAsync(() => api.slippage(), [])
  const [busy, setBusy] = useState('')

  const closeOne = async (id: string) => {
    setBusy(id)
    try {
      await api.closePosition(id)
      await reload()
    } finally {
      setBusy('')
    }
  }

  const closeAll = async () => {
    if (!window.confirm('Close every open paper position at the current price?')) return
    setBusy('all')
    try {
      const result = await api.closeAll()
      window.alert(`Closed ${result.closed} position(s).`)
      await reload()
    } finally {
      setBusy('')
    }
  }

  if (loading && !data) return <div className="spinner">Loading positions…</div>
  if (error) return <ErrorBox message={error} />
  if (!data) return null

  return (
    <div>
      <SimulatedBanner isSimulated={Boolean(data.is_simulated)} />

      <div className="grid c4" style={{ marginBottom: 16 }}>
        <Card>
          <Stat label="Equity" value={money(data.equity)} />
        </Card>
        <Card>
          <Stat label="Cash" value={money(data.cash)} />
        </Card>
        <Card>
          <Stat label="Open P&L" value={money(data.unrealized_pnl)} tone={data.unrealized_pnl >= 0 ? 'good' : 'bad'} />
        </Card>
        <Card>
          <Stat label="Realised today" value={money(data.realized_pnl_today)} tone={data.realized_pnl_today >= 0 ? 'good' : 'bad'} />
        </Card>
        <Card>
          <Stat label="Open risk" value={pct(data.open_risk_pct)} sub="of equity" />
        </Card>
        <Card>
          <Stat label="Gross exposure" value={pct(data.gross_exposure_pct)} />
        </Card>
        <Card>
          <Stat label="Drawdown" value={pct(data.drawdown_pct)} tone={data.drawdown_pct < -0.02 ? 'warn' : ''} />
        </Card>
        <Card>
          <Stat label="Consecutive losses" value={data.consecutive_losses} />
        </Card>
      </div>

      <Card
        title="Open positions"
        actions={
          <button className="danger" onClick={closeAll} disabled={busy === 'all' || data.positions.length === 0}>
            Close all
          </button>
        }
      >
        {data.positions.length === 0 ? (
          <div className="muted">No open positions.</div>
        ) : (
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Sector</th>
                  <th className="num">Qty</th>
                  <th className="num">Entry</th>
                  <th className="num">LTP</th>
                  <th className="num">Stop</th>
                  <th className="num">Target 1</th>
                  <th className="num">P&amp;L</th>
                  <th className="num">R</th>
                  <th className="num">MAE</th>
                  <th className="num">MFE</th>
                  <th className="num">Held</th>
                  <th>Strategy</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {data.positions.map((p: any) => (
                  <tr key={p.position_id}>
                    <td>
                      <strong>{p.symbol}</strong>
                    </td>
                    <td>
                      <Pill value={p.direction} />
                    </td>
                    <td>{p.sector || '—'}</td>
                    <td className="num">{p.quantity}</td>
                    <td className="num">{num(p.entry_price)}</td>
                    <td className="num">{num(p.ltp)}</td>
                    <td className="num">{num(p.current_stop)}</td>
                    <td className="num">{num(p.target_1)}</td>
                    <td className={`num ${p.unrealized_pnl >= 0 ? 'good' : 'bad'}`}>{money(p.unrealized_pnl)}</td>
                    <td className="num">{num(p.r_multiple)}</td>
                    <td className="num muted">{num(p.mae_r)}</td>
                    <td className="num muted">{num(p.mfe_r)}</td>
                    <td className="num">{num(p.hold_minutes, 0)}m</td>
                    <td className="muted">{p.strategy_version}</td>
                    <td>
                      <button className="ghost" onClick={() => closeOne(p.position_id)} disabled={busy === p.position_id}>
                        Close
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <div className="grid c2">
        <Card title="Orders">
          {orders.data?.internal_orders?.length ? (
            <div className="scroll-x">
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Type</th>
                    <th>Leg</th>
                    <th className="num">Qty</th>
                    <th className="num">Filled</th>
                    <th className="num">Avg</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {orders.data.internal_orders
                    .slice()
                    .reverse()
                    .map((o: any) => (
                      <tr key={o.order_id}>
                        <td>{o.symbol}</td>
                        <td>
                          <Pill value={o.transaction_type === 'BUY' ? 'LONG' : 'SHORT'} />
                        </td>
                        <td>{o.order_type}</td>
                        <td>{o.leg}</td>
                        <td className="num">{o.quantity}</td>
                        <td className="num">{o.filled_quantity}</td>
                        <td className="num">{num(o.average_fill_price)}</td>
                        <td>
                          <Pill value={o.status === 'COMPLETE' ? 'OK' : o.status === 'REJECTED' ? 'REJECTED' : 'NEW'} />{' '}
                          <span className="muted">{o.status}</span>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="muted">No orders yet.</div>
          )}
          <div className="note">
            In PAPER mode no broker order is ever sent. These are simulated fills with modelled latency, slippage,
            spread and partial fills — and real Indian transaction charges.
          </div>
        </Card>

        <Card title="Execution slippage">
          {slippage.data ? (
            <>
              <div className="kv">
                <div>Samples</div>
                <div>{slippage.data.summary.samples}</div>
                <div>Median slippage</div>
                <div>{num(slippage.data.summary.median_bps, 1)} bps</div>
                <div>90th percentile</div>
                <div>{num(slippage.data.summary.p90_bps, 1)} bps</div>
                <div>Worst</div>
                <div>{num(slippage.data.summary.worst_bps, 1)} bps</div>
                <div>Total slippage cost</div>
                <div>{money(slippage.data.summary.total_cost)}</div>
                <div>Status</div>
                <div>
                  <Pill value={slippage.data.summary.level} />
                </div>
              </div>
              {Object.keys(slippage.data.summary.by_symbol || {}).length > 0 && (
                <>
                  <h3 style={{ marginTop: 12 }}>By symbol (median bps)</h3>
                  <div className="kv">
                    {Object.entries(slippage.data.summary.by_symbol).map(([k, v]) => (
                      <div key={k} style={{ display: 'contents' }}>
                        <div>{k}</div>
                        <div>{num(v as number, 1)}</div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </>
          ) : (
            <div className="muted">No fills yet.</div>
          )}
        </Card>
      </div>
    </div>
  )
}
