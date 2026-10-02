import { useEffect, useState } from 'react'
import api from '../services/api'
import { Card, ErrorBox, Pill, SimulatedBanner, Tabs, money, num, pct, useAsync } from '../components/common'
import Plot from '../charts/Plot'

type TradeView = 'all' | 'winners' | 'losers'

export default function Backtest() {
  const defaults = useAsync(() => api.backtestDefaults(), [])
  const runs = useAsync(() => api.backtestRuns(20), [])
  const [form, setForm] = useState<any>({
    start_date: '',
    end_date: '',
    universe: [] as string[],
    initial_capital: 500000,
    risk_per_trade_pct: 0.0025,
    slippage_multiplier: 1,
    fee_multiplier: 1,
    max_universe: 40,
    include_shorts: true,
  })
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [detail, setDetail] = useState<any>(null)
  const [error, setError] = useState('')
  const [tradeView, setTradeView] = useState<TradeView>('all')

  useEffect(() => {
    if (defaults.data) {
      setForm((f: any) => ({
        ...f,
        start_date: f.start_date || defaults.data.start_date,
        end_date: f.end_date || defaults.data.end_date,
        universe: f.universe.length ? f.universe : defaults.data.universe,
        max_universe: Math.min(f.max_universe, Math.max(4, defaults.data.universe.length)),
      }))
    }
  }, [defaults.data])

  const run = async () => {
    setRunning(true)
    setError('')
    setResult(null)
    setDetail(null)
    try {
      const payload = { ...form, run_monte_carlo: true }
      const res = await api.runBacktest(payload)
      setResult(res)
      if (res.backtest_id) {
        const d = await api.backtestDetail(res.backtest_id, 1000)
        setDetail(d)
      }
      runs.reload()
    } catch (e: any) {
      setError(e?.message || String(e))
    } finally {
      setRunning(false)
    }
  }

  const loadRun = async (id: string) => {
    setRunning(true)
    try {
      const d = await api.backtestDetail(id, 1000)
      setDetail(d)
      setResult({ backtest_id: id, metrics: d.metrics, data_source: d.config?.data_source })
    } finally {
      setRunning(false)
    }
  }

  const m = detail?.metrics || result?.metrics
  const trades = detail?.trades || []
  const shownTrades =
    tradeView === 'all' ? trades : tradeView === 'winners' ? trades.filter((t: any) => t.net_pnl > 0) : trades.filter((t: any) => t.net_pnl <= 0)

  const equityData = detail
    ? [
        {
          x: detail.equity_curve.map((p: any) => p.date),
          y: detail.equity_curve.map((p: any) => p.equity),
          type: 'scatter',
          mode: 'lines',
          name: 'Equity',
          line: { color: '#34d399', width: 2 },
          fill: 'tozeroy',
          fillcolor: 'rgba(52,211,153,0.08)',
        },
      ]
    : []

  const ddData = detail
    ? [
        {
          x: detail.drawdown_curve.map((p: any) => p.date),
          y: detail.drawdown_curve.map((p: any) => p.drawdown_pct * 100),
          type: 'scatter',
          mode: 'lines',
          name: 'Drawdown %',
          line: { color: '#f87171', width: 2 },
          fill: 'tozeroy',
          fillcolor: 'rgba(248,113,113,0.15)',
        },
      ]
    : []

  const RHistogram = () => {
    if (!trades.length) return null
    return (
      <Plot
        height={260}
        data={[
          {
            x: trades.map((t: any) => t.r_multiple),
            type: 'histogram',
            nbinsx: 40,
            marker: { color: '#3b82f6' },
            name: 'R multiples',
          },
        ]}
        layout={{ xaxis: { title: { text: 'R multiple' } }, yaxis: { title: { text: 'trades' } }, bargap: 0.05 }}
      />
    )
  }

  return (
    <div>
      <SimulatedBanner isSimulated={Boolean(defaults.data?.is_simulated)} />

      <Card title="Run a backtest">
        {defaults.loading && !defaults.data && <div className="spinner">Loading defaults…</div>}
        {defaults.data && (
          <>
            <div className="grid c4">
              <div className="field">
                <label>Start date</label>
                <input type="date" value={form.start_date} onChange={(e) => setForm({ ...form, start_date: e.target.value })} />
              </div>
              <div className="field">
                <label>End date</label>
                <input type="date" value={form.end_date} onChange={(e) => setForm({ ...form, end_date: e.target.value })} />
              </div>
              <div className="field">
                <label>Capital (₹)</label>
                <input type="number" value={form.initial_capital} onChange={(e) => setForm({ ...form, initial_capital: Number(e.target.value) })} />
              </div>
              <div className="field">
                <label>Risk per trade (%)</label>
                <input
                  type="number"
                  step="0.05"
                  value={(form.risk_per_trade_pct * 100).toFixed(2)}
                  onChange={(e) => setForm({ ...form, risk_per_trade_pct: Math.min(0.005, Number(e.target.value) / 100) })}
                />
              </div>
              <div className="field">
                <label>Slippage multiplier</label>
                <input type="number" step="0.5" value={form.slippage_multiplier} onChange={(e) => setForm({ ...form, slippage_multiplier: Number(e.target.value) })} />
              </div>
              <div className="field">
                <label>Fee multiplier</label>
                <input type="number" step="0.5" value={form.fee_multiplier} onChange={(e) => setForm({ ...form, fee_multiplier: Number(e.target.value) })} />
              </div>
              <div className="field">
                <label>Max instruments</label>
                <input type="number" value={form.max_universe} onChange={(e) => setForm({ ...form, max_universe: Number(e.target.value) })} />
              </div>
              <div className="field">
                <label>Include short trades</label>
                <select value={form.include_shorts ? 'yes' : 'no'} onChange={(e) => setForm({ ...form, include_shorts: e.target.value === 'yes' })}>
                  <option value="yes">Yes</option>
                  <option value="no">No (long only)</option>
                </select>
              </div>
            </div>
            <div className="field">
              <label>
                Universe ({form.universe.length} of {defaults.data.universe_size} watchlist symbols — click to toggle)
              </label>
              <div style={{ maxHeight: 130, overflow: 'auto', border: '1px solid var(--border)', borderRadius: 8, padding: 8, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {defaults.data.universe.map((s: string) => {
                  const on = form.universe.includes(s)
                  return (
                    <button
                      key={s}
                      className={on ? '' : 'ghost'}
                      style={{ padding: '3px 9px', fontSize: 12 }}
                      onClick={() =>
                        setForm({
                          ...form,
                          universe: on ? form.universe.filter((x: string) => x !== s) : [...form.universe, s],
                        })
                      }
                    >
                      {s}
                    </button>
                  )
                })}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <button onClick={run} disabled={running || form.universe.length === 0}>
                {running ? 'Running…' : '▶ RUN BACKTEST'}
              </button>
              <span className="muted">
                {defaults.data.data_source === 'SIMULATED'
                  ? 'running on simulated data'
                  : 'running on cached Upstox data'}
              </span>
            </div>
          </>
        )}
      </Card>

      {error && <ErrorBox message={error} />}

      {m && (
        <>
          <Card title={`Results — ${detail?.strategy_version || ''}`}>
            <div className="grid c4">
              <div className="stat"><div className="k">Net return</div><div className={`v ${m.net_return_pct >= 0 ? 'good' : 'bad'}`}>{pct(m.net_return_pct)}</div><div className="s">annualised {pct(m.annualised_return_pct)}</div></div>
              <div className="stat"><div className="k">Max drawdown</div><div className="v bad">{pct(m.max_drawdown_pct)}</div></div>
              <div className="stat"><div className="k">Sharpe</div><div className="v">{num(m.sharpe, 2)}</div></div>
              <div className="stat"><div className="k">Sortino</div><div className="v">{num(m.sortino, 2)}</div></div>
              <div className="stat"><div className="k">Trades</div><div className="v">{m.trades}</div><div className="s">{(m.exposure_pct * 100).toFixed(1)}% time in market</div></div>
              <div className="stat"><div className="k">Win rate</div><div className="v">{pct(m.win_rate)}</div><div className="s">{(m.win_rate * m.trades).toFixed(0)}W / {(m.loss_rate * m.trades).toFixed(0)}L</div></div>
              <div className="stat"><div className="k">Expectancy</div><div className={`v ${m.expectancy_r >= 0 ? 'good' : 'bad'}`}>{num(m.expectancy_r, 3)}R</div><div className="s">{money(m.expectancy)} per trade</div></div>
              <div className="stat"><div className="k">Profit factor</div><div className={`v ${m.profit_factor >= 1.3 ? 'good' : m.profit_factor >= 1 ? 'warn' : 'bad'}`}>{num(m.profit_factor, 2)}</div></div>
              <div className="stat"><div className="k">Total costs</div><div className="v bad">{money((m.fees || 0) + (m.slippage || 0))}</div><div className="s">fees {money(m.fees)} + slippage {money(m.slippage)}</div></div>
              <div className="stat"><div className="k">Avg holding</div><div className="v">{num(m.average_holding_minutes, 0)}m</div></div>
              <div className="stat"><div className="k">Max consecutive losses</div><div className="v warn">{m.max_consecutive_losses}</div></div>
              <div className="stat"><div className="k">Objective score</div><div className="v">{num(m.objective_score, 3)}</div><div className="s">risk-adjusted, not max profit</div></div>
            </div>
            <div className="note">
              Costs are always applied: brokerage, STT, exchange charges, SEBI fee, GST, stamp duty plus modelled
              slippage and spread. A backtest without realistic costs is prohibited in this system.
            </div>
          </Card>

          <div className="grid c2">
            <Card title="Equity curve">
              {equityData.length ? <Plot data={equityData} layout={{ yaxis: { title: { text: 'Equity (₹)' } } }} /> : <div className="muted">No curve.</div>}
            </Card>
            <Card title="Drawdown">
              {ddData.length ? <Plot data={ddData} layout={{ yaxis: { title: { text: 'Drawdown %' } } }} /> : <div className="muted">No curve.</div>}
            </Card>
          </div>

          {result?.monte_carlo && (
            <Card title="Monte Carlo (trade-sequence bootstrap)">
              <div className="grid c4">
                <div className="stat"><div className="k">Median return</div><div className="v">{pct(result.monte_carlo.return_pct.median)}</div><div className="s">5th {pct(result.monte_carlo.return_pct.p5)} · 95th {pct(result.monte_carlo.return_pct.p95)}</div></div>
                <div className="stat"><div className="k">Median drawdown</div><div className="v warn">{pct(result.monte_carlo.max_drawdown_pct.median)}</div><div className="s">95th {pct(result.monte_carlo.max_drawdown_pct.p95)}</div></div>
                <div className="stat"><div className="k">Ruin probability</div><div className={`v ${result.monte_carlo.ruin_probability > 0.02 ? 'bad' : 'good'}`}>{pct(result.monte_carlo.ruin_probability)}</div><div className="s">a {pct(result.monte_carlo.ruin_threshold_pct, 0)} drawdown</div></div>
                <div className="stat"><div className="k">Max losing streak</div><div className="v">{num(result.monte_carlo.max_losing_streak.median, 0)}</div><div className="s">worst {result.monte_carlo.max_losing_streak.worst}</div></div>
              </div>
              {result.monte_carlo.notes?.length > 0 && (
                <ul className="reasons">
                  {result.monte_carlo.notes.map((n: string, i: number) => (
                    <li key={i}>{n}</li>
                  ))}
                </ul>
              )}
            </Card>
          )}

          {detail && (
            <Card title="Trade distribution">
              <RHistogram />
            </Card>
          )}

          {detail && (
            <div className="grid c2">
              <Card title="Performance by market regime">
                <BreakdownTable data={detail.regime_performance} keyName="Regime" />
              </Card>
              <Card title="Performance by sector">
                <BreakdownTable data={detail.sector_performance} keyName="Sector" />
              </Card>
              <Card title="Long vs short">
                <BreakdownTable data={detail.side_performance} keyName="Side" />
              </Card>
              <Card title="Exit reasons">
                <BreakdownTable data={detail.time_of_day_performance} keyName="Entry time (IST)" />
              </Card>
            </div>
          )}

          {detail?.monthly_returns && Object.keys(detail.monthly_returns).length > 0 && (
            <Card title="Monthly returns">
              <div className="scroll-x">
                <table>
                  <thead>
                    <tr>
                      <th>Month</th>
                      <th className="num">Start</th>
                      <th className="num">End</th>
                      <th className="num">Return</th>
                      <th className="num">Days</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(detail.monthly_returns).map(([month, row]: any) => (
                      <tr key={month}>
                        <td>{month}</td>
                        <td className="num">{money(row.start_equity)}</td>
                        <td className="num">{money(row.end_equity)}</td>
                        <td className={`num ${row.return_pct >= 0 ? 'good' : 'bad'}`}>{pct(row.return_pct)}</td>
                        <td className="num">{row.trading_days}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {trades.length > 0 && (
            <Card title={`Trades (${trades.length})`}>
              <Tabs
                tabs={[
                  { id: 'all', label: `All (${trades.length})` },
                  { id: 'winners', label: `Winners (${trades.filter((t: any) => t.net_pnl > 0).length})` },
                  { id: 'losers', label: `Losers (${trades.filter((t: any) => t.net_pnl <= 0).length})` },
                ]}
                active={tradeView}
                onChange={setTradeView}
              />
              <div className="scroll-x dd">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Symbol</th>
                      <th>Side</th>
                      <th>Entry time</th>
                      <th className="num">Entry</th>
                      <th className="num">Exit</th>
                      <th className="num">Qty</th>
                      <th className="num">P&amp;L</th>
                      <th className="num">R</th>
                      <th className="num">MAE</th>
                      <th className="num">MFE</th>
                      <th className="num">Held</th>
                      <th>Exit reason</th>
                      <th>Regime</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shownTrades.map((t: any) => (
                      <tr key={t.trade_index}>
                        <td>{t.trade_index}</td>
                        <td>{t.symbol}</td>
                        <td><Pill value={t.direction} /></td>
                        <td className="muted">{t.entry_ts.slice(0, 16).replace('T', ' ')}</td>
                        <td className="num">{num(t.entry_price)}</td>
                        <td className="num">{num(t.exit_price)}</td>
                        <td className="num">{t.quantity}</td>
                        <td className={`num ${t.net_pnl >= 0 ? 'good' : 'bad'}`}>{money(t.net_pnl)}</td>
                        <td className={`num ${t.r_multiple >= 0 ? 'good' : 'bad'}`}>{num(t.r_multiple, 2)}</td>
                        <td className="num muted">{num(t.mae_r, 2)}</td>
                        <td className="num muted">{num(t.mfe_r, 2)}</td>
                        <td className="num">{num(t.holding_minutes, 0)}m</td>
                        <td>{t.exit_reason}</td>
                        <td className="muted">{t.regime}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}

      <Card title="Previous runs">
        {runs.data?.runs?.length ? (
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Strategy</th>
                  <th>Period</th>
                  <th className="num">Trades</th>
                  <th className="num">Net</th>
                  <th className="num">Max DD</th>
                  <th className="num">Sharpe</th>
                  <th className="num">PF</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {runs.data.runs.map((r: any) => (
                  <tr key={r.backtest_id}>
                    <td className="muted">{r.backtest_id}</td>
                    <td>{r.strategy_version}</td>
                    <td className="muted">
                      {r.start_date} → {r.end_date}
                    </td>
                    <td className="num">{r.trade_count}</td>
                    <td className={`num ${(r.metrics?.net_return_pct ?? 0) >= 0 ? 'good' : 'bad'}`}>{pct(r.metrics?.net_return_pct)}</td>
                    <td className="num bad">{pct(r.metrics?.max_drawdown_pct)}</td>
                    <td className="num">{num(r.metrics?.sharpe, 2)}</td>
                    <td className="num">{num(r.metrics?.profit_factor, 2)}</td>
                    <td>
                      <button className="ghost" onClick={() => loadRun(r.backtest_id)}>
                        Open
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="muted">No backtests yet.</div>
        )}
      </Card>
    </div>
  )
}

function BreakdownTable({ data, keyName }: { data?: Record<string, any>; keyName: string }) {
  if (!data || Object.keys(data).length === 0) return <div className="muted">No data.</div>
  return (
    <div className="scroll-x">
      <table>
        <thead>
          <tr>
            <th>{keyName}</th>
            <th className="num">Trades</th>
            <th className="num">Net</th>
            <th className="num">Win %</th>
            <th className="num">Expectancy R</th>
            <th className="num">PF</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data).map(([k, row]: any) => (
            <tr key={k}>
              <td>{k.replace(/_/g, ' ')}</td>
              <td className="num">{row.trades}</td>
              <td className={`num ${row.net_pnl >= 0 ? 'good' : 'bad'}`}>{money(row.net_pnl)}</td>
              <td className="num">{pct(row.win_rate)}</td>
              <td className={`num ${row.expectancy_r >= 0 ? 'good' : 'bad'}`}>{num(row.expectancy_r, 3)}</td>
              <td className="num">{num(row.profit_factor, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
