import { useState } from 'react'
import api from '../services/api'
import { Card, ErrorBox, Pill, Tabs, money, num, pct, useAsync } from '../components/common'

type Tab = 'trades' | 'review' | 'performance'

const QUALITY_LABEL: Record<string, string> = {
  GOOD_TRADE_GOOD_OUTCOME: 'Good trade / good outcome',
  GOOD_TRADE_BAD_OUTCOME: 'Good trade / bad outcome',
  BAD_TRADE_GOOD_OUTCOME: 'Bad trade / good outcome',
  BAD_TRADE_BAD_OUTCOME: 'Bad trade / bad outcome',
}

export default function Journal() {
  const [tab, setTab] = useState<Tab>('trades')
  const trades = useAsync(() => api.trades(200), [])
  const review = useAsync(() => api.review(), [])
  const performance = useAsync(() => api.performance(365), [])
  const [detail, setDetail] = useState<any>(null)

  const openTrade = async (tradeId: string) => {
    try {
      setDetail(await api.tradeDetail(tradeId))
    } catch (e: any) {
      setDetail({ error: e?.message || String(e) })
    }
  }

  return (
    <div>
      <Tabs
        tabs={[
          { id: 'trades', label: `Trades (${trades.data?.count ?? 0})` },
          { id: 'review', label: 'Daily review' },
          { id: 'performance', label: 'Performance analytics' },
        ]}
        active={tab}
        onChange={setTab}
      />

      {tab === 'trades' && (
        <>
          <Card title="Trade journal">
            <div className="note">
              A losing trade is not automatically a mistake. Classification separates <strong>process</strong> from{' '}
              <strong>outcome</strong>: <em>good trade / bad outcome</em> means the rules were followed and the result
              was simply negative. Only process violations count against the strategy.
            </div>
            {trades.loading && !trades.data && <div className="spinner">Loading…</div>}
            {trades.data?.trades?.length ? (
              <div className="scroll-x dd">
                <table>
                  <thead>
                    <tr>
                      <th>Closed</th>
                      <th>Symbol</th>
                      <th>Side</th>
                      <th className="num">Qty</th>
                      <th className="num">Entry</th>
                      <th className="num">Exit</th>
                      <th className="num">P&amp;L</th>
                      <th className="num">R</th>
                      <th className="num">MAE</th>
                      <th className="num">MFE</th>
                      <th>Exit reason</th>
                      <th>Regime</th>
                      <th>Process</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.data.trades.map((t: any) => (
                      <tr key={t.trade_id}>
                        <td className="muted">{t.exit_ts.slice(0, 16).replace('T', ' ')}</td>
                        <td><strong>{t.symbol}</strong></td>
                        <td><Pill value={t.direction} /></td>
                        <td className="num">{t.quantity}</td>
                        <td className="num">{num(t.entry_price)}</td>
                        <td className="num">{num(t.exit_price)}</td>
                        <td className={`num ${t.net_pnl >= 0 ? 'good' : 'bad'}`}>{money(t.net_pnl)}</td>
                        <td className={`num ${t.r_multiple >= 0 ? 'good' : 'bad'}`}>{num(t.r_multiple, 2)}</td>
                        <td className="num muted">{num(t.mae_r, 2)}</td>
                        <td className="num muted">{num(t.mfe_r, 2)}</td>
                        <td>{t.exit_reason}</td>
                        <td className="muted">{t.regime_at_entry}</td>
                        <td className="muted" style={{ fontSize: 12 }}>{QUALITY_LABEL[t.process_quality] || t.process_quality}</td>
                        <td>
                          <button className="ghost" onClick={() => openTrade(t.trade_id)}>Card</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="muted">
                No journaled trades yet. Trades are journaled after market close by the post-market job, or when the
                scheduler runs.
              </div>
            )}
          </Card>

          {detail && (
            <Card
              title={`Trade card — ${detail.trade?.symbol ?? 'error'}`}
              actions={<button className="secondary" onClick={() => setDetail(null)}>Close</button>}
            >
              {detail.error ? (
                <ErrorBox message={detail.error} />
              ) : (
                <div className="grid c2">
                  <div>
                    <p style={{ fontSize: 15 }}>{detail.explanation?.simple}</p>
                    <div className="kv">
                      <div>Entry</div><div>{num(detail.trade.entry_price)} at {detail.trade.entry_ts.slice(0, 16).replace('T', ' ')}</div>
                      <div>Exit</div><div>{num(detail.trade.exit_price)} at {detail.trade.exit_ts.slice(0, 16).replace('T', ' ')}</div>
                      <div>Quantity</div><div>{detail.trade.quantity}</div>
                      <div>Gross P&amp;L</div><div>{money(detail.trade.gross_pnl)}</div>
                      <div>Fees</div><div>{money(detail.trade.fees)}</div>
                      <div>Slippage cost</div><div>{money(detail.trade.slippage_cost)}</div>
                      <div>Net P&amp;L</div><div>{money(detail.trade.net_pnl)}</div>
                      <div>R multiple</div><div>{num(detail.trade.r_multiple, 2)}</div>
                      <div>MAE / MFE</div><div>{num(detail.trade.mae_r, 2)}R / {num(detail.trade.mfe_r, 2)}R</div>
                      <div>Held</div><div>{num(detail.trade.holding_minutes, 0)} minutes</div>
                      <div>Exit reason</div><div>{detail.trade.exit_reason}</div>
                    </div>
                  </div>
                  <div>
                    <h3>Context at entry</h3>
                    <div className="kv">
                      <div>Regime</div><div>{detail.trade.regime_at_entry}</div>
                      <div>Sector</div><div>{detail.trade.sector}</div>
                      <div>Sector rank</div><div>{detail.trade.sector_rank_at_entry ?? '—'}</div>
                      <div>India VIX</div><div>{num(detail.trade.vix_at_entry, 1)}</div>
                      <div>Strategy version</div><div>{detail.trade.strategy_version}</div>
                      <div>Process classification</div>
                      <div>{QUALITY_LABEL[detail.trade.process_quality] || detail.trade.process_quality}</div>
                    </div>
                    <h3 style={{ marginTop: 12 }}>Feature vector at entry</h3>
                    <pre>{JSON.stringify(detail.trade.features, null, 2)}</pre>
                  </div>
                </div>
              )}
            </Card>
          )}
        </>
      )}

      {tab === 'review' && (
        <Card title={`Daily review — ${review.data?.trading_date ?? ''}`}>
          {review.loading && !review.data && <div className="spinner">Building the review…</div>}
          {review.data && (
            <>
              <p style={{ fontSize: 15 }}>{review.data.simple}</p>
              <div className="grid c4" style={{ margin: '14px 0' }}>
                <div className="stat"><div className="k">Net P&amp;L</div><div className={`v ${review.data.net_pnl >= 0 ? 'good' : 'bad'}`}>{money(review.data.net_pnl)}</div></div>
                <div className="stat"><div className="k">Return</div><div className={`v ${review.data.return_pct >= 0 ? 'good' : 'bad'}`}>{pct(review.data.return_pct)}</div></div>
                <div className="stat"><div className="k">Trades</div><div className="v">{review.data.trades}</div><div className="s">{review.data.wins}W / {review.data.losses}L</div></div>
                <div className="stat"><div className="k">Risk events</div><div className="v warn">{review.data.risk_events}</div></div>
              </div>
              <h3>Process quality</h3>
              <div className="kv" style={{ maxWidth: 480 }}>
                {Object.entries(review.data.process_quality_counts || {}).map(([k, v]) => (
                  <div key={k} style={{ display: 'contents' }}>
                    <div>{QUALITY_LABEL[k] || k}</div>
                    <div>{v as number}</div>
                  </div>
                ))}
                {Object.keys(review.data.process_quality_counts || {}).length === 0 && (
                  <div style={{ display: 'contents' }}>
                    <div>No trades today</div>
                    <div>0</div>
                  </div>
                )}
              </div>
              {review.data.risk_events?.length > 0 && (
                <>
                  <h3 style={{ marginTop: 12 }}>Risk events</h3>
                  <div className="scroll-x">
                    <table>
                      <thead>
                        <tr><th>Time</th><th>Type</th><th>Severity</th><th>Action</th><th>Message</th></tr>
                      </thead>
                      <tbody>
                        {review.data.risk_events.map((e: any, i: number) => (
                          <tr key={i}>
                            <td className="muted">{e.ts.slice(11, 19)}</td>
                            <td>{e.type}</td>
                            <td><Pill value={e.severity} /></td>
                            <td>{e.action}</td>
                            <td className="muted">{e.message}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </>
          )}
        </Card>
      )}

      {tab === 'performance' && (
        <Card title="Performance analytics">
          {performance.data && (
            <>
              <div className="grid c4">
                <div className="stat"><div className="k">Days recorded</div><div className="v">{performance.data.days}</div></div>
                <div className="stat"><div className="k">Total trades</div><div className="v">{performance.data.total_trades}</div></div>
                <div className="stat"><div className="k">Total net P&amp;L</div><div className={`v ${performance.data.total_net_pnl >= 0 ? 'good' : 'bad'}`}>{money(performance.data.total_net_pnl)}</div></div>
                <div className="stat"><div className="k">Total fees</div><div className="v bad">{money(performance.data.total_fees)}</div></div>
                <div className="stat"><div className="k">Average R</div><div className="v">{num(performance.data.average_r_multiple, 3)}</div></div>
                {performance.data.curve && (
                  <>
                    <div className="stat"><div className="k">Sharpe</div><div className="v">{num(performance.data.curve.sharpe, 2)}</div></div>
                    <div className="stat"><div className="k">Sortino</div><div className="v">{num(performance.data.curve.sortino, 2)}</div></div>
                    <div className="stat"><div className="k">Max drawdown</div><div className="v bad">{pct(performance.data.curve.max_drawdown_pct)}</div></div>
                  </>
                )}
              </div>
              <div className="note">
                Equity series is built from the daily performance table. Live (paper) results are separate from backtest
                results and are never mixed together.
              </div>
            </>
          )}
        </Card>
      )}
    </div>
  )
}
