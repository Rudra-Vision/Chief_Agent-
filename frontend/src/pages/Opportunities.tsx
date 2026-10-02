import { useMemo, useState } from 'react'
import api from '../services/api'
import { Card, ErrorBox, Pill, SimulatedBanner, Tabs, money, num, pct, useAsync } from '../components/common'

type Mode = 'simple' | 'advanced'

function ScoreBar({ score, components }: { score: number; components?: Record<string, number> }) {
  return (
    <div>
      <div style={{ fontWeight: 700 }}>{score.toFixed(1)}</div>
      {components && (
        <div style={{ fontSize: 11 }}>
          {Object.entries(components)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([k, v]) => (
              <div key={k} className="muted">
                {k.replace(/_/g, ' ')}: {v.toFixed(1)}
              </div>
            ))}
        </div>
      )}
    </div>
  )
}

export default function Opportunities() {
  const [refresh, setRefresh] = useState(true)
  const { data, loading, error, reload } = useAsync(() => api.opportunities(10, refresh), [refresh])
  const [mode, setMode] = useState<Mode>('simple')
  const [detail, setDetail] = useState<any>(null)
  const [selected, setSelected] = useState<any>(null)
  const [scanning, setScanning] = useState(false)
  const [manual, setManual] = useState({ instrument_key: '', direction: 'LONG', quantity: 1, entry_price: 0, stop_price: 0 })
  const [orderMsg, setOrderMsg] = useState('')

  const rows = useMemo(
    () => [...(data?.longs || []), ...(data?.shorts || [])].sort((a: any, b: any) => b.score - a.score),
    [data],
  )

  const rescan = async () => {
    setScanning(true)
    try {
      await reload()
    } finally {
      setScanning(false)
    }
  }

  const openDetail = async (row: any) => {
    setSelected(row)
    try {
      const result = await api.explain(row.instrument_key)
      setDetail(result.explanation)
    } catch (e: any) {
      setDetail({ simple: 'Could not load the explanation: ' + (e?.message || e) })
    }
  }

  const submitManual = async () => {
    setOrderMsg('')
    try {
      const result = await api.manualOrder({ ...manual, reason: 'manual paper order' })
      setOrderMsg(
        result.ok
          ? `✅ Filled ${result.result.filled_quantity} @ ${num(result.result.average_price)} (fees ₹${num(result.result.fees)})`
          : `⛔ Rejected: ${result.result.message}`,
      )
    } catch (e: any) {
      setOrderMsg('⛔ ' + (e?.message || e))
    }
  }

  return (
    <div>
      <SimulatedBanner isSimulated={Boolean(data?.is_simulated)} />

      <Card
        title="Ranked opportunities"
        actions={
          <>
            <button className="secondary" onClick={() => setMode(mode === 'simple' ? 'advanced' : 'simple')}>
              {mode === 'simple' ? 'Switch to ADVANCED view' : 'Switch to SIMPLE view'}
            </button>
            <button onClick={rescan} disabled={scanning}>
              {scanning ? 'Scanning…' : 'Rescan'}
            </button>
          </>
        }
      >
        {loading && !data && <div className="spinner">Scanning the market…</div>}
        {error && <ErrorBox message={error} />}
        {data?.error && <ErrorBox message={data.error} />}
        {data && !data.error && (
          <>
            <div className="kv" style={{ marginBottom: 12, maxWidth: 720 }}>
              <div>Session scanned</div>
              <div>{data.session || '—'}</div>
              <div>Evaluation points</div>
              <div>{data.evaluations ?? '—'}</div>
              <div>Universe</div>
              <div>{data.universe_scanned ?? 0} instruments</div>
              <div>Market regime</div>
              <div>{data.regime || '—'}</div>
              <div>Account equity used for sizing</div>
              <div>{money(data.equity)}</div>
            </div>
            <div className="note">{data.as_of_note}</div>

            {rows.length === 0 ? (
              <div className="note warn">
                No qualifying setup in this session. That is a valid outcome — most filters exist to keep the system
                out of marginal trades. The rejection census below shows exactly which rules blocked candidates.
              </div>
            ) : (
              <div className="scroll-x">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Symbol</th>
                      <th>Side</th>
                      <th>Score</th>
                      <th className="num">Entry</th>
                      <th className="num">Stop</th>
                      <th className="num">Target 1</th>
                      <th className="num">R:R</th>
                      <th>Sector</th>
                      <th>Regime</th>
                      <th>VWAP</th>
                      <th className="num">RVOL</th>
                      <th className="num">Size</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((o: any) => (
                      <tr key={`${o.instrument_key}-${o.direction}`}>
                        <td>{o.rank}</td>
                        <td>
                          <strong>{o.symbol}</strong>
                        </td>
                        <td>
                          <Pill value={o.direction} />
                        </td>
                        <td>
                          <ScoreBar score={o.score} components={mode === 'advanced' ? o.score_components : undefined} />
                        </td>
                        <td className="num">{num(o.entry_price)}</td>
                        <td className="num">{num(o.stop_price)}</td>
                        <td className="num">{num(o.target_1)}</td>
                        <td className="num">{num(o.risk_reward, 1)}</td>
                        <td>{o.sector || '—'}</td>
                        <td>{o.regime || '—'}</td>
                        <td>{o.vwap_state}</td>
                        <td className="num">{num(o.rvol, 2)}</td>
                        <td className="num">{o.quantity_hint}</td>
                        <td>
                          <button className="ghost" onClick={() => openDetail(o)}>
                            Why?
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {Object.keys(data.rejected_by_gate || {}).length > 0 && (
              <div style={{ marginTop: 14 }}>
                <h3>Why candidates were blocked (whole session)</h3>
                <div className="scroll-x">
                  <table>
                    <thead>
                      <tr>
                        <th>Filter</th>
                        <th className="num">Rejections</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(data.rejected_by_gate).map(([gate, count]) => (
                        <tr key={gate}>
                          <td>{gate.replace(/_/g, ' ')}</td>
                          <td className="num">{count as number}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="note">
                  These are trades the system deliberately did <em>not</em> take. They are recorded so the research
                  engine can later ask "what would have happened if we had?" — that is pure false-negative evidence.
                </div>
              </div>
            )}
          </>
        )}
      </Card>

      {selected && detail && (
        <Card
          title={`${selected.symbol} — ${selected.direction} — score ${selected.score?.toFixed?.(1)}/100`}
          actions={
            <button className="secondary" onClick={() => { setSelected(null); setDetail(null) }}>
              Close
            </button>
          }
        >
          <Tabs
            tabs={[
              { id: 'simple', label: 'SIMPLE (plain English)' },
              { id: 'advanced', label: 'ADVANCED (full detail)' },
            ]}
            active={mode}
            onChange={setMode}
          />
          {mode === 'simple' ? (
            <>
              <p style={{ fontSize: 15 }}>{detail.simple}</p>
              {detail.risks?.length > 0 && (
                <>
                  <h3>Risks</h3>
                  <ul className="reasons">
                    {detail.risks.map((r: string, i: number) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </>
              )}
            </>
          ) : (
            <div className="grid c2">
              <div>
                <h3>Levels</h3>
                <div className="kv">
                  <div>Entry</div>
                  <div>{num(detail.advanced?.entry)}</div>
                  <div>Stop</div>
                  <div>{num(detail.advanced?.stop)}</div>
                  <div>Target 1</div>
                  <div>{num(detail.advanced?.target_1)}</div>
                  <div>Target 2</div>
                  <div>{num(detail.advanced?.target_2)}</div>
                  <div>Risk per share</div>
                  <div>{num(detail.advanced?.risk_per_share)}</div>
                  <div>Risk/reward</div>
                  <div>1:{num(detail.advanced?.risk_reward, 2)}</div>
                </div>
                <h3 style={{ marginTop: 12 }}>Context</h3>
                <div className="kv">
                  <div>Regime</div>
                  <div>{detail.advanced?.regime}</div>
                  <div>Sector</div>
                  <div>
                    {detail.advanced?.sector} {detail.advanced?.sector_rank ? `(#${detail.advanced.sector_rank})` : ''}
                  </div>
                  <div>VWAP</div>
                  <div>{num(detail.advanced?.vwap)}</div>
                  <div>Relative volume</div>
                  <div>{num(detail.advanced?.relative_volume, 2)}x</div>
                  <div>Daily ATR</div>
                  <div>{pct(detail.advanced?.daily_atr_pct)}</div>
                  <div>Spread</div>
                  <div>{pct(detail.advanced?.spread_pct, 3)}</div>
                  <div>ADX</div>
                  <div>{num(detail.advanced?.adx, 1)}</div>
                  <div>ATR percentile</div>
                  <div>{num(detail.advanced?.atr_percentile, 2)}</div>
                  <div>Relative strength vs NIFTY</div>
                  <div>{pct(detail.advanced?.relative_strength)}</div>
                </div>
              </div>
              <div>
                <h3>Opportunity score components</h3>
                <div className="kv">
                  {Object.entries(detail.advanced?.score_components || {})
                    .sort((a: any, b: any) => b[1] - a[1])
                    .map(([k, v]) => (
                      <div key={k} style={{ display: 'contents' }}>
                        <div>{k.replace(/_/g, ' ')}</div>
                        <div>{Number(v).toFixed(2)}</div>
                      </div>
                    ))}
                </div>
                <h3 style={{ marginTop: 12 }}>Reasons the setup qualified</h3>
                <ul className="reasons">
                  {(detail.bullets || []).map((r: string, i: number) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          <div className="note">{detail.disclaimer}</div>
        </Card>
      )}

      <Card title="Manual paper order (still risk-checked)">
        <div className="note">
          The dashboard cannot bypass risk. Every manual order runs through the same risk engine, sizing rules and
          execution lifecycle as a strategy signal.
        </div>
        <div className="grid c3">
          <div className="field">
            <label>Instrument key</label>
            <input
              value={manual.instrument_key}
              placeholder="NSE_EQ|INE002A01018 or SIM|RELIANCE"
              onChange={(e) => setManual({ ...manual, instrument_key: e.target.value })}
            />
          </div>
          <div className="field">
            <label>Direction</label>
            <select value={manual.direction} onChange={(e) => setManual({ ...manual, direction: e.target.value })}>
              <option value="LONG">LONG</option>
              <option value="SHORT">SHORT</option>
            </select>
          </div>
          <div className="field">
            <label>Quantity</label>
            <input type="number" value={manual.quantity} onChange={(e) => setManual({ ...manual, quantity: Number(e.target.value) })} />
          </div>
          <div className="field">
            <label>Entry price</label>
            <input type="number" value={manual.entry_price} onChange={(e) => setManual({ ...manual, entry_price: Number(e.target.value) })} />
          </div>
          <div className="field">
            <label>Stop price (mandatory)</label>
            <input type="number" value={manual.stop_price} onChange={(e) => setManual({ ...manual, stop_price: Number(e.target.value) })} />
          </div>
          <div className="field" style={{ display: 'flex', alignItems: 'flex-end' }}>
            <button onClick={submitManual}>Place paper order</button>
          </div>
        </div>
        {orderMsg && <div className={orderMsg.startsWith('✅') ? 'ok-box' : 'error-box'}>{orderMsg}</div>}
      </Card>
    </div>
  )
}
