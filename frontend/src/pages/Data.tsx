import { useState } from 'react'
import api from '../services/api'
import { Card, ErrorBox, Pill, SimulatedBanner, money, num, pct, useAsync } from '../components/common'

export default function DataPage() {
  const source = useAsync(() => api.dataSource(), [])
  const universe = useAsync(() => api.universe(), [])
  const coverage = useAsync(() => api.coverage('1m'), [])
  const instruments = useAsync(() => api.instruments('', 25), [])
  const quality = useAsync(() => api.quality(), [])
  const [busy, setBusy] = useState('')
  const [msg, setMsg] = useState('')
  const [form, setForm] = useState({ years: 1, max_instruments: 30, timeframe: '1m' })

  const doRefresh = async () => {
    setBusy('instruments')
    setMsg('')
    try {
      const result = await api.refreshInstruments()
      setMsg(`Loaded ${result.instruments_loaded} instruments; ${result.universe} mapped to the watchlist.`)
      instruments.reload()
      universe.reload()
    } catch (e: any) {
      setMsg('Failed: ' + (e?.message || e))
    } finally {
      setBusy('')
    }
  }

  const doDownload = async () => {
    setBusy('download')
    setMsg('')
    try {
      const result = await api.download(form)
      const r = result.report
      setMsg(
        `Downloaded ${r.candles_inserted.toLocaleString()} candles for ${r.instruments_fetched}/${r.instruments_requested} instruments ` +
          `(${r.data_source}) in ${r.duration_seconds}s. Skipped ${r.instruments_skipped_cached} already cached.` +
          (r.failure_count ? ` ${r.failure_count} failure(s).` : ''),
      )
      coverage.reload()
    } catch (e: any) {
      setMsg('Failed: ' + (e?.message || e))
    } finally {
      setBusy('')
    }
  }

  const doQuality = async () => {
    setBusy('quality')
    try {
      const result = await api.runQualityCheck()
      setMsg(result.safe_to_trade ? `Data checks passed: ${result.summary}` : `DATA SAFE MODE: ${result.summary}`)
      quality.reload()
    } catch (e: any) {
      setMsg('Failed: ' + (e?.message || e))
    } finally {
      setBusy('')
    }
  }

  return (
    <div>
      <SimulatedBanner isSimulated={Boolean(source.data?.is_simulated)} />

      <Card title="Market data source">
        {source.data && (
          <>
            <div className="kv" style={{ maxWidth: 640 }}>
              <div>Active source</div>
              <div>
                {source.data.source} {source.data.is_simulated && <span className="badge-sim">SIMULATED</span>}
              </div>
              <div>Live Upstox available</div>
              <div>{source.data.live_available ? 'yes' : 'no (no access token)'}</div>
              <div>Reason</div>
              <div className="muted" style={{ textAlign: 'left' }}>{source.data.reason}</div>
            </div>
            <div className="note">{source.data.note}</div>
          </>
        )}
      </Card>

      <div className="grid c2">
        <Card
          title="Instrument master (Upstox BOD)"
          actions={
            <button onClick={doRefresh} disabled={busy === 'instruments'}>
              {busy === 'instruments' ? 'Downloading…' : 'Refresh instruments'}
            </button>
          }
        >
          <div className="note">
            Downloaded from the official Upstox BOD file. <code>instrument_key</code> is the canonical identifier;
            <code> exchange_token</code> is stored as metadata only because the exchange may reuse it after expiry.
          </div>
          {instruments.data && (
            <div className="kv">
              <div>Instruments in database</div>
              <div>{instruments.data.total_instruments?.toLocaleString()}</div>
              <div>In the watchlist universe</div>
              <div>{instruments.data.in_universe}</div>
            </div>
          )}
        </Card>

        <Card title="Historical download">
          <div className="grid c3">
            <div className="field">
              <label>Years of history</label>
              <input type="number" value={form.years} onChange={(e) => setForm({ ...form, years: Number(e.target.value) })} />
            </div>
            <div className="field">
              <label>Max instruments</label>
              <input type="number" value={form.max_instruments} onChange={(e) => setForm({ ...form, max_instruments: Number(e.target.value) })} />
            </div>
            <div className="field">
              <label>Timeframe</label>
              <select value={form.timeframe} onChange={(e) => setForm({ ...form, timeframe: e.target.value })}>
                <option value="1m">1 minute</option>
                <option value="5m">5 minute</option>
                <option value="15m">15 minute</option>
                <option value="1d">Daily</option>
              </select>
            </div>
          </div>
          <button onClick={doDownload} disabled={busy === 'download'}>
            {busy === 'download' ? 'Downloading…' : 'Download history (cache-first)'}
          </button>
          <div className="note">
            The downloader never re-requests data it already holds — only missing sessions are fetched. Source candles
            are append-only and are never overwritten. Higher timeframes are derived from verified 1-minute data.
          </div>
        </Card>
      </div>

      {msg && <div className={msg.toLowerCase().includes('fail') ? 'error-box' : 'ok-box'}>{msg}</div>}

      <div className="grid c2">
        <Card title="Cached history coverage">
          {coverage.data && (
            <>
              <div className="kv">
                <div>Total 1-minute candles cached</div>
                <div>{coverage.data.total_candles?.toLocaleString()}</div>
                <div>Instruments with data</div>
                <div>{coverage.data.instruments_with_data}</div>
              </div>
              <div className="scroll-x dd" style={{ marginTop: 10 }}>
                <table>
                  <thead>
                    <tr><th>Instrument</th><th className="num">Candles</th><th>From</th><th>To</th></tr>
                  </thead>
                  <tbody>
                    {coverage.data.items.map((row: any) => (
                      <tr key={row.instrument_key}>
                        <td>{row.instrument_key}</td>
                        <td className="num">{row.count?.toLocaleString()}</td>
                        <td className="muted">{row.first?.slice(0, 10)}</td>
                        <td className="muted">{row.last?.slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </Card>

        <Card
          title="Data quality"
          actions={
            <button onClick={doQuality} disabled={busy === 'quality'}>
              {busy === 'quality' ? 'Checking…' : 'Run checks now'}
            </button>
          }
        >
          {quality.data && (
            <>
              <div style={{ marginBottom: 10 }}>
                DATA SAFE MODE: {quality.data.data_safe_mode ? <Pill value="FAIL" /> : <Pill value="PASS" />}{' '}
                <span className="muted">{quality.data.last_report?.summary || 'not evaluated yet'}</span>
              </div>
              <div className="note">
                If market data cannot be trusted the system refuses to trade and enters DATA_SAFE_MODE. It fails closed,
                never open.
              </div>
              {quality.data.recent_incidents?.length > 0 ? (
                <div className="scroll-x dd">
                  <table>
                    <thead>
                      <tr><th>When</th><th>Type</th><th>Severity</th><th>Detail</th></tr>
                    </thead>
                    <tbody>
                      {quality.data.recent_incidents.map((i: any, idx: number) => (
                        <tr key={idx}>
                          <td className="muted">{i.ts?.slice(0, 19).replace('T', ' ')}</td>
                          <td>{i.type}</td>
                          <td><Pill value={i.severity === 'CRITICAL' ? 'FAIL' : i.severity} /></td>
                          <td className="muted">{(i.details?.detail || '').slice(0, 90)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="muted">No incidents recorded.</div>
              )}
            </>
          )}
        </Card>
      </div>

      <Card title={`Watchlist universe (${universe.data?.count ?? 0} symbols)`}>
        {universe.data && (
          <>
            <div className="note">
              Symbols are resolved to <code>instrument_key</code> using the official Upstox master. Resolved:{' '}
              <strong>{universe.data.resolved}</strong>, unresolved: <strong>{universe.data.unresolved}</strong>.
              Unresolved symbols use a clearly-marked <code>SIM|SYMBOL</code> placeholder so simulated runs can never be
              mistaken for real instruments.
            </div>
            <div className="grid c2">
              <div>
                <h3>Sector distribution</h3>
                <div className="kv">
                  {Object.entries(universe.data.sectors || {}).map(([k, v]) => (
                    <div key={k} style={{ display: 'contents' }}>
                      <div>{k}</div>
                      <div>{v as number}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <h3>Symbols</h3>
                <div className="scroll-x dd" style={{ maxHeight: 300 }}>
                  <table>
                    <thead>
                      <tr><th>Symbol</th><th>Sector</th><th>Tier</th><th>Resolved</th></tr>
                    </thead>
                    <tbody>
                      {universe.data.items.slice(0, 200).map((row: any) => (
                        <tr key={row.symbol}>
                          <td>{row.symbol}</td>
                          <td className="muted">{row.sector}</td>
                          <td className="num">{row.tier}</td>
                          <td>{row.resolved ? '✅' : '⚠ placeholder'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </>
        )}
      </Card>
    </div>
  )
}
