/**
 * API client.
 *
 * All broker access happens server-side; this client only ever talks to the
 * Chief Agent backend. No broker token is present in the browser, and the
 * backend never returns one.
 */

export interface ModeSummary {
  mode: 'SANDBOX' | 'PAPER' | 'LIVE'
  requested_mode: string
  is_simulated: boolean
  is_live: boolean
  deployment_stage: string
  deployment_stage_value: number
  live_permitted: boolean
  live_blocked_reasons: string[]
  banner: string
  data_source: string
  strategy_version: string | null
}

export interface PortfolioSnapshot {
  equity: number
  cash: number
  positions_value: number
  starting_equity_today: number
  realized_pnl_today: number
  unrealized_pnl: number
  fees_today: number
  daily_return_pct: number
  peak_equity: number
  drawdown_pct: number
  open_risk_pct: number
  gross_exposure_pct: number
  open_position_count: number
  trades_today: number
  consecutive_losses: number
  weekly_return_pct: number
  open_positions: any[]
}

export interface SystemStatus {
  app: { name: string; version: string; environment: string; server_time: string }
  mode: ModeSummary
  kill_switch: { engaged: boolean; reason: string; source?: string | null; engaged_at?: string | null }
  portfolio: PortfolioSnapshot
  broker: {
    mode: string
    authenticated: boolean
    token_source: string
    token_valid_today: boolean
    instrument_master_loaded: boolean
    instrument_count: number
    static_ip_configured: boolean
  }
  data: { source: string; is_simulated: boolean; reason: string; data_safe_mode: boolean; quality_summary: string }
  market: any
  strategy: { version: string | null; family?: string; config_hash?: string | null }
  risk: Record<string, any>
  reconciliation: { pending: boolean; last_report: any }
  slippage: any
  counters: Record<string, any>
  notifications: { channels: string[]; counts: Record<string, number>; history_size: number }
  preflight: any
}

const CSRF_HEADER = 'X-CSRF-Token'

let csrfToken: string | null = null

export function setCsrfToken(token: string | null) {
  csrfToken = token
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) }
  if (init.body && !headers['Content-Type']) headers['Content-Type'] = 'application/json'
  if (csrfToken && init.method && init.method !== 'GET') headers[CSRF_HEADER] = csrfToken

  const response = await fetch(path, { ...init, headers, credentials: 'same-origin' })
  const text = await response.text()
  let payload: any = null
  try {
    payload = text ? JSON.parse(text) : null
  } catch {
    payload = { detail: text.slice(0, 400) }
  }
  if (!response.ok) {
    const message = payload?.detail || payload?.message || `HTTP ${response.status}`
    const error = new Error(typeof message === 'string' ? message : JSON.stringify(message))
    ;(error as any).status = response.status
    ;(error as any).payload = payload
    throw error
  }
  return payload as T
}

const get = <T>(path: string) => request<T>(path)
const post = <T>(path: string, body?: any) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })
const put = <T>(path: string, body?: any) =>
  request<T>(path, { method: 'PUT', body: body === undefined ? undefined : JSON.stringify(body) })

export const api = {
  // ---- system -----------------------------------------------------------
  health: () => get<any>('/api/system/health'),
  status: () => get<SystemStatus>('/api/system/status'),
  mode: () => get<ModeSummary>('/api/system/mode'),
  config: () => get<any>('/api/system/config'),
  notifications: (limit = 50) => get<any>(`/api/system/notifications?limit=${limit}`),
  scheduler: () => get<any>('/api/system/scheduler'),
  runJob: (job: string) => post<any>('/api/system/scheduler/run', { job }),
  authStatus: () => get<any>('/api/auth/status'),
  login: (username: string, password: string) => post<any>('/api/auth/login', { username, password }),
  logout: () => post<any>('/api/auth/logout'),

  // ---- broker -----------------------------------------------------------
  brokerStatus: () => get<any>('/api/broker/status'),
  funds: () => get<any>('/api/broker/funds'),
  brokerPositions: () => get<any>('/api/broker/positions'),
  holdings: () => get<any>('/api/broker/holdings'),
  orders: () => get<any>('/api/broker/orders'),
  loginUrl: () => get<{ url: string; state: string }>('/api/broker/upstox/login'),
  attachToken: (access_token: string) => post<any>('/api/broker/upstox/token', { access_token }),
  logoutBroker: () => post<any>('/api/broker/upstox/logout'),
  preflight: (deep = false) => get<any>(`/api/broker/preflight?deep=${deep}`),
  reconciliation: () => get<any>('/api/broker/reconciliation'),
  runReconciliation: () => post<any>('/api/broker/reconciliation/run'),
  killSwitch: () => get<any>('/api/broker/kill-switch'),
  engageKillSwitch: (reason: string, useBrokerSwitch = false) =>
    post<any>('/api/broker/kill-switch/engage', { reason, use_broker_switch: useBrokerSwitch }),
  releaseKillSwitch: () => post<any>('/api/broker/kill-switch/release'),

  // ---- data -------------------------------------------------------------
  dataSource: () => get<any>('/api/data/source'),
  instruments: (query = '', limit = 50) => get<any>(`/api/data/instruments?query=${encodeURIComponent(query)}&limit=${limit}`),
  refreshInstruments: () => post<any>('/api/data/instruments/refresh'),
  universe: () => get<any>('/api/data/universe'),
  coverage: (timeframe = '1m') => get<any>(`/api/data/coverage?timeframe=${timeframe}`),
  download: (body: any) => post<any>('/api/data/download', body),
  candles: (instrument_key: string, timeframe = '1m', start?: string, end?: string, limit = 2000) =>
    get<any>(
      `/api/data/candles?instrument_key=${encodeURIComponent(instrument_key)}&timeframe=${timeframe}` +
        (start ? `&start=${start}` : '') +
        (end ? `&end=${end}` : '') +
        `&limit=${limit}`,
    ),
  quality: () => get<any>('/api/data/quality'),
  runQualityCheck: () => post<any>('/api/data/quality/check'),

  // ---- trading ----------------------------------------------------------
  opportunities: (topN = 10, refresh = true) => get<any>(`/api/trading/opportunities?top_n=${topN}&refresh=${refresh}`),
  explain: (instrumentKey: string) => get<any>(`/api/trading/opportunities/${encodeURIComponent(instrumentKey)}/explain`),
  positions: () => get<any>('/api/trading/positions'),
  paperAccount: () => get<any>('/api/trading/paper/account'),
  manualOrder: (body: any) => post<any>('/api/trading/paper/order', body),
  closePosition: (positionId: string) => post<any>(`/api/trading/positions/${positionId}/close`),
  closeAll: () => post<any>('/api/trading/positions/close-all'),
  runCycle: () => post<any>('/api/trading/cycle'),
  signals: (limit = 50) => get<any>(`/api/trading/signals?limit=${limit}`),
  slippage: () => get<any>('/api/trading/slippage'),

  // ---- backtest ---------------------------------------------------------
  backtestDefaults: () => get<any>('/api/backtest/defaults'),
  runBacktest: (body: any) => post<any>('/api/backtest/run', body),
  backtestRuns: (limit = 25) => get<any>(`/api/backtest/runs?limit=${limit}`),
  backtestDetail: (id: string, tradeLimit = 500) => get<any>(`/api/backtest/${id}?trade_limit=${tradeLimit}`),

  // ---- research ---------------------------------------------------------
  champion: () => get<any>('/api/research/champion'),
  versions: (limit = 50) => get<any>(`/api/research/versions?limit=${limit}`),
  version: (v: string) => get<any>(`/api/research/versions/${v}`),
  hypotheses: (limit = 50) => get<any>(`/api/research/hypotheses?limit=${limit}`),
  generateHypotheses: (maxProposals = 5) =>
    post<any>('/api/research/hypotheses/generate', { persist: true, max_proposals: maxProposals }),
  experiments: (limit = 50) => get<any>(`/api/research/experiments?limit=${limit}`),
  runExperiment: (body: any) => post<any>('/api/research/experiments/run', body),
  promote: (experimentId: string) => post<any>(`/api/research/experiments/${experimentId}/promote`),
  learnings: (query = '', limit = 50) =>
    get<any>(`/api/research/learnings?limit=${limit}${query ? `&query=${encodeURIComponent(query)}` : ''}`),
  askMemory: (question: string) => get<any>(`/api/research/memory/ask?question=${encodeURIComponent(question)}`),
  graph: (experimentId: string) => get<any>(`/api/research/graph/${experimentId}`),
  multipleTesting: () => get<any>('/api/research/multiple-testing'),
  journalStats: () => get<any>('/api/research/journal-stats'),

  // ---- journal ----------------------------------------------------------
  trades: (limit = 100) => get<any>(`/api/journal/trades?limit=${limit}`),
  tradeDetail: (tradeId: string) => get<any>(`/api/journal/trades/${tradeId}`),
  daily: (limit = 90) => get<any>(`/api/journal/daily?limit=${limit}`),
  review: (date?: string) => get<any>(`/api/journal/review${date ? `?date=${date}` : ''}`),
  performance: (days = 180) => get<any>(`/api/journal/performance?limit_days=${days}`),

  // ---- news -------------------------------------------------------------
  news: (keys = '', refresh = false) =>
    get<any>(`/api/news?refresh=${refresh}${keys ? `&instrument_keys=${encodeURIComponent(keys)}` : ''}`),
  newsContext: () => get<any>('/api/news/context'),
}

export default api
