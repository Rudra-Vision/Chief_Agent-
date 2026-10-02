import { ReactNode, useCallback, useEffect, useState } from 'react'

export function Card({
  title,
  children,
  actions,
  className = '',
}: {
  title?: string
  children: ReactNode
  actions?: ReactNode
  className?: string
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          {title && <h2 style={{ margin: 0 }}>{title}</h2>}
          <div style={{ display: 'flex', gap: 8 }}>{actions}</div>
        </div>
      )}
      {children}
    </section>
  )
}

export function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: string
  value: ReactNode
  sub?: ReactNode
  tone?: 'good' | 'bad' | 'warn' | ''
}) {
  return (
    <div className="stat">
      <div className="k">{label}</div>
      <div className={`v ${tone || ''}`}>{value}</div>
      {sub !== undefined && <div className="s">{sub}</div>}
    </div>
  )
}

export function Pill({ value }: { value?: string | null }) {
  if (!value) return <span className="muted">—</span>
  return <span className={`pill ${value}`}>{value.replace(/_/g, ' ')}</span>
}

export function Loading({ what = 'data' }: { what?: string }) {
  return <div className="spinner">Loading {what}…</div>
}

export function ErrorBox({ message }: { message: string }) {
  return <div className="error-box">{message}</div>
}

export function pct(value: number | undefined | null, digits = 2): string {
  if (value === undefined || value === null || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

export function money(value: number | undefined | null, digits = 0): string {
  if (value === undefined || value === null || Number.isNaN(value)) return '—'
  const sign = value < 0 ? '-' : ''
  return `${sign}₹${Math.abs(value).toLocaleString('en-IN', { maximumFractionDigits: digits, minimumFractionDigits: digits })}`
}

export function num(value: number | undefined | null, digits = 2): string {
  if (value === undefined || value === null || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

/** Small data-fetching hook: keeps pages declarative without a heavy library. */
export function useAsync<T>(fn: () => Promise<T>, deps: any[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string>('')

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const result = await fn()
      setData(result)
      setError('')
    } catch (e: any) {
      setError(e?.message || String(e))
    } finally {
      setLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    load()
  }, [load])

  return { data, loading, error, reload: load }
}

export function Tabs<T extends string>({
  tabs,
  active,
  onChange,
}: {
  tabs: { id: T; label: string }[]
  active: T
  onChange: (id: T) => void
}) {
  return (
    <div className="tabs">
      {tabs.map((tab) => (
        <button key={tab.id} className={active === tab.id ? 'active' : ''} onClick={() => onChange(tab.id)}>
          {tab.label}
        </button>
      ))}
    </div>
  )
}

export function SimulatedBanner({ isSimulated }: { isSimulated: boolean }) {
  if (!isSimulated) return null
  return (
    <div className="note warn">
      <strong>SIMULATED DATA.</strong> No Upstox access token is configured, so the system is running on clearly
      labelled synthetic market data. Everything works end to end, but results computed from this data are
      <strong> not evidence of any trading edge</strong> — a random-walk fixture has no exploitable structure. Connect
      Upstox and download real history for meaningful research.
    </div>
  )
}
