import { useState } from 'react'
import api from '../services/api'
import { Card, ErrorBox, Loading, Pill, Tabs, money, num, pct, useAsync } from '../components/common'

type Tab = 'champion' | 'hypotheses' | 'experiments' | 'memory' | 'budget'

function DeltaRow({ label, delta, champion, challenger, format = 'num' }: any) {
  const fmt = (v: any) => (v === undefined || v === null ? '—' : format === 'pct' ? pct(v) : num(v, 3))
  const good = delta > 0
  return (
    <tr>
      <td>{label}</td>
      <td className="num muted">{fmt(champion)}</td>
      <td className="num">
        <strong>{fmt(challenger)}</strong>
      </td>
      <td className={`num ${delta > 0 ? 'good' : delta < 0 ? 'bad' : ''}`}>{delta > 0 ? '+' : ''}{fmt(delta)}</td>
    </tr>
  )
}

export default function Research() {
  const [tab, setTab] = useState<Tab>('champion')
  const champion = useAsync(() => api.champion(), [])
  const versions = useAsync(() => api.versions(50), [])
  const hypotheses = useAsync(() => api.hypotheses(50), [])
  const experiments = useAsync(() => api.experiments(50), [])
  const learnings = useAsync(() => api.learnings('', 50), [])
  const budget = useAsync(() => api.multipleTesting(), [])
  const stats = useAsync(() => api.journalStats(), [])
  const [busy, setBusy] = useState('')
  const [msg, setMsg] = useState('')
  const [selected, setSelected] = useState<any>(null)
  const [ask, setAsk] = useState('')
  const [answer, setAnswer] = useState<any>(null)

  const generate = async () => {
    setBusy('hyp')
    setMsg('')
    try {
      const result = await api.generateHypotheses(5)
      setMsg(
        result.count
          ? `Generated ${result.count} hypothesis/hypotheses from ${result.journal_trades_analysed} journaled trades.`
          : `Not enough journal data yet: ${result.journal_trades_analysed} trades analysed (25 needed for a meaningful segment).`,
      )
      hypotheses.reload()
    } catch (e: any) {
      setMsg('Failed: ' + (e?.message || e))
    } finally {
      setBusy('')
    }
  }

  const runHypothesis = async (hypothesis: any, universe: string[]) => {
    if (!window.confirm(
      `Run an experiment for ${hypothesis.hypothesis_id}?\n\n` +
        `Variable: ${hypothesis.variable}\n${hypothesis.old_value} → ${hypothesis.new_value}\n\n` +
        `This creates an immutable challenger and runs backtest → walk-forward → robustness → ` +
        `Monte Carlo → holdout. It can take a few minutes.`,
    ))
      return
    setBusy(hypothesis.hypothesis_id)
    setMsg('')
    try {
      const result = await api.runExperiment({
        hypothesis_id: hypothesis.hypothesis_id,
        variable: hypothesis.variable,
        new_value: parseValue(hypothesis.new_value),
        statement: hypothesis.statement,
        universe: universe.slice(0, 15),
        n_windows: 4,
        max_universe: 15,
        run_robustness: true,
        run_monte_carlo: true,
      })
      setSelected(result)
      setMsg(`Experiment ${result.experiment_id} finished with status ${result.status}.`)
      experiments.reload()
      versions.reload()
    } catch (e: any) {
      setMsg('Experiment failed: ' + (e?.message || e))
    } finally {
      setBusy('')
    }
  }

  const promote = async (experimentId: string) => {
    if (!window.confirm('Promote this challenger to CHAMPION?\n\nThe live strategy will change. This is only permitted after the deterministic promotion gate has passed.')) return
    setBusy('promote')
    try {
      const result = await api.promote(experimentId)
      setMsg(`Promoted ${result.promoted.version}. Previous champion: ${result.previous_champion || 'none'}.`)
      champion.reload()
      versions.reload()
      experiments.reload()
    } catch (e: any) {
      setMsg('Promotion refused: ' + (e?.message || e))
    } finally {
      setBusy('')
    }
  }

  const askMemory = async () => {
    if (!ask.trim()) return
    setBusy('ask')
    try {
      setAnswer(await api.askMemory(ask))
    } finally {
      setBusy('')
    }
  }

  return (
    <div>
      <Tabs
        tabs={[
          { id: 'champion', label: 'Champion & versions' },
          { id: 'hypotheses', label: `Hypotheses (${hypotheses.data?.count ?? 0})` },
          { id: 'experiments', label: `Experiments (${experiments.data?.count ?? 0})` },
          { id: 'memory', label: `Research memory (${learnings.data?.stats?.total_learnings ?? 0})` },
          { id: 'budget', label: 'Multiple testing' },
        ]}
        active={tab}
        onChange={setTab}
      />

      {msg && <div className={msg.toLowerCase().includes('fail') || msg.toLowerCase().includes('refus') ? 'error-box' : 'ok-box'}>{msg}</div>}

      {tab === 'champion' && (
        <>
          <Card title="Current champion">
            {champion.loading && !champion.data && <Loading what="champion" />}
            <div className="grid c2">
              <div>
                <div className="kv">
                  <div>Version</div>
                  <div><strong>{champion.data?.champion?.version ?? '—'}</strong></div>
                  <div>Family</div>
                  <div>{champion.data?.strategy_family ?? champion.data?.champion?.family}</div>
                  <div>Status</div>
                  <div><Pill value={champion.data?.champion?.status} /></div>
                  <div>Config hash</div>
                  <div className="muted">{champion.data?.config_hash?.slice(0, 16)}…</div>
                  <div>Published</div>
                  <div>{champion.data?.champion?.created_at?.slice(0, 19).replace('T', ' ')}</div>
                  <div>Promoted</div>
                  <div>{champion.data?.champion?.promoted_at?.slice(0, 19).replace('T', ' ') || 'baseline'}</div>
                  <div>Reason</div>
                  <div className="muted" style={{ textAlign: 'left', maxWidth: 420 }}>
                    {champion.data?.champion?.reason_for_change}
                  </div>
                </div>
              </div>
              <div>
                <h3>Key levers of the champion</h3>
                <div className="kv">
                  <div>Opening range</div>
                  <div>{champion.data?.config?.opening_range?.duration_minutes} min</div>
                  <div>Min relative volume</div>
                  <div>{champion.data?.config?.long?.min_rvol}x</div>
                  <div>Require retest</div>
                  <div>{String(champion.data?.config?.long?.require_retest)}</div>
                  <div>Entry window</div>
                  <div>
                    {champion.data?.config?.long?.entry_window_start_minutes}–
                    {champion.data?.config?.long?.entry_window_end_minutes} min
                  </div>
                  <div>Stop model</div>
                  <div>{champion.data?.config?.stops?.model}</div>
                  <div>Trailing</div>
                  <div>{champion.data?.config?.trailing?.model} @ {champion.data?.config?.trailing?.atr_multiplier}x ATR</div>
                  <div>Target 1 / 2</div>
                  <div>
                    {champion.data?.config?.targets?.r_multiple_t1}R / {champion.data?.config?.targets?.r_multiple_t2}R
                  </div>
                  <div>Allowed regimes (long)</div>
                  <div style={{ textAlign: 'left' }}>{(champion.data?.config?.long?.allowed_regimes || []).join(', ')}</div>
                </div>
              </div>
            </div>
          </Card>

          <Card title="All published strategy versions (immutable)">
            {versions.data?.versions?.length ? (
              <div className="scroll-x dd">
                <table>
                  <thead>
                    <tr>
                      <th>Version</th>
                      <th>Status</th>
                      <th>Parent</th>
                      <th>Variable changed</th>
                      <th>Change</th>
                      <th>Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {versions.data.versions.map((v: any) => (
                      <tr key={v.version}>
                        <td><strong>{v.version}</strong></td>
                        <td><Pill value={v.status} /></td>
                        <td className="muted">{v.parent_version || '—'}</td>
                        <td>{v.variable_changed || 'baseline'}</td>
                        <td className="muted">{v.variable_changed ? `${v.old_value} → ${v.new_value}` : '—'}</td>
                        <td className="muted">{v.created_at?.slice(0, 19).replace('T', ' ')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="muted">No versions yet.</div>
            )}
            <div className="note">
              A published version can never be edited — only superseded by a new immutable version. That is what
              makes every P&amp;L number attributable to an auditable strategy definition.
            </div>
          </Card>
        </>
      )}

      {tab === 'hypotheses' && (
        <>
          <Card
            title="Research hypotheses"
            actions={
              <button onClick={generate} disabled={busy === 'hyp'}>
                {busy === 'hyp' ? 'Analysing journal…' : 'Generate hypotheses from the journal'}
              </button>
            }
          >
            <div className="note">
              Hypotheses come from measured segments of your own trade journal — never from a language model guessing.
              Journaled trades: <strong>{stats.data?.strategy_trades ?? 0}</strong> (25 needed before a segment is
              statistically meaningful). Win rate {pct(stats.data?.win_rate)}, expectancy {num(stats.data?.expectancy_r, 3)}R.
            </div>
            {hypotheses.data?.hypotheses?.length ? (
              hypotheses.data.hypotheses.map((h: any) => (
                <div key={h.hypothesis_id} className="card" style={{ background: 'var(--panel-2)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                    <strong>
                      {h.hypothesis_id} · {h.variable}
                    </strong>
                    <span>
                      <Pill value={h.status} />
                    </span>
                  </div>
                  <p style={{ margin: '8px 0' }}>{h.statement}</p>
                  <div className="kv" style={{ maxWidth: 620 }}>
                    <div>Proposed change</div>
                    <div>{h.old_value} → {h.new_value}</div>
                    <div>Sample size</div>
                    <div>{h.sample_size} trades</div>
                    <div>Confidence</div>
                    <div>{num(h.confidence, 2)}</div>
                    <div>Segment expectancy</div>
                    <div>{num(h.evidence?.segment_expectancy_r, 3)}R</div>
                    <div>Baseline expectancy</div>
                    <div>{num(h.evidence?.baseline_expectancy_r, 3)}R</div>
                    <div>p-value</div>
                    <div>{h.evidence?.p_value === null || h.evidence?.p_value === undefined ? '—' : num(h.evidence.p_value, 4)}</div>
                  </div>
                  {h.expected_mechanism && (
                    <p className="muted" style={{ marginBottom: 4 }}>
                      <strong>Expected mechanism:</strong> {h.expected_mechanism}
                    </p>
                  )}
                  {h.potential_downside && (
                    <p className="muted" style={{ marginBottom: 8 }}>
                      <strong>Potential downside:</strong> {h.potential_downside}
                    </p>
                  )}
                  <button
                    onClick={() =>
                      runHypothesis(
                        h,
                        champion.data?.config ? [] : [],
                      )
                    }
                    disabled={busy === h.hypothesis_id}
                  >
                    {busy === h.hypothesis_id ? 'Running experiment…' : 'Run one-variable experiment'}
                  </button>
                </div>
              ))
            ) : (
              <div className="muted">
                No hypotheses yet. Analyse the journal once you have journaled trades, or run the weekly research job.
              </div>
            )}
          </Card>
        </>
      )}

      {tab === 'experiments' && (
        <>
          {selected && (
            <Card title={`Experiment ${selected.experiment_id} — ${selected.status}`}>
              <div className="grid c2">
                <div>
                  <h3>What changed</h3>
                  <div className="kv">
                    <div>Parent (champion)</div>
                    <div>{selected.champion_version || '—'}</div>
                    <div>Challenger</div>
                    <div><strong>{selected.challenger_version}</strong></div>
                    <div>Variable</div>
                    <div>{selected.variable_changed}</div>
                    <div>Change</div>
                    <div>{String(selected.old_value)} → {String(selected.new_value)}</div>
                  </div>
                  <h3 style={{ marginTop: 12 }}>Champion vs challenger</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th className="num">Champion</th>
                        <th className="num">Challenger</th>
                        <th className="num">Delta</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.delta_metrics?.expectancy_r !== undefined && (
                        <DeltaRow label="Expectancy (R)" delta={selected.delta_metrics.expectancy_r} champion={selected.delta_metrics.expectancy_r_champion} challenger={selected.delta_metrics.expectancy_r_challenger} />
                      )}
                      {selected.delta_metrics?.profit_factor !== undefined && (
                        <DeltaRow label="Profit factor" delta={selected.delta_metrics.profit_factor} champion={selected.delta_metrics.profit_factor_champion} challenger={selected.delta_metrics.profit_factor_challenger} />
                      )}
                      {selected.delta_metrics?.sortino !== undefined && (
                        <DeltaRow label="Sortino" delta={selected.delta_metrics.sortino} champion={selected.delta_metrics.sortino_champion} challenger={selected.delta_metrics.sortino_challenger} />
                      )}
                      {selected.delta_metrics?.max_drawdown_pct !== undefined && (
                        <DeltaRow label="Max drawdown" delta={selected.delta_metrics.max_drawdown_pct} champion={selected.delta_metrics.max_drawdown_pct_champion} challenger={selected.delta_metrics.max_drawdown_pct_challenger} format="pct" />
                      )}
                    </tbody>
                  </table>
                </div>
                <div>
                  <h3>Validation pipeline</h3>
                  <div className="kv">
                    <div>Walk-forward</div>
                    <div>
                      {selected.walk_forward
                        ? `${pct(selected.walk_forward.profitable_fraction)} of ${selected.walk_forward.total_windows} windows profitable ${selected.walk_forward.passed ? '✅' : '❌'}`
                        : '—'}
                    </div>
                    <div>Robustness</div>
                    <div>
                      {selected.robustness
                        ? `${selected.robustness.passed_scenarios}/${selected.robustness.total_scenarios} stress scenarios ${selected.robustness.passed ? '✅' : '❌'}`
                        : '—'}
                    </div>
                    <div>Monte Carlo ruin probability</div>
                    <div>{selected.monte_carlo ? pct(selected.monte_carlo.ruin_probability) : '—'}</div>
                    <div>Holdout</div>
                    <div>
                      {selected.holdout
                        ? `${selected.holdout.trades} trades, PF ${num(selected.holdout.profit_factor, 2)} ${selected.holdout.accepted ? '✅' : '❌'}`
                        : '—'}
                    </div>
                    <div>Statistical significance</div>
                    <div>
                      {selected.significance
                        ? `P(challenger better) = ${num(selected.significance.probability_better, 2)}`
                        : '—'}
                    </div>
                    <div>Promotion gate</div>
                    <div>{selected.gate ? (selected.gate.passed ? 'PASSED ✅' : `FAILED ❌ (${selected.gate.blocking.length} blocking)`) : '—'}</div>
                  </div>
                  {selected.gate?.reasons?.length > 0 && (
                    <ul className="reasons">
                      {selected.gate.reasons.slice(0, 8).map((r: string, i: number) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  )}
                  {selected.status === 'PAPER_VALIDATION' && (
                    <button onClick={() => promote(selected.experiment_id)} disabled={busy === 'promote'} style={{ marginTop: 10 }}>
                      Promote to champion
                    </button>
                  )}
                  <div style={{ marginTop: 12 }}>
                    <h3>Step log</h3>
                    <pre>{selected.steps?.map((s: any) => `${s.ts?.slice(11, 19)}  ${s.step}: ${s.detail}`).join('\n')}</pre>
                  </div>
                </div>
              </div>
            </Card>
          )}

          <Card title="Experiment history">
            {experiments.data?.experiments?.length ? (
              <div className="scroll-x dd">
                <table>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Challenger</th>
                      <th>Variable</th>
                      <th>Change</th>
                      <th className="num">Trades</th>
                      <th className="num">Exp. R</th>
                      <th className="num">PF</th>
                      <th className="num">Max DD</th>
                      <th>WF</th>
                      <th>Holdout</th>
                      <th>Status</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {experiments.data.experiments.map((e: any) => (
                      <tr key={e.experiment_id}>
                        <td className="muted">{e.experiment_id}</td>
                        <td>{e.candidate_strategy}</td>
                        <td>{e.variable_changed}</td>
                        <td className="muted">{e.old_value} → {e.new_value}</td>
                        <td className="num">{e.trade_count}</td>
                        <td className={`num ${(e.expectancy_r ?? 0) >= 0 ? 'good' : 'bad'}`}>{num(e.expectancy_r, 3)}</td>
                        <td className="num">{num(e.profit_factor, 2)}</td>
                        <td className="num bad">{pct(e.max_drawdown_pct)}</td>
                        <td>{e.walk_forward_summary ? (e.walk_forward_summary.passed ? '✅' : '❌') : '—'}</td>
                        <td>{e.holdout_summary ? (e.holdout_summary.accepted ? '✅' : '❌') : '—'}</td>
                        <td><Pill value={e.status} /></td>
                        <td>
                          <button className="ghost" onClick={() => setSelected(e)}>
                            Detail
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="muted">
                No experiments yet. Generate a hypothesis from the journal and run it — that is the self-improvement
                loop.
              </div>
            )}
          </Card>
        </>
      )}

      {tab === 'memory' && (
        <>
          <Card title="Ask the research memory">
            <div className="note">
              This is a retrieval function over stored, dated records — not a generative model. It returns only what the
              system has actually measured, and reports nothing when it has no evidence.
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <input
                value={ask}
                placeholder='e.g. "what have we learned about gap-up mornings?" or "why was ORB_v1.1.0 rejected?"'
                onChange={(e) => setAsk(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && askMemory()}
              />
              <button onClick={askMemory} disabled={busy === 'ask'}>
                Ask
              </button>
            </div>
            {answer && (
              <div style={{ marginTop: 12 }}>
                {!answer.found && <div className="note warn">No matching record. The system has not measured this yet.</div>}
                {answer.matching_learnings?.length > 0 && (
                  <>
                    <h3>Matching learnings</h3>
                    {answer.matching_learnings.map((l: any) => (
                      <div key={l.learning_id} className="note">
                        <strong>{l.learning_id}</strong> — {l.finding}
                        <div className="muted">
                          decision {l.decision || '—'} · result {l.result || '—'} · sample {l.sample_size} · confidence{' '}
                          {num(l.confidence, 2)}
                        </div>
                      </div>
                    ))}
                  </>
                )}
                {answer.matching_experiments?.length > 0 && (
                  <>
                    <h3>Matching experiments</h3>
                    <div className="kv">
                      {answer.matching_experiments.map((e: any) => (
                        <div key={e.experiment_id} style={{ display: 'contents' }}>
                          <div>{e.experiment_id} · {e.change}</div>
                          <div>{e.status}</div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </Card>

          <Card title="Research memory (learnings)">
            {learnings.data?.learnings?.length ? (
              <div className="scroll-x dd">
                <table>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Finding</th>
                      <th className="num">Sample</th>
                      <th className="num">Confidence</th>
                      <th>Decision</th>
                      <th>Result</th>
                      <th>Strategy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {learnings.data.learnings.map((l: any) => (
                      <tr key={l.learning_id}>
                        <td className="muted">{l.learning_id}</td>
                        <td style={{ whiteSpace: 'normal', maxWidth: 520 }}>{l.finding}</td>
                        <td className="num">{l.sample_size}</td>
                        <td className="num">{num(l.confidence, 2)}</td>
                        <td>{l.decision || '—'}</td>
                        <td>{l.result || '—'}</td>
                        <td className="muted">{l.strategy_version}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="muted">No learnings recorded yet.</div>
            )}
          </Card>
        </>
      )}

      {tab === 'budget' && (
        <Card title="Multiple-testing control">
          {budget.data ? (
            <>
              <div className="grid c3">
                <div className="stat"><div className="k">Total trials recorded</div><div className="v">{budget.data.total_trials}</div></div>
                <div className="stat"><div className="k">Trials this quarter</div><div className="v">{budget.data.trials_last_quarter}</div><div className="s">budget {budget.data.max_trials_per_quarter}</div></div>
                <div className="stat"><div className="k">Budget used</div><div className={`v ${budget.data.budget_used_pct > 0.8 ? 'warn' : ''}`}>{pct(budget.data.budget_used_pct)}</div></div>
              </div>
              <div className="note warn">{budget.data.note}</div>
              <h3>Recent trials</h3>
              <div className="scroll-x dd">
                <table>
                  <thead>
                    <tr>
                      <th>When</th>
                      <th>Kind</th>
                      <th>Strategy</th>
                      <th>Variable</th>
                      <th>Value</th>
                      <th className="num">Expectancy R</th>
                    </tr>
                  </thead>
                  <tbody>
                    {budget.data.recent.map((t: any, i: number) => (
                      <tr key={i}>
                        <td className="muted">{t.ts?.slice(0, 19).replace('T', ' ')}</td>
                        <td>{t.kind}</td>
                        <td>{t.strategy_version}</td>
                        <td>{t.variable}</td>
                        <td className="muted">{t.value}</td>
                        <td className="num">{num(t.metric_value, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <Loading what="the ledger" />
          )}
        </Card>
      )}
    </div>
  )
}

function parseValue(raw: any) {
  if (typeof raw !== 'string') return raw
  const trimmed = raw.trim()
  if (trimmed === 'true') return true
  if (trimmed === 'false') return false
  if (trimmed === 'null') return null
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed)
  if (trimmed.startsWith('[')) {
    try {
      return JSON.parse(trimmed.replace(/'/g, '"'))
    } catch {
      return trimmed
    }
  }
  return trimmed
}
