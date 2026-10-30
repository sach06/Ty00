import { useState, useEffect, useCallback } from 'react'
import MetricCard from './components/MetricCard'
import PositionsTable from './components/PositionsTable'
import TradesTable from './components/TradesTable'
import LogFeed from './components/LogFeed'
import EquityChart from './components/EquityChart'
import BudgetPanel from './components/BudgetPanel'
import SentimentPanel from './components/SentimentPanel'
import RLPanel from './components/RLPanel'
import CountdownPanel from './components/CountdownPanel'
import SprintControlPanel from './components/SprintControlPanel'

const STATE_POLL_MS  = 5_000
const TRADES_POLL_MS = 10_000
const LOGS_POLL_MS   = 5_000

async function fetchJSON(url) {
  const res = await fetch(url)
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || `${res.status}`)
  return data
}

export default function App() {
  const [state,  setState]  = useState(null)
  const [trades, setTrades] = useState([])
  const [logs,   setLogs]   = useState([])
  const [error,  setError]  = useState(null)
  const [lastOk, setLastOk] = useState(null)
  const [now, setNow] = useState(new Date())
  const [realized, setRealized] = useState(null)
  const [countdownEpoch, setCountdownEpoch] = useState(null)
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('ty00-theme')
    return saved === 'light' || saved === 'dark'
      ? saved
      : window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
  })
  const [username, setUsername] = useState(() => localStorage.getItem('ty00-username') || 'Trader')
  const [nameDraft, setNameDraft] = useState('')
  const [editingName, setEditingName] = useState(false)

  const loadState = useCallback(async () => {
    try {
      const data = await fetchJSON('/api/state')
      if (data.error) { setError(data.error); return }
      setState(data)
      setError(null)
      setLastOk(new Date())
    } catch (e) {
      setError(e instanceof Error && e.message !== 'Failed to fetch'
        ? e.message
        : 'Cannot reach API server (python server.py running?)')
    }
  }, [])

  const loadTrades = useCallback(async () => {
    try { setTrades(await fetchJSON('/api/trades')) } catch {}
  }, [])

  const loadLogs = useCallback(async () => {
    try { setLogs(await fetchJSON('/api/logs')) } catch {}
  }, [])

  const loadRealized = useCallback(async () => {
    try { setRealized(await fetchJSON('/api/realized-pnl')) } catch {}
  }, [])

  const loadCountdown = useCallback(async () => {
    try {
      const data = await fetchJSON('/api/countdown')
      if (data.epoch != null) setCountdownEpoch(data.epoch)
    } catch {}
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('ty00-theme', theme)
  }, [theme])

  useEffect(() => {
    loadState();  loadTrades();  loadLogs(); loadRealized(); loadCountdown()
    const t1 = setInterval(loadState,  STATE_POLL_MS)
    const t2 = setInterval(loadTrades, TRADES_POLL_MS)
    const t3 = setInterval(loadLogs,   LOGS_POLL_MS)
    const t5 = setInterval(loadRealized, STATE_POLL_MS)
    const t4 = setInterval(() => setNow(new Date()), 1000)
    return () => { clearInterval(t1); clearInterval(t2); clearInterval(t3); clearInterval(t4); clearInterval(t5) }
  }, [loadState, loadTrades, loadLogs, loadRealized, loadCountdown])

  // ── Status badge ────────────────────────────────────────────────────────────
  const statusBadge = () => {
    if (error)             return <Badge color="red">OFFLINE</Badge>
    if (!state)            return <Badge color="yellow">CONNECTING…</Badge>
    if (state.halted)      return <Badge color="red" pulse>HALTED</Badge>
    if (state.dry_run)     return <Badge color="sky">DRY-RUN</Badge>
    return                        <Badge color="emerald" pulse>LIVE</Badge>
  }

  // ── Derived metrics ─────────────────────────────────────────────────────────
  const equity    = state?.equity ?? '—'
  const pnl       = state?.daily_pnl ?? null
  const drawdown  = state?.drawdown_pct ?? null
  const winRate   = state?.win_rate ?? null
  const totalTr   = state?.total_trades ?? '—'
  const winTr     = state?.winning_trades ?? '—'
  const realizedPnlToday = Number(realized?.realized_pnl_today ?? 0)
  const realizedCount = Number(realized?.transaction_count ?? 0)
  const hasRealized = realized !== null && realizedCount > 0

  function saveUsername(event) {
    event.preventDefault()
    const nextName = nameDraft.trim().slice(0, 32)
    if (nextName) {
      setUsername(nextName)
      localStorage.setItem('ty00-username', nextName)
    }
    setEditingName(false)
  }

  const pnlColor  = pnl === null ? 'text-slate-100' : pnl >= 0 ? 'text-emerald-400' : 'text-red-400'
  const ddColor   = drawdown === null ? 'text-slate-100' : drawdown < 3 ? 'text-emerald-400' : drawdown < 6 ? 'text-yellow-400' : 'text-red-400'

  return (
    <div className="app-shell" data-theme={theme}>
      <header className="app-topbar">
        <div className="topbar-inner">
          <a className="brand-lockup" href="#overview" aria-label="Ty00 dashboard home">
            <span className="brand-mark" aria-hidden="true">T</span>
            <span className="brand-copy">
              <strong>TY00</strong>
              <small>TRADING DESK</small>
            </span>
          </a>

          <div className="topbar-center">
            {statusBadge()}
            <span className="topbar-divider" />
            <span className="connection-copy">
              <span className={`connection-dot ${error ? 'is-offline' : state ? 'is-online' : ''}`} />
              {lastOk
                ? `Synced ${lastOk.toLocaleTimeString('en-GB', { hour12: false })}`
                : error ? 'API unavailable' : 'Connecting'}
            </span>
          </div>

          <div className="topbar-actions">
            <div className="theme-switch" role="group" aria-label="Color theme">
              <button
                type="button"
                aria-pressed={theme === 'light'}
                onClick={() => setTheme('light')}
              >Light</button>
              <button
                type="button"
                aria-pressed={theme === 'dark'}
                onClick={() => setTheme('dark')}
              >Dark</button>
            </div>
            <div className="profile-area">
              {editingName ? (
                <form className="profile-edit" onSubmit={saveUsername}>
                  <input
                    aria-label="Display name"
                    autoFocus
                    maxLength={32}
                    value={nameDraft}
                    onChange={event => setNameDraft(event.target.value)}
                  />
                  <button type="submit" className="profile-save">Save</button>
                  <button type="button" className="profile-cancel" onClick={() => setEditingName(false)}>Cancel</button>
                </form>
              ) : (
                <>
                  <span className="profile-avatar" aria-hidden="true">{username.slice(0, 1).toUpperCase()}</span>
                  <span className="profile-copy">
                    <strong>{username}</strong>
                    <small>Local profile</small>
                  </span>
                  <button
                    type="button"
                    className="profile-edit-button"
                    aria-label="Edit display name"
                    title="Edit display name"
                    onClick={() => { setNameDraft(username); setEditingName(true) }}
                  >Edit</button>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      <main id="overview" className="dashboard-content">
        <div className="page-heading">
          <div>
            <p className="eyebrow">PORTFOLIO MONITOR</p>
            <h1>Overview</h1>
          </div>
          <div className="market-clock">
            <span className="market-clock-label">UTC</span>
            <strong>{now.toLocaleTimeString('en-GB', { hour12: false })}</strong>
            <span>{now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' })}</span>
          </div>
        </div>

        {error && (
          <div className="error-banner" role="alert">
            <span className="error-indicator" aria-hidden="true">!</span>
            <div>
              <strong>Connection unavailable</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        <section className="metric-grid" aria-label="Portfolio performance">
        <MetricCard
          label="Equity"
          value={state ? `€${equity.toFixed(2)}` : '—'}
          sub={state ? `Peak €${state.peak_equity?.toFixed(2)}` : undefined}
        />
        <MetricCard
          label="Daily P&L"
          value={pnl !== null ? `${pnl >= 0 ? '+' : ''}€${pnl.toFixed(2)}` : '—'}
          color={pnlColor}
          sub={state?.consecutive_losses > 0 ? `${state.consecutive_losses} consec. losses` : undefined}
        />
        <MetricCard
          label="Drawdown"
          value={drawdown !== null ? `${drawdown.toFixed(2)}%` : '—'}
          color={ddColor}
          sub="7% = kill switch"
        />
        <MetricCard
          label="Win Rate"
          value={winRate !== null ? `${winRate.toFixed(1)}%` : '—'}
          color={winRate !== null && winRate >= 55 ? 'text-emerald-400' : 'text-slate-100'}
          sub={`${winTr} / ${totalTr} trades`}
        />
        <MetricCard
          label="Realized P&L (Broker)"
          value={hasRealized ? `${realizedPnlToday >= 0 ? '+' : ''}€${realizedPnlToday.toFixed(2)}` : '—'}
          color={hasRealized ? (realizedPnlToday >= 0 ? 'text-emerald-400' : 'text-red-400') : 'text-slate-300'}
          sub={hasRealized ? `Today UTC · ${realizedCount} broker tx` : 'Broker did not return realized P&L yet'}
        />
        </section>

        <div className="workstation-grid">
          <section className="workstation-main" aria-label="Positions and performance">
            {state?.equity_history?.length > 1 && (
              <EquityChart history={state.equity_history} />
            )}
            <PositionsTable positions={state?.open_positions ?? []} />
            <TradesTable trades={trades} />
          </section>

          <aside className="workstation-side" aria-label="Trading controls and signals">
            <BudgetPanel
              budget={state?.budget ?? null}
              used={state?.budget_used ?? 0}
              available={state?.budget_available ?? (state?.budget ?? 0)}
              onBudgetChange={newVal => setState(prev => prev ? {
                ...prev,
                budget: newVal,
                budget_available: Math.max(0, newVal - (prev.budget_used ?? 0)),
              } : prev)}
            />
            <SentimentPanel sentiment={state?.sentiment} />
            <CountdownPanel epoch={countdownEpoch} onEpochChange={setCountdownEpoch} />
            <SprintControlPanel />
            <RLPanel rl={state?.rl} />
          </aside>
        </div>

        <LogFeed lines={logs} />

        <footer className="dashboard-footer">
          <span>RSI(14) · Bollinger Bands(20, 2σ)</span>
          <span>Session 08:00–16:00 UTC</span>
          <span>Risk 1% / trade · Drawdown limit 7%</span>
        </footer>
      </main>
    </div>
  )
}

// ── Badge helper ─────────────────────────────────────────────────────────────
function Badge({ color = 'slate', pulse = false, children }) {
  const colors = {
    emerald: 'bg-emerald-500/20 text-emerald-400 border-emerald-600',
    sky:     'bg-sky-500/20     text-sky-400     border-sky-600',
    yellow:  'bg-yellow-500/20  text-yellow-400  border-yellow-600',
    red:     'bg-red-500/20     text-red-400     border-red-600',
    slate:   'bg-slate-700      text-slate-300   border-slate-600',
  }
  return (
    <span className={`status-badge ${colors[color]} ${pulse ? 'animate-pulse' : ''}`}>
      {children}
    </span>
  )
}
