import { useState, useEffect, useCallback } from 'react'
import MetricCard from './components/MetricCard'
import PositionsTable from './components/PositionsTable'
import TradesTable from './components/TradesTable'
import LogFeed from './components/LogFeed'
import EquityChart from './components/EquityChart'
import BudgetPanel from './components/BudgetPanel'
import SentimentPanel from './components/SentimentPanel'
import RLPanel from './components/RLPanel'

const STATE_POLL_MS  = 5_000
const TRADES_POLL_MS = 10_000
const LOGS_POLL_MS   = 5_000

async function fetchJSON(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status}`)
  return res.json()
}

export default function App() {
  const [state,  setState]  = useState(null)
  const [trades, setTrades] = useState([])
  const [logs,   setLogs]   = useState([])
  const [error,  setError]  = useState(null)
  const [lastOk, setLastOk] = useState(null)
  const [now, setNow] = useState(new Date())
  const [realized, setRealized] = useState(null)

  const loadState = useCallback(async () => {
    try {
      const data = await fetchJSON('/api/state')
      if (data.error) { setError(data.error); return }
      setState(data)
      setError(null)
      setLastOk(new Date())
    } catch (e) {
      setError('Cannot reach API server (python server.py running?)')
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

  useEffect(() => {
    loadState();  loadTrades();  loadLogs(); loadRealized()
    const t1 = setInterval(loadState,  STATE_POLL_MS)
    const t2 = setInterval(loadTrades, TRADES_POLL_MS)
    const t3 = setInterval(loadLogs,   LOGS_POLL_MS)
    const t5 = setInterval(loadRealized, STATE_POLL_MS)
    const t4 = setInterval(() => setNow(new Date()), 1000)
    return () => { clearInterval(t1); clearInterval(t2); clearInterval(t3); clearInterval(t4); clearInterval(t5) }
  }, [loadState, loadTrades, loadLogs, loadRealized])

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

  const target14 = new Date(now)
  target14.setHours(14, 0, 0, 0)
  const countdownSec = Math.floor((target14.getTime() - now.getTime()) / 1000)
  const isPastTarget = countdownSec < 0
  const absSec = Math.abs(countdownSec)
  const h = Math.floor(absSec / 3600)
  const m = Math.floor((absSec % 3600) / 60)
  const s = absSec % 60
  const countdownLabel = `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`

  const pnlColor  = pnl === null ? 'text-slate-100' : pnl >= 0 ? 'text-emerald-400' : 'text-red-400'
  const ddColor   = drawdown === null ? 'text-slate-100' : drawdown < 3 ? 'text-emerald-400' : drawdown < 6 ? 'text-yellow-400' : 'text-red-400'

  return (
    <div className="min-h-screen bg-slate-900 p-4 lg:p-6">
      {/* ── Header ── */}
      <header className="flex flex-wrap items-center justify-between gap-3 mb-6">
        <div className="flex items-center gap-3">
          <span className="text-2xl font-semibold text-slate-100 tracking-tight">Ty00</span>
          <span className="text-slate-600 text-lg">│</span>
          <span className="text-slate-400 text-sm">Trading Dashboard</span>
        </div>
        <div className="flex items-center gap-4">
          {statusBadge()}
          {lastOk && (
            <span className="text-xs text-slate-600">
              Updated {lastOk.toLocaleTimeString('en-GB', { hour12: false })}
            </span>
          )}
        </div>
      </header>

      {/* ── Error banner ── */}
      {error && (
        <div className="mb-4 bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* ── Metric cards ── */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-3 mb-4">
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
          label="Countdown to 14:00"
          value={isPastTarget ? `-${countdownLabel}` : countdownLabel}
          color={isPastTarget ? 'text-red-400' : 'text-amber-300'}
          sub={isPastTarget ? 'Target passed' : 'Local time'}
        />
        <MetricCard
          label="Realized P&L (Broker)"
          value={hasRealized ? `${realizedPnlToday >= 0 ? '+' : ''}€${realizedPnlToday.toFixed(2)}` : '—'}
          color={hasRealized ? (realizedPnlToday >= 0 ? 'text-emerald-400' : 'text-red-400') : 'text-slate-300'}
          sub={hasRealized ? `Today UTC · ${realizedCount} broker tx` : 'Broker did not return realized P&L yet'}
        />
      </div>

      {/* ── Equity chart ── */}
      {state?.equity_history?.length > 1 && (
        <div className="mb-4">
          <EquityChart history={state.equity_history} />
        </div>
      )}

      {/* ── Budget + Sentiment row ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mb-4">
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
      </div>

      {/* ── RL Agent panel ── */}
      <div className="mb-4">
        <RLPanel rl={state?.rl} />
      </div>

      {/* ── Positions ── */}
      <div className="mb-4">
        <PositionsTable positions={state?.open_positions ?? []} />
      </div>

      {/* ── Trades ── */}
      <div className="mb-4">
        <TradesTable trades={trades} />
      </div>

      {/* ── Log feed ── */}
      <LogFeed lines={logs} />

      {/* ── Footer ── */}
      <footer className="mt-4 text-xs text-slate-700 text-center">
        Strategy: RSI(14) + Bollinger Bands(20,2σ) · Intraday only 08:00–16:00 UTC ·
        Max drawdown 7% · Risk 1%/trade · Budget €{state?.budget?.toFixed(0) ?? '—'} ·
        Sentiment-filtered signals
      </footer>
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
    <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${colors[color]} ${pulse ? 'animate-pulse' : ''}`}>
      {children}
    </span>
  )
}
