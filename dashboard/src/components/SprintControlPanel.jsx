import { useState, useEffect, useCallback } from 'react'

const SPRINTS = [
  {
    key:     'fx',
    label:   'FX Sprint',
    sub:     'EURUSD · GBPUSD',
    color:   'sky',
  },
  {
    key:     'oil',
    label:   'Crude Oil Sprint',
    sub:     'OIL_CRUDE',
    color:   'amber',
  },
  {
    key:     'crypto',
    label:   'Crypto Sprint',
    sub:     'BTCUSD · ETHUSD · SOLUSD',
    color:   'violet',
  },
]

const COLOR_MAP = {
  sky:    { ring: 'ring-sky-600',    dot: 'bg-sky-400',    btn: 'bg-sky-600 hover:bg-sky-500',    badge: 'bg-sky-500/20 text-sky-300 border-sky-600' },
  amber:  { ring: 'ring-amber-600',  dot: 'bg-amber-400',  btn: 'bg-amber-600 hover:bg-amber-500',  badge: 'bg-amber-500/20 text-amber-300 border-amber-600' },
  violet: { ring: 'ring-violet-600', dot: 'bg-violet-400', btn: 'bg-violet-600 hover:bg-violet-500', badge: 'bg-violet-500/20 text-violet-300 border-violet-600' },
}

async function apiCall(path, body) {
  const res = await fetch(path, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  return res.json()
}

export default function SprintControlPanel() {
  const [status,   setStatus]   = useState({ fx: {running:false}, oil: {running:false}, crypto: {running:false} })
  const [minutes,  setMinutes]  = useState({ fx: 120, oil: 120, crypto: 120 })
  const [loading,  setLoading]  = useState({})
  const [feedback, setFeedback] = useState({})

  const loadStatus = useCallback(async () => {
    try {
      const data = await fetch('/api/sprint/status').then(r => r.json())
      setStatus(data)
    } catch {}
  }, [])

  useEffect(() => {
    loadStatus()
    const t = setInterval(loadStatus, 4000)
    return () => clearInterval(t)
  }, [loadStatus])

  async function handleStart(key) {
    setLoading(l => ({ ...l, [key]: true }))
    setFeedback(f => ({ ...f, [key]: '' }))
    try {
      const res = await apiCall('/api/sprint/start', { sprint: key, minutes: minutes[key] })
      setFeedback(f => ({
        ...f,
        [key]: res.status === 'started'         ? `Started (pid ${res.pid})`
              : res.status === 'already_running' ? 'Already running'
              : res.error ?? 'Error',
      }))
      setTimeout(loadStatus, 800)
    } catch {
      setFeedback(f => ({ ...f, [key]: 'Server error' }))
    } finally {
      setLoading(l => ({ ...l, [key]: false }))
    }
  }

  async function handleStop(key) {
    setLoading(l => ({ ...l, [key]: true }))
    setFeedback(f => ({ ...f, [key]: '' }))
    try {
      const res = await apiCall('/api/sprint/stop', { sprint: key })
      setFeedback(f => ({
        ...f,
        [key]: res.status === 'stopped'     ? 'Stopped'
              : res.status === 'not_running' ? 'Not running'
              : res.error ?? 'Error',
      }))
      setTimeout(loadStatus, 800)
    } catch {
      setFeedback(f => ({ ...f, [key]: 'Server error' }))
    } finally {
      setLoading(l => ({ ...l, [key]: false }))
    }
  }

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 space-y-3">
      <h3 className="text-slate-200 text-sm font-semibold">Sprint Control</h3>

      {SPRINTS.map(({ key, label, sub, color }) => {
        const c       = COLOR_MAP[color]
        const running = status[key]?.running ?? false
        const busy    = loading[key] ?? false
        const note    = feedback[key] ?? ''

        return (
          <div
            key={key}
            className={`flex flex-wrap items-center gap-3 bg-slate-700/40 rounded-lg px-3 py-2.5 ring-1 ${running ? c.ring : 'ring-slate-700'}`}
          >
            {/* Status dot */}
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${running ? c.dot + ' animate-pulse' : 'bg-slate-600'}`} />
              <div className="min-w-0">
                <div className="text-xs font-medium text-slate-200 leading-tight">{label}</div>
                <div className="text-[10px] text-slate-500 leading-tight">{sub}</div>
              </div>
            </div>

            {/* Running badge or minute selector */}
            {running ? (
              <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${c.badge}`}>
                RUNNING
              </span>
            ) : (
              <div className="flex items-center gap-1.5">
                <label className="text-[10px] text-slate-500">min</label>
                <input
                  type="number"
                  min="5"
                  max="480"
                  value={minutes[key]}
                  onChange={e => setMinutes(m => ({ ...m, [key]: Math.max(5, Number(e.target.value)) }))}
                  className="w-14 bg-slate-700 border border-slate-600 rounded px-1.5 py-0.5 text-xs text-slate-200 text-right focus:outline-none focus:ring-1 focus:ring-slate-500"
                />
              </div>
            )}

            {/* Start / Stop */}
            {!running ? (
              <button
                onClick={() => handleStart(key)}
                disabled={busy}
                className={`text-xs font-medium px-3 py-1 rounded-md text-white transition-colors ${c.btn} disabled:opacity-40`}
              >
                {busy ? '…' : 'Start'}
              </button>
            ) : (
              <button
                onClick={() => handleStop(key)}
                disabled={busy}
                className="text-xs font-medium px-3 py-1 rounded-md bg-red-700 hover:bg-red-600 text-white transition-colors disabled:opacity-40"
              >
                {busy ? '…' : 'Stop'}
              </button>
            )}

            {/* Feedback */}
            {note && (
              <span className="text-[10px] text-slate-400 italic w-full pl-4">{note}</span>
            )}
          </div>
        )
      })}
    </div>
  )
}
