import { useState, useEffect } from 'react'

/**
 * Countdown panel backed by a Unix epoch (seconds) so it supports multi-day
 * countdowns. The +24h button extends the stored epoch; Edit lets you set a
 * new HH:MM which becomes the next occurrence from now.
 */
export default function CountdownPanel({ epoch, onEpochChange }) {
  const [now,       setNow]       = useState(Date.now())
  const [editing,   setEditing]   = useState(false)
  const [input,     setInput]     = useState('')
  const [saving,    setSaving]    = useState(false)
  const [extending, setExtending] = useState(false)
  const [err,       setErr]       = useState(null)

  // Tick every second
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])

  // ── Compute countdown ──────────────────────────────────────────────────────
  let countdownLabel = '--:--:--'
  let isPast   = false
  let color    = 'text-amber-300'
  let targetLabel = 'No target set'

  if (epoch != null) {
    const diffSec = Math.floor((epoch * 1000 - now) / 1000)
    isPast = diffSec < 0
    const absSec = Math.abs(diffSec)
    const days = Math.floor(absSec / 86400)
    const hh   = Math.floor((absSec % 86400) / 3600)
    const mm   = Math.floor((absSec % 3600) / 60)
    const ss   = absSec % 60
    const pad  = n => String(n).padStart(2, '0')
    countdownLabel = days > 0
      ? `${days}d ${pad(hh)}:${pad(mm)}:${pad(ss)}`
      : `${pad(hh)}:${pad(mm)}:${pad(ss)}`
    color = isPast ? 'text-red-400' : 'text-amber-300'
    const d = new Date(epoch * 1000)
    targetLabel = `Target ${pad(d.getDate())}/${pad(d.getMonth() + 1)} ${pad(d.getHours())}:${pad(d.getMinutes())}`
  }

  // ── Edit helpers ───────────────────────────────────────────────────────────
  function startEdit() {
    let val = '18:40'
    if (epoch != null) {
      const d = new Date(epoch * 1000)
      val = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
    }
    setInput(val)
    setErr(null)
    setEditing(true)
  }

  function cancelEdit() { setEditing(false); setErr(null) }

  async function save() {
    const val = input.trim()
    if (!/^\d{1,2}:\d{2}$/.test(val)) { setErr('Enter time as HH:MM (e.g. 18:40)'); return }
    setSaving(true); setErr(null)
    try {
      const res  = await fetch('/api/countdown', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target: val }),
      })
      const data = await res.json()
      if (!res.ok) { setErr(data.error || 'Failed to save'); return }
      onEpochChange?.(data.epoch)
      setEditing(false)
    } catch {
      setErr('Network error – is server.py running?')
    } finally { setSaving(false) }
  }

  async function extend() {
    setExtending(true); setErr(null)
    try {
      const res  = await fetch('/api/countdown/extend', { method: 'POST' })
      const data = await res.json()
      if (!res.ok) { setErr(data.error || 'Failed to extend'); return }
      onEpochChange?.(data.epoch)
    } catch {
      setErr('Network error – is server.py running?')
    } finally { setExtending(false) }
  }

  function handleKey(e) {
    if (e.key === 'Enter')  save()
    if (e.key === 'Escape') cancelEdit()
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs text-slate-400 uppercase tracking-widest">Countdown</span>

        {!editing ? (
          <div className="flex items-center gap-2">
            <button
              onClick={extend}
              disabled={extending}
              className="text-xs text-emerald-400 hover:text-emerald-300 border border-emerald-700 hover:border-emerald-500 rounded px-2 py-0.5 transition-colors disabled:opacity-50"
            >
              {extending ? '…' : '+24h'}
            </button>
            <button
              onClick={startEdit}
              className="text-xs text-sky-400 hover:text-sky-300 border border-sky-700 hover:border-sky-500 rounded px-2 py-0.5 transition-colors"
            >
              Edit
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <input
              type="text"
              placeholder="HH:MM"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              autoFocus
              className="w-20 bg-slate-700 border border-sky-600 rounded px-2 py-0.5 text-sm text-slate-100 focus:outline-none focus:ring-1 focus:ring-sky-500"
            />
            <button
              onClick={save}
              disabled={saving}
              className="text-xs bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white rounded px-2 py-0.5 transition-colors"
            >
              {saving ? '…' : 'Save'}
            </button>
            <button
              onClick={cancelEdit}
              className="text-xs text-slate-400 hover:text-slate-200 transition-colors"
            >
              ✕
            </button>
          </div>
        )}
      </div>

      {err && <p className="text-xs text-red-400 mb-2">{err}</p>}

      <div className="flex flex-col items-center py-2">
        <span className={`text-3xl font-mono font-semibold tabular-nums ${color}`}>
          {isPast ? `-${countdownLabel}` : countdownLabel}
        </span>
        <span className="text-xs text-slate-500 mt-1">
          {targetLabel}{isPast ? ' · passed' : ''}
        </span>
      </div>
    </div>
  )
}
