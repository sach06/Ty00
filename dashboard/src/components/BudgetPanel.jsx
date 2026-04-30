import { useState } from 'react'

/** Budget allocation panel with progress bar and inline editor */
export default function BudgetPanel({ budget, used, available, onBudgetChange }) {
  const [editing, setEditing]   = useState(false)
  const [input,   setInput]     = useState('')
  const [saving,  setSaving]    = useState(false)
  const [err,     setErr]       = useState(null)

  if (budget == null) return null

  const pct      = budget > 0 ? Math.min(100, (used / budget) * 100) : 0
  const barColor = pct < 50 ? 'bg-emerald-500' : pct < 80 ? 'bg-yellow-500' : 'bg-red-500'
  const usedColor= pct < 50 ? 'text-emerald-400' : pct < 80 ? 'text-yellow-400' : 'text-red-400'

  function startEdit() {
    setInput(budget.toFixed(2))
    setErr(null)
    setEditing(true)
  }

  function cancelEdit() {
    setEditing(false)
    setErr(null)
  }

  async function save() {
    const val = parseFloat(input)
    if (isNaN(val) || val < 1) { setErr('Enter a valid amount ≥ €1'); return }
    setSaving(true)
    setErr(null)
    try {
      const res = await fetch('/api/budget', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ budget: val }),
      })
      const data = await res.json()
      if (!res.ok) { setErr(data.error || 'Failed to save'); return }
      onBudgetChange?.(data.budget)
      setEditing(false)
    } catch {
      setErr('Network error – is server.py running?')
    } finally {
      setSaving(false)
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter')  save()
    if (e.key === 'Escape') cancelEdit()
  }

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs text-slate-400 uppercase tracking-widest">Trading Budget</span>
        {!editing ? (
          <button
            onClick={startEdit}
            className="text-xs text-sky-400 hover:text-sky-300 border border-sky-700 hover:border-sky-500 rounded px-2 py-0.5 transition-colors"
          >
            Edit
          </button>
        ) : (
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">€</span>
            <input
              type="number"
              min="1"
              step="10"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              autoFocus
              className="w-24 bg-slate-700 border border-sky-600 rounded px-2 py-0.5 text-sm text-slate-100 focus:outline-none focus:ring-1 focus:ring-sky-500"
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

      {/* Allocation numbers */}
      <div className="flex justify-between mb-2">
        <div className="flex flex-col">
          <span className="text-xs text-slate-500">In use</span>
          <span className={`text-lg font-semibold ${usedColor}`}>€{used.toFixed(2)}</span>
        </div>
        <div className="flex flex-col items-center">
          <span className="text-xs text-slate-500">Budget</span>
          <span className="text-lg font-semibold text-slate-100">€{budget.toFixed(2)}</span>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-xs text-slate-500">Available</span>
          <span className="text-lg font-semibold text-emerald-400">€{available.toFixed(2)}</span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="mt-1 text-right text-xs text-slate-600">{pct.toFixed(1)}% of budget deployed</div>
    </div>
  )
}
