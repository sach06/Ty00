/** News & sentiment scores per instrument */
export default function SentimentPanel({ sentiment }) {
  if (!sentiment || Object.keys(sentiment).length === 0) return null

  const entries = Object.entries(sentiment).sort((a, b) => a[0].localeCompare(b[0]))

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <span className="text-xs text-slate-400 uppercase tracking-widest block mb-3">
        News & Market Sentiment
      </span>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
        {entries.map(([epic, score]) => {
          const pct      = Math.round(score * 100)
          const absScore = Math.abs(score)
          const label    = absScore < 0.1  ? 'Neutral'
                         : score > 0       ? 'Bullish'
                         :                   'Bearish'
          const color    = absScore < 0.1  ? 'text-slate-400 border-slate-600 bg-slate-700'
                         : score > 0       ? 'text-emerald-400 border-emerald-700 bg-emerald-900/30'
                         :                   'text-red-400 border-red-700 bg-red-900/30'
          const barFill  = absScore < 0.1  ? 'bg-slate-500'
                         : score > 0       ? 'bg-emerald-500'
                         :                   'bg-red-500'

          return (
            <div key={epic} className={`rounded-lg p-3 border ${color} flex flex-col gap-1`}>
              <div className="flex justify-between items-baseline">
                <span className="text-xs font-mono font-semibold">{epic}</span>
                <span className="text-xs opacity-70">{pct > 0 ? '+' : ''}{pct}%</span>
              </div>
              <span className="text-xs opacity-80">{label}</span>
              {/* Mini bar */}
              <div className="w-full bg-slate-600/50 rounded-full h-1 overflow-hidden mt-1">
                <div
                  className={`h-1 rounded-full transition-all duration-500 ${barFill}`}
                  style={{ width: `${Math.abs(pct)}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>
      <p className="mt-2 text-xs text-slate-600">
        Sources: Yahoo Finance RSS · Reddit r/Forex · Capital.com client sentiment ·
        Refreshed every 5 min · Signals opposing strong sentiment (&gt;35%) are suppressed.
      </p>
    </div>
  )
}
