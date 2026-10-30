/** Stat card with coloured value and optional sub-label */
export default function MetricCard({ label, value, sub, color = 'text-slate-100', pulse = false }) {
  return (
    <div className="metric-card bg-slate-800 rounded-xl p-4 border border-slate-700 flex flex-col gap-1">
      <span className="metric-label text-xs text-slate-400 uppercase">{label}</span>
      <span className={`metric-value ${color} ${pulse ? 'animate-pulse' : ''}`}>
        {value}
      </span>
      {sub && <span className="metric-sub text-xs text-slate-500">{sub}</span>}
    </div>
  )
}
