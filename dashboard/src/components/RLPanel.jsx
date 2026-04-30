import { useState } from 'react'

const VARIANT_COLORS = [
  'bg-slate-600',   // baseline
  'bg-sky-700',     // tighter-RSI
  'bg-sky-800',     // strictest-RSI
  'bg-indigo-700',  // relaxed-RSI
  'bg-violet-700',  // tighter-BB
  'bg-violet-800',  // wider-BB
  'bg-amber-700',   // tight-stop/wide-TP
  'bg-orange-700',  // wider-stop
  'bg-rose-700',    // quick-TP
  'bg-emerald-700', // let-winners-run
  'bg-red-700',     // aggressive
  'bg-teal-700',    // conservative
]

function qColor(q) {
  if (q === null || q === undefined) return 'text-slate-400'
  if (q > 0.5) return 'text-emerald-400'
  if (q > 0)   return 'text-green-400'
  if (q > -0.5) return 'text-yellow-400'
  return 'text-red-400'
}

export default function RLPanel({ rl }) {
  const [showAll, setShowAll] = useState(false)

  if (!rl || rl.total_updates === undefined) {
    return (
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
        <h3 className="text-slate-400 text-sm font-medium mb-2">RL Agent</h3>
        <p className="text-slate-500 text-xs">No trades learned yet — agent will self-improve after first close.</p>
      </div>
    )
  }

  const stats   = rl.action_stats ?? []
  const visited = stats.filter(s => s.visits > 0)
  const topStates = (rl.per_state ?? []).slice(0, showAll ? 15 : 5)

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-slate-200 text-sm font-semibold">RL Agent · Self-improving Strategy</h3>
        <span className="text-xs text-slate-500">{rl.total_updates} trades learned</span>
      </div>

      {/* Global best */}
      {rl.best_global_action && rl.total_updates > 0 && (
        <div className="bg-slate-700/50 rounded-lg px-3 py-2 flex items-center gap-3">
          <div className="flex-1">
            <div className="text-xs text-slate-400">Current best global strategy</div>
            <div className="text-sm font-medium text-emerald-300">{rl.best_global_action}</div>
          </div>
          <div className={`text-sm font-semibold ${qColor(rl.best_global_q)}`}>
            Q = {rl.best_global_q?.toFixed(3)}
          </div>
        </div>
      )}

      {/* Action variant stats */}
      {visited.length > 0 && (
        <div>
          <div className="text-xs text-slate-500 mb-2 uppercase tracking-wide">Variant performance (visits × avg reward)</div>
          <div className="space-y-1">
            {stats.map((s, i) => {
              const maxVisits = Math.max(...stats.map(x => x.visits), 1)
              const barW = s.visits > 0 ? Math.max(4, Math.round((s.visits / maxVisits) * 100)) : 0
              return (
                <div key={s.name} className="flex items-center gap-2 text-xs">
                  <div className="w-36 text-slate-300 truncate" title={s.name}>{s.name}</div>
                  <div className="flex-1 bg-slate-700 rounded-full h-1.5">
                    <div
                      className={`h-1.5 rounded-full ${VARIANT_COLORS[i] ?? 'bg-slate-500'}`}
                      style={{ width: `${barW}%` }}
                    />
                  </div>
                  <div className="w-10 text-right text-slate-400">{s.visits}x</div>
                  <div className={`w-14 text-right font-mono ${qColor(s.avg_q)}`}>
                    {s.visits > 0 ? (s.avg_q >= 0 ? '+' : '') + s.avg_q.toFixed(3) : '—'}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Per-state best params */}
      {topStates.length > 0 && (
        <div>
          <div className="text-xs text-slate-500 mb-2 uppercase tracking-wide">
            Best params by context ({rl.states_explored} states explored)
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-slate-300">
              <thead>
                <tr className="text-slate-500 border-b border-slate-700">
                  <th className="text-left py-1 pr-3 font-normal">Context</th>
                  <th className="text-left py-1 pr-3 font-normal">Best Variant</th>
                  <th className="text-center py-1 pr-2 font-normal">RSI</th>
                  <th className="text-center py-1 pr-2 font-normal">BB σ</th>
                  <th className="text-center py-1 pr-2 font-normal">Stop</th>
                  <th className="text-center py-1 pr-2 font-normal">TP</th>
                  <th className="text-right py-1 font-normal">Q</th>
                </tr>
              </thead>
              <tbody>
                {topStates.map(row => (
                  <tr key={row.state} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                    <td className="py-1 pr-3 font-mono text-slate-400 text-[10px]">{row.state}</td>
                    <td className="py-1 pr-3 text-slate-200">{row.best_action}</td>
                    <td className="py-1 pr-2 text-center">{row.params.rsi_os}/{row.params.rsi_ob}</td>
                    <td className="py-1 pr-2 text-center">{row.params.bb_std}</td>
                    <td className="py-1 pr-2 text-center">{row.params.stop_mult}×</td>
                    <td className="py-1 pr-2 text-center">{row.params.tp_ratio}×</td>
                    <td className={`py-1 text-right font-mono ${qColor(row.best_q)}`}>
                      {row.best_q >= 0 ? '+' : ''}{row.best_q.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {(rl.per_state?.length ?? 0) > 5 && (
            <button
              onClick={() => setShowAll(v => !v)}
              className="mt-2 text-xs text-slate-500 hover:text-slate-300 transition-colors"
            >
              {showAll ? '▲ show less' : `▼ show all ${rl.per_state.length} states`}
            </button>
          )}
        </div>
      )}

      {rl.total_updates === 0 && (
        <p className="text-slate-500 text-xs">
          Exploring all variants in round-robin until each has one trade. Then UCB1 exploitation begins.
        </p>
      )}
    </div>
  )
}
