/** Open positions table */
export default function PositionsTable({ positions = [] }) {
  if (!positions.length) {
    return (
      <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <h2 className="text-sm text-slate-400 uppercase tracking-widest mb-3">Open Positions</h2>
        <p className="text-slate-500 text-sm">No open positions.</p>
      </div>
    )
  }

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 overflow-x-auto">
      <h2 className="text-sm text-slate-400 uppercase tracking-widest mb-3">
        Open Positions ({positions.length})
      </h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-slate-500 text-xs border-b border-slate-700">
            <th className="text-left pb-2 pr-4">Epic</th>
            <th className="text-left pb-2 pr-4">Dir</th>
            <th className="text-right pb-2 pr-4">Entry</th>
            <th className="text-right pb-2 pr-4">Stop</th>
            <th className="text-right pb-2 pr-4">Target</th>
            <th className="text-right pb-2 pr-4">Size</th>
            <th className="text-right pb-2 pr-4">UPL</th>
            <th className="text-right pb-2">Age</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p) => (
            <tr key={p.deal_id || p.epic} className="border-b border-slate-700/50 hover:bg-slate-700/30">
              <td className="py-2 pr-4 font-semibold text-slate-100">
                {p.epic}
                {p.source === 'capital-account' && (
                  <div className="text-[11px] font-normal text-slate-500">account</div>
                )}
              </td>
              <td className={`py-2 pr-4 font-semibold ${p.direction === 'BUY' ? 'text-emerald-400' : 'text-red-400'}`}>
                {p.direction}
              </td>
              <td className="py-2 pr-4 text-right text-slate-300">{typeof p.entry === 'number' ? p.entry.toFixed(5) : '—'}</td>
              <td className="py-2 pr-4 text-right text-red-400">{typeof p.stop === 'number' ? p.stop.toFixed(5) : '—'}</td>
              <td className="py-2 pr-4 text-right text-emerald-400">{typeof p.tp === 'number' ? p.tp.toFixed(5) : '—'}</td>
              <td className="py-2 pr-4 text-right text-slate-300">{p.size}</td>
              <td className={`py-2 pr-4 text-right font-semibold ${(p.upl ?? 0) > 0 ? 'text-emerald-400' : (p.upl ?? 0) < 0 ? 'text-red-400' : 'text-slate-400'}`}>
                {typeof p.upl === 'number' ? `${p.upl > 0 ? '+' : ''}${p.upl.toFixed(2)}` : '—'}
              </td>
              <td className="py-2 text-right text-slate-400">{typeof p.age_hours === 'number' ? p.age_hours.toFixed(1) : '0.0'}h</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
