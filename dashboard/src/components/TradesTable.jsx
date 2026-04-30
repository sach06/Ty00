/** Recent closed trades table */
export default function TradesTable({ trades = [] }) {
  if (!trades.length) {
    return (
      <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <h2 className="text-sm text-slate-400 uppercase tracking-widest mb-3">Recent Trades</h2>
        <p className="text-slate-500 text-sm">No closed trades yet.</p>
      </div>
    )
  }

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 overflow-x-auto">
      <h2 className="text-sm text-slate-400 uppercase tracking-widest mb-3">
        Recent Trades ({trades.length})
      </h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-slate-500 text-xs border-b border-slate-700">
            <th className="text-left pb-2 pr-4">Time (UTC)</th>
            <th className="text-left pb-2 pr-4">Epic</th>
            <th className="text-left pb-2 pr-4">Dir</th>
            <th className="text-right pb-2 pr-4">Entry</th>
            <th className="text-right pb-2 pr-4">Exit</th>
            <th className="text-right pb-2 pr-4">Size</th>
            <th className="text-right pb-2 pr-4">P&L (€)</th>
            <th className="text-left pb-2">Reason</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t, i) => {
            const pnlColor = t.pnl_eur > 0 ? 'text-emerald-400' : t.pnl_eur < 0 ? 'text-red-400' : 'text-slate-400'
            const timeStr = t.time ? new Date(t.time).toLocaleTimeString('en-GB', { timeZone: 'UTC', hour12: false }) : '—'
            return (
              <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                <td className="py-2 pr-4 text-slate-500 text-xs">{timeStr}</td>
                <td className="py-2 pr-4 font-semibold text-slate-100">{t.epic}</td>
                <td className={`py-2 pr-4 font-semibold ${t.direction === 'BUY' ? 'text-emerald-400' : 'text-red-400'}`}>
                  {t.direction}
                </td>
                <td className="py-2 pr-4 text-right text-slate-300">{t.entry?.toFixed(5)}</td>
                <td className="py-2 pr-4 text-right text-slate-300">{t.exit?.toFixed(5)}</td>
                <td className="py-2 pr-4 text-right text-slate-300">{t.size}</td>
                <td className={`py-2 pr-4 text-right font-semibold ${pnlColor}`}>
                  {t.pnl_eur > 0 ? '+' : ''}{t.pnl_eur?.toFixed(2)}
                </td>
                <td className="py-2 text-slate-500 text-xs">{t.reason || '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
