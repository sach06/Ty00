/** Small equity sparkline using recharts */
import { AreaChart, Area, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

export default function EquityChart({ history = [] }) {
  if (history.length < 2) return null

  const data = history.map((h) => ({ t: h.t, v: h.v }))
  const minVal = Math.min(...data.map((d) => d.v))
  const maxVal = Math.max(...data.map((d) => d.v))
  const domain = [minVal * 0.9995, maxVal * 1.0005]

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <h2 className="text-sm text-slate-400 uppercase tracking-widest mb-3">Equity Curve</h2>
      <ResponsiveContainer width="100%" height={120}>
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
          </defs>
          <YAxis domain={domain} hide />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.t?.slice(11, 19) + ' UTC'}
            formatter={(v) => [`€${v.toFixed(2)}`, 'Equity']}
          />
          <Area
            type="monotone"
            dataKey="v"
            stroke="#10b981"
            strokeWidth={2}
            fill="url(#eqGrad)"
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
