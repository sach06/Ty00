import { useEffect, useRef, useState } from 'react'

/** Auto-scrolling live log feed */
export default function LogFeed({ lines = [] }) {
  const bottomRef = useRef(null)
  const [collapsed, setCollapsed] = useState(true)

  useEffect(() => {
    if (collapsed) return
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines, collapsed])

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-700 flex flex-col" style={{ height: collapsed ? 'auto' : '280px' }}>
      <div className="px-4 py-2 border-b border-slate-700 flex items-center justify-between">
        <button
          type="button"
          onClick={() => setCollapsed(v => !v)}
          className="text-sm text-slate-400 uppercase tracking-widest hover:text-slate-200 transition-colors"
        >
          Live Log {collapsed ? '▸' : '▾'}
        </button>
        <span className="text-xs text-slate-600">{lines.length} lines</span>
      </div>
      {!collapsed && (
      <div className="flex-1 overflow-y-auto px-4 py-2 text-xs text-slate-400 leading-5">
        {lines.length === 0 && <p className="text-slate-600">No log output yet.</p>}
        {lines.map((line, i) => {
          let color = 'text-slate-400'
          if (line.includes('ERROR') || line.includes('CRITICAL') || line.includes('KILL')) color = 'text-red-400'
          else if (line.includes('WARNING') || line.includes('HALT') || line.includes('PAUSE')) color = 'text-yellow-400'
          else if (line.includes('SIGNAL') || line.includes('order submitted') || line.includes('BUY') || line.includes('SELL')) color = 'text-emerald-400'
          else if (line.includes('DRY-RUN')) color = 'text-sky-400'

          return (
            <div key={i} className={`${color} whitespace-pre-wrap break-all`}>
              {line}
            </div>
          )
        })}
        <div ref={bottomRef} />
      </div>
      )}
    </div>
  )
}
