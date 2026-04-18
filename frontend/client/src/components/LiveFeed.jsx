import React, { useMemo, useRef, useEffect, useState } from 'react'

/* ─── Live Telemetry Feed ────────────────────────────── */
export default function LiveFeed({ objects, risks }) {
  const [tick, setTick] = useState(0)

  // Flash indicator every update
  useEffect(() => { setTick(t => t + 1) }, [objects.length])

  const riskSet = useMemo(() => {
    const s = new Set()
    risks.forEach(r => { s.add(r.a); s.add(r.b) })
    return s
  }, [risks])

  const feedItems = objects.slice(0, 60)

  const speed = (o) => {
    const vx = o.vx || 0, vy = o.vy || 0, vz = o.vz || 0
    return Math.sqrt(vx*vx + vy*vy + vz*vz).toFixed(2)
  }

  const typeLabel = (t) => ({ payload: 'PAYL', debris: 'DEB', rocket: 'RKT' })[t] || '??'
  const typeColor = (t) => t === 'debris' ? 'var(--red)' : t === 'rocket' ? 'var(--orange)' : 'var(--green)'

  return (
    <div className="live-feed-panel">
      {/* Header */}
      <div className="live-feed-header">
        <span className="panel-title" style={{ fontSize: 9, color: 'var(--cyan)' }}>Live Telemetry</span>
        <span className="live-badge">
          <span className="pulse-dot" />
          LIVE
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          {objects.length.toLocaleString()} obj
        </span>
      </div>

      {/* Scrollable table */}
      <div className="feed-scroll">
        {/* Column headers */}
        <div className="feed-row feed-header-row">
          <span>Name</span>
          <span>Alt</span>
          <span>Spd</span>
          <span>Inc</span>
          <span>Type</span>
          <span>Status</span>
        </div>

        {feedItems.map((o, i) => {
          const isRisk = riskSet.has(o.id?.toString())
          return (
            <div
              key={o.id || i}
              className={`feed-row ${isRisk ? 'feed-row-risk' : ''}`}
              style={{ animationDelay: `${i * 0.01}s` }}
            >
              <span className="feed-name" title={o.name}>{o.name || o.id}</span>
              <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)', fontSize: 9.5 }}>
                {o.altitude_km ?? '—'}
              </span>
              <span style={{ color: 'var(--blue)', fontFamily: 'var(--font-mono)', fontSize: 9.5 }}>
                {speed(o)}
              </span>
              <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 9.5 }}>
                {o.inclination_deg ?? '—'}
              </span>
              <span style={{ color: typeColor(o.type), fontSize: 9, fontWeight: 700 }}>
                {typeLabel(o.type)}
              </span>
              <span style={{
                color: isRisk ? 'var(--orange)' : 'var(--green)',
                fontSize: 9, fontWeight: 700,
              }}>
                {isRisk ? '⚠ RISK' : '● OK'}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
