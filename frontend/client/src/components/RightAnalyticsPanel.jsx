import React, { useMemo, useRef, useEffect, useState } from 'react'

/* ─── Right Analytics Panel — Full Responsive ─────────── */
export default function RightAnalyticsPanel({ selected, risks, objects, onExecuteManeuver, onViewRadar }) {
  const highRisks = useMemo(() =>
    risks.filter(r => r.risk === 'HIGH').slice(0, 5), [risks])

  const allConjunctions = useMemo(() =>
    risks.slice(0, 8), [risks])

  const selectedRisk = useMemo(() => {
    if (!selected) return null
    return risks.find(r =>
      r.a === selected.id?.toString() ||
      r.b === selected.id?.toString()
    ) || null
  }, [selected, risks])

  const topRisk = highRisks[0] || risks[0] || null

  return (
    <aside className="layout-right">
      <div className="right-panel-head">
        <span className="panel-title" style={{ color: 'var(--cyan)' }}>Analytics</span>
        <span style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          {risks.length} conjunctions
        </span>
      </div>

      <div className="right-panel-scroll">

        {/* ── 1. OBJECT PROFILE ─────────────────────────── */}
        <Card title="Object Profile" badge={selected?.type?.toUpperCase() || 'SELECT'}>
          {selected ? (
            <>
              <div className="obj-name">{selected.name || selected.id}</div>
              <Row k="NORAD ID"     v={selected.id} mono />
              <Row k="Type"         v={<span className={`tag ${selected.type}`}>{selected.type}</span>} />
              <Row k="Altitude"     v={`${selected.altitude_km ?? '—'} km`} color="cyan" />
              <Row k="Inclination"  v={`${selected.inclination_deg ?? '—'}°`} />
              <Row k="Speed"        v={selected.vx !== undefined
                  ? Math.sqrt((selected.vx||0)**2 + (selected.vy||0)**2 + (selected.vz||0)**2).toFixed(3) + ' km/s'
                  : '—'
                }
              />
            </>
          ) : (
            <div className="card-empty">Click a satellite in the list or globe</div>
          )}
        </Card>

        {/* ── 2. TRAJECTORY PATH ─────────────────────────── */}
        <Card title="Trajectory Path" badge="SGP4 + LSTM">
          <TrajectoryCanvas selected={selected} />
          <div className="card-note">
            Hybrid LSTM residual correction · RMSE <span className="text-cyan">14.76 km</span>
          </div>
        </Card>

        {/* ── 3. CONJUNCTION DETECTION ───────────────────── */}
        <Card
          title="Conjunction Detection"
          badge={highRisks.length > 0
            ? <span className="text-red">{highRisks.length} HIGH</span>
            : <span className="text-green">CLEAR</span>
          }
        >
          {allConjunctions.length > 0 ? (
            <div className="conj-list" style={{ maxHeight: 180, overflowY: 'auto' }}>
              {allConjunctions.map((r, i) => (
                <ConjCard key={i} r={r} />
              ))}
            </div>
          ) : (
            <div className="card-empty">No active conjunctions detected</div>
          )}
        </Card>

        {/* ── 4. RISK ASSESSMENT ─────────────────────────── */}
        <Card title="Risk Assessment" badge="XGBoost · AI">
          <RiskGauge risk={selectedRisk || topRisk} />
        </Card>

        {/* ── 5. RADAR ───────────────────────────────────── */}
        <div 
          onClick={onViewRadar} 
          style={{ cursor: 'pointer', transition: 'transform 0.2s' }}
          onMouseOver={e => e.currentTarget.style.transform = 'scale(1.015)'}
          onMouseOut={e => e.currentTarget.style.transform = 'scale(1)'}
          title="Click to launch Advanced Radar View"
        >
          <Card title="Radar Assessment" badge="Nearest Scan">
            <RadarChart risks={risks} />
            {risks.length > 0 && (
              <div style={{ padding: '8px 10px', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 8, marginTop: 12 }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--red)', textTransform: 'uppercase', marginBottom: 4, letterSpacing: '0.8px' }}>
                  🎯 Click to launch Advanced Radar
                </div>
                {(() => {
                  const nearest = risks.reduce((prev, curr) => (curr.distance < prev.distance ? curr : prev))
                  return (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span className="mono" style={{ fontSize: 10.5, color: '#fff' }}>{nearest.a} ↔ {nearest.b}</span>
                      <span className="mono" style={{ fontSize: 11, color: 'var(--cyan)', fontWeight: 700 }}>{nearest.distance?.toFixed(2)} km</span>
                    </div>
                  )
                })()}
              </div>
            )}
          </Card>
        </div>

        {/* ── 6. COLLISION PREVENTION ─────────────────────── */}
        <CollisionPreventionCard risk={topRisk} onExecute={onExecuteManeuver} />

      </div>
    </aside>
  )
}

/* ─── Reusable Card wrapper ──────────────────────────── */
function Card({ title, badge, children, accent }) {
  return (
    <div className="r-card">
      <div className="r-card-head" style={accent ? { borderLeftColor: accent } : {}}>
        <span className="r-card-title">{title}</span>
        <span className="r-card-badge">{badge}</span>
      </div>
      <div className="r-card-body">{children}</div>
    </div>
  )
}

function Row({ k, v, mono, color }) {
  return (
    <div className="r-row">
      <span className="r-key">{k}</span>
      <span className={`r-val ${mono ? 'mono' : ''} ${color ? 'text-' + color : ''}`}>{v}</span>
    </div>
  )
}

/* ─── Trajectory Canvas ──────────────────────────────── */
function TrajectoryCanvas({ selected }) {
  const canvasRef = useRef()
  const histRef   = useRef([])
  const animRef   = useRef()
  const phaseRef  = useRef(0)

  useEffect(() => {
    if (selected) {
      histRef.current.push({ x: selected.x || 0, y: selected.z || 0 })
      if (histRef.current.length > 120) histRef.current.shift()
    } else {
      histRef.current = []
    }
    draw()
  }, [selected])

  // Animate the trajectory line drawing
  useEffect(() => {
    const tick = () => {
      phaseRef.current += 0.03
      if (histRef.current.length > 1) draw()
      animRef.current = requestAnimationFrame(tick)
    }
    animRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animRef.current)
  }, [])

  const draw = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)

    // Grid
    ctx.strokeStyle = 'rgba(34,211,238,0.07)'
    ctx.lineWidth = 0.5
    for (let i = 1; i < 5; i++) {
      ctx.beginPath(); ctx.moveTo(i * W / 5, 0); ctx.lineTo(i * W / 5, H); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(0, i * H / 5); ctx.lineTo(W, i * H / 5); ctx.stroke()
    }

    const hist = histRef.current
    if (hist.length < 2) {
      // Draw demo orbit when no object selected
      drawDemoOrbit(ctx, W, H, phaseRef.current)
      return
    }

    const xs = hist.map(p => p.x), ys = hist.map(p => p.y)
    const minX = Math.min(...xs), maxX = Math.max(...xs) || minX + 1
    const minY = Math.min(...ys), maxY = Math.max(...ys) || minY + 1
    const pad  = 10

    const nx = x => pad + ((x - minX) / (maxX - minX)) * (W - pad * 2)
    const ny = y => H - pad - ((y - minY) / (maxY - minY)) * (H - pad * 2)

    // Draw trajectory gradient line
    const segCount = hist.length - 1
    for (let i = 0; i < segCount; i++) {
      const alpha = 0.15 + 0.85 * (i / segCount)
      ctx.beginPath()
      ctx.moveTo(nx(xs[i]), ny(ys[i]))
      ctx.lineTo(nx(xs[i + 1]), ny(ys[i + 1]))
      ctx.strokeStyle = selected?.type === 'debris'
        ? `rgba(239,68,68,${alpha})`
        : selected?.type === 'rocket'
        ? `rgba(249,115,22,${alpha})`
        : `rgba(34,211,238,${alpha})`
      ctx.lineWidth = 1.5
      ctx.stroke()
    }

    // Current position dot + glow
    const lx = nx(xs[xs.length - 1])
    const ly = ny(ys[ys.length - 1])
    const dotColor = selected?.type === 'debris' ? '#ef4444'
      : selected?.type === 'rocket' ? '#f97316'
      : '#22d3ee'

    ctx.shadowColor = dotColor; ctx.shadowBlur = 12
    ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2)
    ctx.fillStyle = dotColor; ctx.fill()
    ctx.shadowBlur = 0

    // Predicted next point (dashed)
    if (hist.length > 5) {
      const dx = xs[xs.length - 1] - xs[xs.length - 3]
      const dy = ys[ys.length - 1] - ys[ys.length - 3]
      const px = nx(xs[xs.length - 1] + dx * 2)
      const py = ny(ys[ys.length - 1] + dy * 2)
      ctx.setLineDash([3, 3])
      ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(px, py)
      ctx.strokeStyle = dotColor; ctx.globalAlpha = 0.4; ctx.lineWidth = 1
      ctx.stroke()
      ctx.setLineDash([]); ctx.globalAlpha = 1

      ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2)
      ctx.fillStyle = dotColor; ctx.globalAlpha = 0.5; ctx.fill()
      ctx.globalAlpha = 1
    }
  }

  const drawDemoOrbit = (ctx, W, H, phase) => {
    const cx = W / 2, cy = H / 2
    const rx = W * 0.38, ry = H * 0.32
    ctx.beginPath()
    for (let a = 0; a < Math.PI * 2; a += 0.05) {
      const x = cx + Math.cos(a) * rx, y = cy + Math.sin(a) * ry
      a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
    ctx.closePath()
    ctx.strokeStyle = 'rgba(34,211,238,0.18)'; ctx.lineWidth = 1; ctx.stroke()

    // Moving dot
    const a = phase
    const dotX = cx + Math.cos(a) * rx, dotY = cy + Math.sin(a) * ry
    ctx.shadowColor = '#22d3ee'; ctx.shadowBlur = 10
    ctx.beginPath(); ctx.arc(dotX, dotY, 4, 0, Math.PI * 2)
    ctx.fillStyle = '#22d3ee'; ctx.fill()
    ctx.shadowBlur = 0

    ctx.fillStyle = 'rgba(148,163,184,0.35)'
    ctx.font = '9px Inter'; ctx.textAlign = 'center'
    ctx.fillText('Select object to track trajectory', cx, cy + ry + 14)
  }

  return (
    <canvas
      ref={canvasRef}
      width={268} height={160}
      style={{ width: '100%', height: 160, display: 'block',
        background: 'var(--bg-base)', borderRadius: 6,
        border: '1px solid var(--border)' }}
    />
  )
}

/* ─── Conjunction Card ───────────────────────────────── */
function ConjCard({ r }) {
  return (
    <div className={`conj-card ${r.risk}`}>
      <div className="conj-pair">{r.a} ↔ {r.b}</div>
      <div className="conj-stats">
        <span className="r-key">Dist</span>
        <span className="conj-val">{r.distance?.toFixed(2)} km</span>
        <span className="r-key">Vel</span>
        <span className="conj-val">{r.velocity?.toFixed(2)} km/s</span>
      </div>
      <div style={{ marginTop: 3 }}>
        <RiskBadge level={r.risk} />
      </div>
    </div>
  )
}

/* ─── Risk Badge ─────────────────────────────────────── */
function RiskBadge({ level }) {
  const colorMap = { HIGH: 'var(--red)', MEDIUM: 'var(--orange)', LOW: 'var(--green)' }
  const bgMap    = { HIGH: 'var(--red-dim)', MEDIUM: 'var(--orange-dim)', LOW: 'var(--green-dim)' }
  return (
    <span style={{
      display: 'inline-block', padding: '1px 7px', borderRadius: 20,
      fontSize: 9, fontWeight: 700, letterSpacing: '0.5px',
      background: bgMap[level] || 'var(--bg-hover)',
      color: colorMap[level] || 'var(--text-muted)',
      border: `1px solid ${colorMap[level] || 'var(--border)'}`,
      textTransform: 'uppercase',
    }}>
      {level}
    </span>
  )
}

/* ─── Risk Gauge ─────────────────────────────────────── */
function RiskGauge({ risk }) {
  if (!risk) {
    return <div className="card-empty">No risk data — select an object or wait for conjunction</div>
  }
  const level = risk.risk || 'LOW'
  const conf  = risk.confidence != null ? (risk.confidence * 100).toFixed(1) : '—'
  const probs = risk.probabilities || {}

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <RiskBadge level={level} />
        <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          {conf}% confidence
        </span>
      </div>

      {/* Probability bars */}
      {['HIGH', 'MEDIUM', 'LOW'].map(k => {
        const pct  = ((probs[k] || 0) * 100).toFixed(1)
        const col  = k === 'HIGH' ? 'var(--red)' : k === 'MEDIUM' ? 'var(--orange)' : 'var(--green)'
        return (
          <div key={k} style={{ marginBottom: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginBottom: 3 }}>
              <span>{k}</span><span className="mono">{pct}%</span>
            </div>
            <div style={{ height: 4, background: 'var(--bg-hover)', borderRadius: 2, overflow: 'hidden' }}>
              <div style={{ width: `${pct}%`, height: '100%', background: col,
                borderRadius: 2, transition: 'width 0.6s ease' }} />
            </div>
          </div>
        )
      })}

      <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 3 }}>
        <Row k="Objects"  v={`${risk.a} ↔ ${risk.b}`} mono />
        <Row k="Distance" v={`${risk.distance?.toFixed(2)} km`} color="cyan" />
        <Row k="Rel. Vel" v={`${risk.velocity?.toFixed(3)} km/s`} />
      </div>
    </div>
  )
}

/* ─── Radar Chart ────────────────────────────────────── */
function RadarChart({ risks }) {
  const size = 130, cx = size / 2, cy = size / 2, r = 48
  const axes = ['Proximity', 'Velocity', 'Debris', 'Conjunction', 'Coverage']

  const vals = useMemo(() => {
    const high = risks.filter(r => r.risk === 'HIGH').length
    const med  = risks.filter(r => r.risk === 'MEDIUM').length
    return [
      Math.min(1, high / 3),
      Math.min(1, risks.length > 0 ? (risks[0]?.velocity || 0) / 15 : 0.2),
      Math.min(1, (med + high) / 10),
      Math.min(1, risks.length / 20),
      0.82,
    ]
  }, [risks])

  const pts = axes.map((_, i) => {
    const a = (i / axes.length) * 2 * Math.PI - Math.PI / 2
    return [cx + Math.cos(a) * r * vals[i], cy + Math.sin(a) * r * vals[i]]
  })

  const gridPts = (s) => axes.map((_, i) => {
    const a = (i / axes.length) * 2 * Math.PI - Math.PI / 2
    return [cx + Math.cos(a) * r * s, cy + Math.sin(a) * r * s]
  })

  const labelPts = axes.map((_, i) => {
    const a = (i / axes.length) * 2 * Math.PI - Math.PI / 2
    return [cx + Math.cos(a) * (r + 14), cy + Math.sin(a) * (r + 14)]
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', maxWidth: 170 }}>
        {[0.25, 0.5, 0.75, 1.0].map(s => (
          <polygon key={s}
            points={gridPts(s).map(([x, y]) => `${x},${y}`).join(' ')}
            fill="none" stroke="rgba(34,211,238,0.1)" strokeWidth="0.5" />
        ))}
        {labelPts.map(([x, y], i) => (
          <line key={i} x1={cx} y1={cy} x2={x - (x - cx) * 0.07} y2={y - (y-cy) * 0.07}
            stroke="rgba(34,211,238,0.08)" strokeWidth="0.5" />
        ))}
        <polygon points={pts.map(([x, y]) => `${x},${y}`).join(' ')}
          fill="rgba(249,115,22,0.12)" stroke="#f97316" strokeWidth="1.5" />
        {pts.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={2.5} fill="#f97316" />
        ))}
        {axes.map((label, i) => (
          <text key={i} x={labelPts[i][0]} y={labelPts[i][1]}
            textAnchor="middle" dominantBaseline="middle"
            fill="rgba(148,163,184,0.65)" fontSize="5.5">{label}</text>
        ))}
      </svg>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, justifyContent: 'center' }}>
        {axes.map((a, i) => (
          <span key={i} style={{ fontSize: 9, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 3 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#f97316', opacity: 0.5 + vals[i] * 0.5, display: 'inline-block' }} />
            {(vals[i] * 100).toFixed(0)}%
          </span>
        ))}
      </div>
    </div>
  )
}

/* ─── Collision Prevention Card (prominent) ──────────── */
function CollisionPreventionCard({ risk, onExecute }) {
  const steps = [
    { n: '1', title: 'Prograde Burn',   text: 'Increase periapsis above conjunction zone via forward thrust.' },
    { n: '2', title: 'Retrograde Burn', text: 'Decelerate to pass below the debris field safely.' },
    { n: '3', title: 'Cross-track',     text: 'Lateral thrust — shift orbital plane by 0.1–0.5° to clear path.' },
    { n: '4', title: 'Phasing',         text: 'Adjust orbital period so satellite arrives at safe time offset.' },
  ]

  return (
    <div className="r-card prevention-card">
      <div className="r-card-head" style={{ background: 'rgba(99,102,241,0.12)', borderLeft: '3px solid var(--indigo)' }}>
        <span className="r-card-title" style={{ color: 'var(--indigo)' }}>Collision Prevention</span>
        <span className="r-card-badge">PPO RL Agent</span>
      </div>
      <div className="r-card-body">

        {/* PPO recommended maneuver box */}
        {risk ? (
          <div className="maneuver-box">
            <div className="maneuver-box-title">
              ⚡ PPO Recommended Maneuver
            </div>
            {risk.maneuver ? (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
                <div>
                  <div className="r-key">ΔV Vector</div>
                  <div className="maneuver-val">
                    [{risk.maneuver.delta_v?.map(v => v.toFixed(3)).join(', ')}]
                  </div>
                </div>
                <div>
                  <div className="r-key">Fuel Cost</div>
                  <div className="maneuver-val" style={{ color: 'var(--orange)' }}>
                    {risk.maneuver.fuel_cost} km/s
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 8 }}>
                ΔV computed for HIGH risk events only.
              </div>
            )}
            <button className="execute-btn" onClick={onExecute}>
              🚀 Execute Avoidance Maneuver
            </button>
          </div>
        ) : (
          <div className="card-empty" style={{ marginBottom: 8 }}>
            No active HIGH risk — system nominal
          </div>
        )}

        {/* Step-by-step methods */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 4 }}>
          {steps.map(s => (
            <div key={s.n} className="prevention-item">
              <span className="prevention-num">{s.n}</span>
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 1 }}>
                  {s.title}
                </div>
                <div style={{ fontSize: 9.5, color: 'var(--text-muted)', lineHeight: 1.4 }}>
                  {s.text}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
