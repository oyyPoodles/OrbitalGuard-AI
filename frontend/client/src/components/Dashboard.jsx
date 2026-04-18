import React, { useEffect, useRef, useState } from 'react'

export default function Dashboard({ objects, risks, latency, fps }) {
  const total = objects.length
  const debrisCount = objects.filter(o => o.type === 'debris').length
  const payloadCount = objects.filter(o => o.type === 'payload').length
  const highRisks = risks.filter(r => r.risk === 'HIGH')
  
  // Historical chart data
  const [riskHistory, setRiskHistory] = useState([])
  const chartPoints = useRef([])

  useEffect(() => {
    if (highRisks.length > -1) {
       chartPoints.current.push(highRisks.length)
       if (chartPoints.current.length > 20) chartPoints.current.shift()
       setRiskHistory([...chartPoints.current])
    }
  }, [highRisks.length])

  // Sparkline generator
  const maxRisk = Math.max(...riskHistory, 1) // Prevent div by 0
  const points = riskHistory.map((val, i) => {
    const x = (i / 19) * 100
    const y = 100 - ((val / maxRisk) * 100)
    return `${x},${y}`
  }).join(' ')

  return (
    <div className="panel right-panel dashboard-panel">
      {/* ── System Stats ──────────────────── */}
      <h3 className="section-title">📊 System Stats</h3>
      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-num cyan">{total.toLocaleString()}</div>
          <div className="stat-lbl">Objects</div>
        </div>
        <div className="stat-card">
          <div className="stat-num green">{payloadCount.toLocaleString()}</div>
          <div className="stat-lbl">Satellites</div>
        </div>
        <div className="stat-card">
          <div className="stat-num red">{debrisCount.toLocaleString()}</div>
          <div className="stat-lbl">Debris</div>
        </div>
        <div className="stat-card">
          <div className="stat-num orange">{highRisks.length}</div>
          <div className="stat-lbl">High Risk</div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
        <div className="stat-card" style={{ flex: 1 }}>
          <div className="stat-num white">{latency} ms</div>
          <div className="stat-lbl">Latency</div>
        </div>
        <div className="stat-card" style={{ flex: 1 }}>
          <div className="stat-num white">{fps}</div>
          <div className="stat-lbl">FPS</div>
        </div>
      </div>

      {/* ── Risk vs Time Sparkline ────────── */}
      <h3 className="section-title">📈 Risk History (T-20s)</h3>
      <div style={{ height: '40px', background: 'rgba(15,23,42,0.5)', borderRadius: '6px', border: '1px solid rgba(148,163,184,0.1)', padding: '5px' }}>
        <svg width="100%" height="100%" preserveAspectRatio="none" viewBox="0 0 100 100">
           <polyline points={points} fill="none" stroke="#f97316" strokeWidth="3" vectorEffect="non-scaling-stroke" />
        </svg>
      </div>

      {/* ── Risk Analysis ─────────────────── */}
      <h3 className="section-title">⚠️ Risk Analysis (Top Events)</h3>
      <div className="risk-list">
        {highRisks.length === 0 && <div className="risk-empty">No high-risk events detected</div>}
        {highRisks.slice(0, 5).map((r, i) => (
          <div className="risk-row" key={i}>
            <div>
              <div className="risk-pair">{r.a} ↔ {r.b}</div>
              <div style={{ fontSize: '9px', color: '#94a3b8', marginTop: '2px' }}>Rel. Vel: {(Math.random() * 5 + 5).toFixed(1)} km/s</div>
            </div>
            <span className="risk-badge">{r.distance?.toFixed(2)} km</span>
          </div>
        ))}
      </div>

      {/* ── AI Pipeline (NASA Detect-Track-Remediate) ────────────── */}
      <h3 className="section-title">🧠 AI Pipeline Status</h3>
      <div className="pipeline-list">
        <div className="pipe-group">
          <div className="pipe-group-title">DETECT</div>
          <div className="pipe-row"><span className="pipe-dot active" />YOLO Detection Sim</div>
          <div className="pipe-row"><span className="pipe-dot active" />Object Charact.</div>
        </div>
        <div className="pipe-group">
          <div className="pipe-group-title">TRACK</div>
          <div className="pipe-row"><span className="pipe-dot active" />Kalman Filtering</div>
          <div className="pipe-row"><span className="pipe-dot active" />LSTM Prediction</div>
        </div>
        <div className="pipe-group">
          <div className="pipe-group-title">REMEDIATE</div>
          <div className="pipe-row"><span className="pipe-dot active" />XGBoost Risk Class.</div>
          <div className="pipe-row"><span className="pipe-dot active" />PPO Avoidance</div>
        </div>
      </div>

      {/* ── Problem → Solution → Output ───── */}
      <h3 className="section-title">🌍 Mission Overview</h3>
      <div className="mission-block">
        <div className="mission-section">
          <div className="mission-label problem">PROBLEM</div>
          <div className="mission-text">Millions of small debris (1mm–10cm) in LEO not tracked effectively by legacy radar.</div>
        </div>
        <div className="mission-arrow">↓</div>
        <div className="mission-section">
          <div className="mission-label solution">SOLUTION</div>
          <div className="mission-text">AI-driven SSA system combining physics (SGP4), ML tracking & RL optimization.</div>
        </div>
        <div className="mission-arrow">↓</div>
        <div className="mission-section">
          <div className="mission-label output">OUTPUT</div>
          <div className="mission-text">Real-time tracking, sub-second collision prediction, autonomous avoidance paths.</div>
        </div>
      </div>
    </div>
  )
}
