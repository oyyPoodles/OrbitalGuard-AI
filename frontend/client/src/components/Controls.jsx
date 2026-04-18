import React from 'react'

export default function Controls({ showDebris, setShowDebris, showHighRiskOnly, setShowHighRiskOnly, searchQuery, setSearchQuery, onExport }) {
  return (
    <div className="panel left-panel">
      <div className="brand">
        <div className="brand-icon">🛰</div>
        <div>
          <h1 className="brand-title">OrbitalGuard AI</h1>
          <div className="brand-sub">Space Situational Awareness</div>
        </div>
      </div>

      <div className="control-group">
        <label>Search Object</label>
        <input
          type="text" className="search-input"
          placeholder="Name or NORAD ID..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      <div className="control-group" style={{ opacity: 0.6 }}>
        <label>Simulated Time Offset</label>
        <input type="range" min="0" max="100" defaultValue="0" disabled title="Time travel currently synced with backend" />
        <div style={{ fontSize: '9px', textAlign: 'right', marginTop: '4px', color: '#94a3b8' }}>+0s (Realtime)</div>
      </div>

      <div className="control-group">
        <div className="toggle-row"><span>Show Debris</span><input type="checkbox" checked={showDebris} onChange={e => setShowDebris(e.target.checked)} /></div>
        <div className="toggle-row"><span>High Risk Only</span><input type="checkbox" checked={showHighRiskOnly} onChange={e => setShowHighRiskOnly(e.target.checked)} /></div>
      </div>

      <button 
        onClick={onExport}
        style={{ width: '100%', padding: '10px', background: 'rgba(34, 211, 238, 0.1)', border: '1px solid rgba(34, 211, 238, 0.3)', color: '#22d3ee', borderRadius: '8px', cursor: 'pointer', fontFamily: "'Inter', sans-serif", fontSize: '11px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '1px', marginTop: '10px' }}
      >
        📥 Export Risk Data
      </button>

      <div className="legend">
        <div className="legend-item"><span className="dot payload" /> Active Payload (Green)</div>
        <div className="legend-item"><span className="dot starlink" /> Starlink (Cyan)</div>
        <div className="legend-item"><span className="dot debris" /> Debris (Red)</div>
        <div className="legend-item"><span className="dot danger" style={{background: '#ffaa00'}} /> High Risk (Orange)</div>
      </div>
    </div>
  )
}
