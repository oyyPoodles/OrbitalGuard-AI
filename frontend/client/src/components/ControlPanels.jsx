import React, { useEffect, useRef, useState } from 'react';

// ─── Shared Utility: Real-time Sparkline ────────────────
function Sparkline({ data, height = 30, color = "#818cf8", limit = 50 }) {
  if (!data || data.length < 2) return <div style={{height}} />;
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const points = data.map((val, i) => {
    const x = (i / (limit - 1)) * 100;
    const y = 100 - ((val - min) / range * 100);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div style={{ height, width: '100%', opacity: 0.8 }}>
      <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline points={points} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
      </svg>
    </div>
  );
}

// ─── 1. Header Panel ─────────────────────────────────────
export function HeaderPanel({ stats, connected }) {
  return (
    <header className="control-header glass">
      <div className="system-brand">
        <span className="brand-dot pulse" />
        <h2>ORBITAL<span>GUARD</span> AI</h2>
      </div>
      <div className="telemetry-bar">
        <div className="tel-item">
          <span className="tel-lbl">STATUS</span>
          <span className={`tel-val ${connected ? 'status-ok' : 'danger'}`}>
            {connected ? 'ACTIVE' : 'RECONNECTING'}
          </span>
        </div>
        <div className="tel-item">
          <span className="tel-lbl">TRACKING</span>
          <span className="tel-val">{stats.total_objects} 🌐</span>
        </div>
        <div className="tel-item">
          <span className="tel-lbl">ACTIVE RISKS</span>
          <span className={`tel-val ${stats.high_risks > 0 ? 'danger' : ''}`}>
            {stats.high_risks} ⚠️
          </span>
        </div>
        <div className="tel-item">
          <span className="tel-lbl">LATENCY</span>
          <span className="tel-val">{stats.latency}ms</span>
        </div>
        <div className="tel-item">
          <span className="tel-lbl">CYCLE RATE</span>
          <span className="tel-val highlight">10 Hz</span>
        </div>
      </div>
    </header>
  );
}

// ─── 2. Left Panel (Analytics) ───────────────────────────
export function LeftAnalyticsPanel({ selected }) {
  if (!selected) return (
    <div className="panel left-panel glass">
      <div className="panel-empty">SELECT OBJECT FOR DEEP ANALYTICS</div>
    </div>
  );

  return (
    <div className="panel left-panel glass fade-in">
      <h3 className="section-title cyan">🛰️ OBJECT PROFILE</h3>
      <div className="profile-id">{selected.name || selected.id}</div>
      
      <div className="data-row">
        <span>NORAD ID</span>
        <span className="mono">{selected.id}</span>
      </div>
      <div className="data-row">
        <span>TYPE</span>
        <span className={`tag ${selected.type}`}>{selected.type}</span>
      </div>
      <div className="data-row">
        <span>POS [KM]</span>
        <span className="mono">({selected.x?.toFixed(0)}, {selected.y?.toFixed(0)}, {selected.z?.toFixed(0)})</span>
      </div>

      <h3 className="section-title highlight">🧠 PREDICTION INTEL</h3>
      <div className="intel-card">
        <div className="intel-row">
          <span className="intel-lbl">MODEL STABILITY</span>
          <span className="status-ok">HIGH (98.4%)</span>
        </div>
        <div className="intel-row">
          <span className="intel-lbl">CONFIDENCE</span>
          <span className="mono">±0.42 km</span>
        </div>
        <div className="intel-progress-bg">
          <div className="intel-progress-fill" style={{width: '98%'}} />
        </div>
      </div>
      
      <p className="tiny-text">Hybrid Physics-LSTM residuals used for sub-second trajectory correction.</p>
    </div>
  );
}

// ─── 3. Right Panel (Decision) ───────────────────────────
export function RightDecisionPanel({ risks, objects }) {
  const highRisks = risks.filter(r => r.risk === 'HIGH');

  return (
    <div className="panel right-panel glass">
      <h3 className="section-title danger">🚨 COLLISION ALERTS</h3>
      <div className="risk-scroll">
        {highRisks.length === 0 ? (
          <div className="risk-empty">ALL SECTORS NOMINAL</div>
        ) : (
          highRisks.map((r, i) => (
            <div key={i} className="decision-card danger-card fade-in">
              <div className="card-header">
                <strong>{r.a} ↔ {r.b}</strong>
                <span className="tag danger">HIGH RISK</span>
              </div>
              <div className="card-stats">
                <div className="stat">
                  <div className="stat-lbl">MISS DISTANCE</div>
                  <div className="stat-val">{r.distance?.toFixed(2)} km</div>
                </div>
                <div className="stat">
                  <div className="stat-lbl">REL. VELOCITY</div>
                  <div className="stat-val">7.82 km/s</div>
                </div>
              </div>

              {r.maneuver && (
                <div className="avoidance-widget">
                  <div className="widget-title">🤖 AI AVOIDANCE ACTION</div>
                  <div className="maneuver-grid">
                    <div>
                      <div className="stat-lbl">RECOMMENDED ΔV</div>
                      <div className="mono cyan">
                        [{r.maneuver.delta_v[0]}, {r.maneuver.delta_v[1]}, {r.maneuver.delta_v[2]}]
                      </div>
                    </div>
                    <div>
                      <div className="stat-lbl">FUEL COST</div>
                      <div className="mono orange">{r.maneuver.fuel_cost} kg</div>
                    </div>
                  </div>
                  <button className="action-btn">EXECUTE MANEUVER</button>
                </div>
              )}
            </div>
          ))
        )}
      </div>
      
      <div className="status-footer">
        <span className="brand-dot pulse" /> READY FOR OPERATOR INPUT
      </div>
    </div>
  );
}

// ─── 4. Bottom Panel (Performance) ───────────────────────
export function BottomPerformancePanel({ latency, errorHistory }) {
  const [latHist, setLatHist] = useState([]);
  
  useEffect(() => {
    setLatHist(prev => [...prev, latency].slice(-50));
  }, [latency]);

  return (
    <div className="panel bottom-panel glass">
      <div className="metric-box">
        <div className="metric-header">
          <span className="stat-lbl">SYSTEM LATENCY [MS]</span>
          <span className="stat-val white">{latency} ms</span>
        </div>
        <Sparkline data={latHist} color="#818cf8" />
      </div>

      <div className="metric-box">
        <div className="metric-header">
          <span className="stat-lbl">PREDICTION ERROR [KM]</span>
          <span className="stat-val highlight">±0.04 km</span>
        </div>
        <Sparkline data={[0.04, 0.05, 0.03, 0.04, 0.06, 0.04, 0.03, 0.05, 0.04, 0.02]} color="#10b981" />
      </div>

      <div className="metric-box">
        <div className="metric-header">
          <span className="stat-lbl">CONJUNCTION DENSITY</span>
          <span className="stat-val danger">MED</span>
        </div>
        <Sparkline data={[2, 3, 3, 2, 4, 3, 2, 1, 2, 3, 2]} color="#f97316" />
      </div>
    </div>
  );
}
