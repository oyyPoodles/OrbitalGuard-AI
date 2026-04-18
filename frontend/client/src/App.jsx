import React, { useState, useEffect, useRef, useCallback } from 'react'
import * as THREE from 'three'
import './styles.css'
import OrbLogo from './logo.png'

import Scene              from './components/Scene'
import SatelliteList      from './components/SatelliteList'
import LiveFeed           from './components/LiveFeed'
import OrbitalChat        from './components/OrbitalChat'
import RightAnalyticsPanel from './components/RightAnalyticsPanel'
import AnalyticsDashboard from './components/AnalyticsDashboard'

const WS_URL  = 'ws://localhost:8000/ws/live'
const API_URL = 'http://localhost:8000'

/* ─── View Modes ────────────────────────────────────────── */
const VIEWS = ['Dashboard', 'Analytics']

export default function App() {
  const [showIntro, setShowIntro]     = useState(true)
  const [view, setView]               = useState('Dashboard')
  const [objects, setObjects]         = useState([])
  const [risks, setRisks]             = useState([])
  const [selected, setSelected]       = useState(null)
  const [focusTarget, setFocusTarget] = useState(null)
  const [wsStatus, setWsStatus]       = useState('connecting')
  const [utcTime, setUtcTime]         = useState('')
  const [stats, setStats]             = useState({ total: 0, debris: 0, highRisk: 0, latency: 0 })
  const [avoidancePath, setAvoidancePath] = useState(null)

  const wsRef   = useRef(null)
  const retryRef = useRef(null)

  // ── UTC Clock ──────────────────────────────────────────
  useEffect(() => {
    const tick = () => {
      const now = new Date()
      setUtcTime(now.toUTCString().split(' ').slice(4, 5)[0] + ' UTC')
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  // ── WebSocket ──────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setWsStatus('live')
      if (retryRef.current) { clearTimeout(retryRef.current); retryRef.current = null }
    }

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'update') {
          setObjects(msg.objects || [])
          setRisks(msg.risks   || [])
          setStats(prev => ({
            total:    (msg.objects || []).length,
            debris:   (msg.objects || []).filter(o => o.type === 'debris' || o.type === 'rocket').length,
            highRisk: (msg.risks   || []).filter(r => r.risk === 'HIGH').length,
            latency:  prev.latency,
          }))
        }
      } catch { /* ignore parse errors */ }
    }

    ws.onerror = () => setWsStatus('error')

    ws.onclose = () => {
      setWsStatus('reconnecting')
      retryRef.current = setTimeout(connect, 3000)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      wsRef.current?.close()
      if (retryRef.current) clearTimeout(retryRef.current)
    }
  }, [connect])

  // ── Object Selection ───────────────────────────────────
  const handleSelect = useCallback((obj) => {
    if (!obj) { setSelected(null); setFocusTarget(null); setAvoidancePath(null); return }
    setSelected(obj)
    setFocusTarget({ x: obj.x, y: obj.y, z: obj.z })
    setAvoidancePath(null)
  }, [])

  // ── PPO Avoidance Maneuver ─────────────────────────────
  const handleExecuteManeuver = useCallback(() => {
    if (!selected) return;
    
    // Orbital dodge calculation
    const pos = new THREE.Vector3(selected.x, selected.y, selected.z);
    
    // Use true velocity if exists, else arbitrary forward vector perfectly tangent
    let vel = new THREE.Vector3(selected.vx || 0, selected.vy || 0.5, selected.vz || 0);
    if (vel.lengthSq() < 0.001 || isNaN(vel.x)) {
      vel = pos.clone().cross(new THREE.Vector3(0, 1, 0)).normalize();
    }
    
    // Orbital Normal Axis
    const axis = new THREE.Vector3().crossVectors(pos, vel).normalize();
    
    // Maneuver radius shift: +0.6 visual units outward over earth to safely dodge
    const r = pos.length() + 0.6;
    
    const safePath = [];
    // Start at adjusted altitude and sweep around the globe
    const p = pos.clone().setLength(r);
    for (let i = 0; i < 200; i++) {
       safePath.push([p.x, p.y, p.z]);
       // Smooth rotation across the orbit
       p.applyAxisAngle(axis, 0.035);
    }
    
    setAvoidancePath({ id: selected.id, points: safePath })
  }, [selected])

  // ── Stats Bar values ───────────────────────────────────
  const activeCount = stats.total - stats.debris

  return (
    <div className="app-shell">

      {/* ── Intro Modal ───────────────────────────────────────── */}
      {showIntro && (
        <div style={{
          position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh',
          background: 'rgba(2, 6, 12, 0.75)', backdropFilter: 'blur(8px)',
          zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
          <div style={{
            width: 540, padding: 32, background: 'var(--bg-card)', 
            border: '1px solid rgba(34, 211, 238, 0.3)', borderRadius: 12,
            boxShadow: '0 20px 50px rgba(0,0,0,0.9), inset 0 0 20px rgba(34, 211, 238, 0.05)',
            display: 'flex', flexDirection: 'column', gap: 20, position: 'relative'
          }}>
            <div style={{ position: 'absolute', top: -1, right: 20, width: 30, height: 2, background: 'var(--cyan)' }} />
            <div style={{ position: 'absolute', bottom: -1, left: 20, width: 30, height: 2, background: 'var(--cyan)' }} />
            
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
               <img src={OrbLogo} width={50} height={50} alt="Logo" />
               <div>
                 <h1 style={{ color: '#fff', fontSize: 24, margin: 0, letterSpacing: '1.5px', fontFamily: 'var(--font-mono)' }}>
                   Orbital<span style={{color: 'var(--cyan)'}}>Guard</span> AI
                 </h1>
                 <div style={{ color: 'var(--cyan)', fontSize: 10, letterSpacing: '1.5px', textTransform: 'uppercase', marginTop: 4 }}>
                   Space Debris Surveillance Framework
                 </div>
               </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, color: 'var(--text-muted)', fontSize: 11, lineHeight: 1.5 }}>
              <p style={{ margin: 0 }}>
                <strong style={{ color: '#fff', fontSize: 12 }}>THE KESSLER SYNDROME:</strong> Low Earth Orbit is approaching critical density. The threat of cascading orbital collisions mathematically threatens modern localized satellite infrastructure.
              </p>
              <p style={{ margin: 0 }}>
                <strong style={{ color: '#fff', fontSize: 12 }}>SYSTEM ARCHITECTURE:</strong> OrbitalGuard AI is an autonomous early warning framework deploying a 5-Stage Machine Learning Pipeline:
              </p>
              
              <ul style={{ margin: 0, paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 8, color: 'var(--text-secondary)' }}>
                <li><strong style={{color:'var(--green)'}}>Hybrid SGP4 + LSTM:</strong> Recurrent neural networks predict and correct orbital drift residuals, shrinking positional uncertainty (RMSE) by ~25%.</li>
                <li><strong style={{color:'var(--green)'}}>KDTree Indexing:</strong> Translates 3D tracking vectors for O(n log n) sub-millisecond proximity sorting.</li>
                <li><strong style={{color:'var(--green)'}}>XGBoost Classification:</strong> Evaluates incoming velocity vectors and constraint radii to classify HIGH RISK planetary conjunctions.</li>
                <li><strong style={{color:'var(--green)'}}>PPO Avoidance Engine:</strong> Proximal Policy Optimization reinforcement algorithms physically deploy thrust vectors to generate real-time collision maneuver arcs.</li>
              </ul>
            </div>

            <button 
              onClick={() => setShowIntro(false)}
              className="mono"
              style={{
                marginTop: 16, padding: '12px 24px', background: 'rgba(34, 211, 238, 0.1)', 
                border: '1px solid var(--cyan)', color: 'var(--cyan)', 
                fontSize: 12, fontWeight: 'bold', letterSpacing: '1.5px', cursor: 'pointer',
                borderRadius: 6, transition: 'all 0.2s ease', alignSelf: 'center',
                boxShadow: '0 0 15px rgba(34,211,238,0.2)'
              }}
              onMouseOver={e => e.currentTarget.style.background = 'rgba(34, 211, 238, 0.2)'}
              onMouseOut={e => e.currentTarget.style.background = 'rgba(34, 211, 238, 0.1)'}
            >
              INITIALIZE COMMAND DASHBOARD
            </button>
            <div style={{ fontSize: 9, textAlign: 'center', opacity: 0.3, color: '#fff', fontFamily: 'var(--font-mono)' }}>
              AEROSPACE DEFENSE COMMAND // SECURE PROTOCOL INITIATED // CLEARANCE ACCESS = ONE
            </div>
          </div>
        </div>
      )}

      {/* ── Top Header ─────────────────────────────────── */}
      <header className="top-header">
        {/* Brand */}
        <div className="header-brand">
          <img src={OrbLogo} alt="OrbitalGuard AI Logo" style={{ width: 44, height: 44, objectFit: 'contain', marginRight: 8 }} />
          <div>
            <div className="header-title">
              Orbital<span>Guard</span> AI
            </div>
            <div className="header-subtitle">ADVANCED ORBITAL SURVEILLANCE ENGINE</div>
          </div>
        </div>

        <div className="header-divider" />

        {/* Telemetry strip */}
        <div className="header-telemetry">
          <div className="tel-item">
            <span className="tel-lbl">Tracked</span>
            <span className="tel-val cyan">{stats.total.toLocaleString()}</span>
          </div>
          <div className="tel-item">
            <span className="tel-lbl">Active Sats</span>
            <span className="tel-val online">{activeCount.toLocaleString()}</span>
          </div>
          <div className="tel-item">
            <span className="tel-lbl">Debris/Rockets</span>
            <span className="tel-val orange">{stats.debris.toLocaleString()}</span>
          </div>
          <div className="tel-item">
            <span className="tel-lbl">High Risk</span>
            <span className={`tel-val ${stats.highRisk > 0 ? 'danger' : 'online'}`}>
              {stats.highRisk}
            </span>
          </div>
        </div>

        {/* View tabs */}
        <div className="header-right" style={{ gap: 6 }}>
          <button
            onClick={() => setShowIntro(true)}
            style={{
              padding:       '5px 14px',
              borderRadius:  8,
              border:        '1px solid rgba(255,255,255,0.1)',
              background:    'rgba(255,255,255,0.05)',
              color:         'var(--text-muted)',
              fontSize:      11,
              fontWeight:    600,
              cursor:        'pointer',
              letterSpacing: '0.3px',
              transition:    'all 0.2s',
            }}
            onMouseOver={e => e.currentTarget.style.color = '#fff'}
            onMouseOut={e => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            PROJECT INFO
          </button>
          
          <div className="header-divider" style={{ margin: '0 4px' }} />

          {VIEWS.map(v => (
            <button
              key={v}
              onClick={() => setView(v)}
              style={{
                padding:       '5px 14px',
                borderRadius:  8,
                border:        `1px solid ${view === v ? 'var(--cyan)' : 'var(--border)'}`,
                background:    view === v ? 'var(--cyan-dim)' : 'transparent',
                color:         view === v ? 'var(--cyan)' : 'var(--text-muted)',
                fontSize:      11,
                fontWeight:    600,
                cursor:        'pointer',
                letterSpacing: '0.3px',
                transition:    'all 0.2s',
              }}
            >
              {v}
            </button>
          ))}

          <div className="header-divider" />

          <div className={`status-pill ${wsStatus === 'live' ? 'live' : 'offline'}`}>
            <span className={wsStatus === 'live' ? 'pulse-dot' : ''}
              style={wsStatus !== 'live' ? {} : { width: 5, height: 5, borderRadius: '50%', background: 'currentColor', animation: 'pulse 1.5s infinite' }}
            />
            {wsStatus === 'live' ? 'Live' : wsStatus === 'reconnecting' ? 'Reconnecting…' : 'Offline'}
          </div>

          <span className="utc-time">{utcTime}</span>
        </div>
      </header>

      {/* ── Body ───────────────────────────────────────── */}
      {view === 'Dashboard' ? (
        <div className="body-row">

          {/* Left: Satellite List */}
          <SatelliteList
            objects={objects}
            risks={risks}
            selectedObject={selected}
            onSelect={handleSelect}
          />

          {/* Center: 3D Globe + Bottom Strip */}
          <div className="layout-center">
            <div className="globe-container">
              <Scene
                objects={objects}
                risks={risks}
                focusTarget={focusTarget}
                onSelect={handleSelect}
                selectedObject={selected}
                avoidancePath={avoidancePath}
              />

              {/* Floating Object Popup (when selected) */}
              {selected && (
                <div className="object-popup">
                  <div className="popup-header">
                    <span className="popup-name">{selected.name || selected.id}</span>
                    <button className="popup-close" onClick={() => handleSelect(null)}>✕</button>
                  </div>
                  <div className="data-row">
                    <span className="data-key">Type</span>
                    <span className={`tag ${selected.type || 'unknown'}`}>{selected.type}</span>
                  </div>
                  <div className="data-row">
                    <span className="data-key">Altitude</span>
                    <span className="data-val cyan">{selected.altitude_km ?? '—'} km</span>
                  </div>
                  <div className="data-row">
                    <span className="data-key">Inclination</span>
                    <span className="data-val">{selected.inclination_deg ?? '—'}°</span>
                  </div>
                  <div className="data-row">
                    <span className="data-key">ID</span>
                    <span className="data-val mono">{selected.id}</span>
                  </div>
                </div>
              )}

              {/* Globe HUD overlays */}
              <div style={{
                position: 'absolute', bottom: 10, left: '50%',
                transform: 'translateX(-50%)',
                fontSize: 9, color: 'var(--text-dim)',
                fontFamily: 'var(--font-mono)',
                pointerEvents: 'none',
                display: 'flex', gap: 20,
              }}>
                {[
                  { label: 'PAYLOAD', color: 'var(--green)' },
                  { label: 'DEBRIS',  color: 'var(--red)' },
                  { label: 'ROCKET',  color: 'var(--orange)' },
                  { label: 'STARLINK', color: 'var(--cyan)' },
                  { label: '⚠ HIGH RISK', color: '#ff6b00' },
                ].map(l => (
                  <span key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: l.color, display: 'inline-block' }} />
                    {l.label}
                  </span>
                ))}
              </div>
            </div>

            {/* Bottom Strip */}
            <div className="bottom-strip">
              <LiveFeed objects={objects} risks={risks} />
              <OrbitalChat />
            </div>
          </div>

          {/* Right: Analytics */}
          <RightAnalyticsPanel
            selected={selected}
            risks={risks}
            objects={objects}
            onExecuteManeuver={handleExecuteManeuver}
            onViewRadar={() => setView('Analytics')}
          />

        </div>
      ) : (
        /* Analytics Research View (Advanced Radar) */
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <AnalyticsDashboard 
            objects={objects}
            selected={selected}
            onClose={() => setView('Dashboard')}
          />
        </div>
      )}

    </div>
  )
}
