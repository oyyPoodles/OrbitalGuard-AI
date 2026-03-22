import React, { useState, useEffect, useRef, useMemo } from 'react'
import Scene from './components/Scene'
import Dashboard from './components/Dashboard'
import Controls from './components/Controls'

const WS_URL = 'ws://localhost:3001'

export default function App() {
  const [rawObjects, setRawObjects] = useState([])
  const [risks, setRisks] = useState([])
  const [latency, setLatency] = useState(0)
  const [fps, setFps] = useState(60)
  const [connected, setConnected] = useState(false)

  const [showDebris, setShowDebris] = useState(true)
  const [showHighRiskOnly, setShowHighRiskOnly] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [focusTarget, setFocusTarget] = useState(null)
  const [selectedObject, setSelectedObject] = useState(null)

  const lastMsg = useRef(Date.now())
  const frameCount = useRef(0)
  const lastFpsTime = useRef(Date.now())

  // ─── FPS ──────────────────────────────────────────────
  useEffect(() => {
    let rafId
    const tick = () => { frameCount.current++; rafId = requestAnimationFrame(tick) }
    rafId = requestAnimationFrame(tick)
    const interval = setInterval(() => {
      const now = Date.now()
      const elapsed = (now - lastFpsTime.current) / 1000
      if (elapsed > 0) setFps(Math.round(frameCount.current / elapsed))
      frameCount.current = 0
      lastFpsTime.current = now
    }, 1000)
    return () => { cancelAnimationFrame(rafId); clearInterval(interval) }
  }, [])

  // ─── WebSocket ────────────────────────────────────────
  useEffect(() => {
    let ws, timer
    function connect() {
      ws = new WebSocket(WS_URL)
      ws.onopen = () => setConnected(true)
      ws.onmessage = (e) => {
        const now = Date.now()
        setLatency(now - lastMsg.current)
        lastMsg.current = now
        try {
          const data = JSON.parse(e.data)
          if (data.objects) setRawObjects(data.objects)
          if (data.risks) setRisks(data.risks)
        } catch { }
      }
      ws.onclose = () => { setConnected(false); timer = setTimeout(connect, 2000) }
      ws.onerror = () => ws.close()
    }
    connect()
    return () => { ws?.close(); clearTimeout(timer) }
  }, [])

  // ─── Object Enrichment ────────────────────────────────
  // Map high risk status directly to object array for O(1) rendering checks
  const objects = useMemo(() => {
    const riskSet = new Set()
    risks.forEach(r => {
      if (r.risk === 'HIGH') {
        riskSet.add(r.a)
        riskSet.add(r.b)
      }
    })
    return rawObjects.map(o => ({
      ...o,
      isHighRisk: riskSet.has(o.id)
    }))
  }, [rawObjects, risks])

  // ─── Search → Focus ───────────────────────────────────
  useEffect(() => {
    if (!searchQuery) { setFocusTarget(null); return }
    const q = searchQuery.toLowerCase()
    const target = objects.find(o =>
      o.id === searchQuery ||
      (o.name && o.name.toLowerCase().includes(q))
    )
    if (target) { setFocusTarget(target); setSelectedObject(target) }
  }, [searchQuery, objects])

  const handleSelect = (obj) => { setSelectedObject(obj); setFocusTarget(obj) }

  // ─── Data Export ──────────────────────────────────────
  const handleExport = () => {
    const payload = JSON.stringify({
      timestamp: new Date().toISOString(),
      stats: {
        total_objects: objects.length,
        high_risk_events: risks.filter(r => r.risk === 'HIGH').length
      },
      risks: risks
    }, null, 2)
    
    const blob = new Blob([payload], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `orbitalguard_risks_${Date.now()}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <>
      <Scene
        objects={objects} risks={risks}
        showDebris={showDebris} showHighRiskOnly={showHighRiskOnly}
        focusTarget={focusTarget} onSelect={handleSelect} selectedObject={selectedObject}
      />

      <Controls
        showDebris={showDebris} setShowDebris={setShowDebris}
        showHighRiskOnly={showHighRiskOnly} setShowHighRiskOnly={setShowHighRiskOnly}
        searchQuery={searchQuery} setSearchQuery={setSearchQuery}
        onExport={handleExport}
      />

      <Dashboard objects={objects} risks={risks} latency={latency} fps={fps} />

      {/* ── Selected Object Info ──────────── */}
      {selectedObject && (
        <div className="panel info-panel">
          <div className="info-header">
            <span className="info-title">🛰 {selectedObject.name || selectedObject.id}</span>
            <button className="info-close" onClick={() => { setSelectedObject(null); setFocusTarget(null) }}>✕</button>
          </div>
          <div className="info-row"><span className="info-label">ID</span><span className="info-val mono">{selectedObject.id}</span></div>
          <div className="info-row"><span className="info-label">Type</span><span className={`info-val tag ${selectedObject.type}`}>{selectedObject.isHighRisk ? '⚠ HIGH RISK' : selectedObject.type}</span></div>
          <div className="info-row"><span className="info-label">Position</span><span className="info-val mono">({selectedObject.x?.toFixed(1)}, {selectedObject.y?.toFixed(1)}, {selectedObject.z?.toFixed(1)})</span></div>
        </div>
      )}

      {/* ── Status Bar ────────────────────── */}
      <div className="status-bar">
        <span className={`status-dot ${connected ? 'connected' : ''}`} />
        <span>{connected ? 'Live · Streaming' : 'Reconnecting...'}</span>
      </div>
    </>
  )
}
