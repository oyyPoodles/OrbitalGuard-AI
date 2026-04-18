import React, { useState, useMemo, useRef, useEffect } from 'react'

const API = 'http://localhost:8000'

/* ─── SatelliteList — Left Sidebar ──────────────────────── */
export default function SatelliteList({ objects, risks, selectedObject, onSelect }) {
  const [search, setSearch]       = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const listRef = useRef()

  const riskSet = useMemo(() => {
    const s = new Set()
    risks.forEach(r => { if (r.risk === 'HIGH') { s.add(r.a); s.add(r.b) } })
    return s
  }, [risks])

  const filtered = useMemo(() => {
    const q = search.toLowerCase()
    return objects.filter(o => {
      const nameMatch = !q || o.name?.toLowerCase().includes(q) || o.id?.toString().includes(q)
      const typeMatch = typeFilter === 'all' || o.type === typeFilter
      return nameMatch && typeMatch
    })
  }, [objects, search, typeFilter])

  // Auto-scroll to selected
  useEffect(() => {
    if (selectedObject && listRef.current) {
      const el = listRef.current.querySelector(`[data-id="${selectedObject.id}"]`)
      if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  }, [selectedObject])

  const counts = useMemo(() => ({
    payload: objects.filter(o => o.type === 'payload').length,
    debris:  objects.filter(o => o.type === 'debris').length,
    rocket:  objects.filter(o => o.type === 'rocket').length,
  }), [objects])

  return (
    <aside className="layout-left">
      {/* Header */}
      <div className="panel-header">
        <div className="panel-title">Tracked Objects</div>
        <input
          className="sat-search"
          type="text"
          placeholder="Search name or ID..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {/* Type Filters */}
      <div className="sat-type-filters">
        {[
          { key: 'all',     label: 'All',     count: objects.length },
          { key: 'payload', label: 'Payload', count: counts.payload },
          { key: 'debris',  label: 'Debris',  count: counts.debris },
          { key: 'rocket',  label: 'Rocket',  count: counts.rocket },
        ].map(f => (
          <button
            key={f.key}
            className={`type-filter-btn ${typeFilter === f.key ? `active-${f.key}` : ''}`}
            onClick={() => setTypeFilter(f.key)}
          >
            {f.label} ({f.count.toLocaleString()})
          </button>
        ))}
      </div>

      <div className="sat-count-bar">
        Showing {filtered.length.toLocaleString()} of {objects.length.toLocaleString()} objects
      </div>

      {/* Virtualized-style list */}
      <div className="sat-list" ref={listRef}>
        {filtered.slice(0, 800).map(obj => {
          const isSelected  = selectedObject?.id === obj.id
          const isHighRisk  = riskSet.has(obj.id?.toString())
          const isStarlink  = obj.name?.toUpperCase().includes('STARLINK')
          const dotType     = isStarlink ? 'payload' : (obj.type || 'unknown')

          return (
            <div
              key={obj.id}
              data-id={obj.id}
              className={`sat-item ${isSelected ? 'selected' : ''} ${isHighRisk ? 'high-risk' : ''}`}
              onClick={() => onSelect(obj)}
              title={`${obj.name} · ID: ${obj.id} · ${obj.type}`}
            >
              <span className={`sat-type-dot ${dotType}`} />
              <span className="sat-name">{obj.name || obj.id}</span>
              <span className="sat-id-badge">{obj.id}</span>
            </div>
          )
        })}
        {filtered.length > 800 && (
          <div className="sat-count-bar" style={{ padding: '8px 16px', textAlign: 'center' }}>
            + {(filtered.length - 800).toLocaleString()} more — refine search
          </div>
        )}
      </div>
    </aside>
  )
}
