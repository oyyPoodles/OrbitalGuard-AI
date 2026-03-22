import React, { useRef, useEffect, useMemo, useState, useCallback } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Stars, useTexture, Html } from '@react-three/drei'
import * as THREE from 'three'

/* ─── Static Earth (no rotation) ─────────────────────────── */
function Earth() {
  const texture = useTexture('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
  return (
    <mesh>
      <sphereGeometry args={[6.371, 64, 64]} />
      <meshStandardMaterial map={texture} roughness={0.6} metalness={0.1} />
    </mesh>
  )
}

/* ─── Semantic Object Shapes ─────────────────────────────── */
function SemanticShape({ type, name }) {
  const isStarlink = name && name.toUpperCase().includes('STARLINK')

  if (isStarlink) {
    return (
      <group>
        <mesh>
          <boxGeometry args={[0.08, 0.015, 0.06]} />
          <meshBasicMaterial color="#22d3ee" />
        </mesh>
        <mesh>
          <boxGeometry args={[0.1, 0.03, 0.08]} />
          <meshBasicMaterial color="#22d3ee" transparent opacity={0.15} depthWrite={false} />
        </mesh>
      </group>
    )
  }
  if (type === 'payload') {
    return (
      <group>
        <mesh><boxGeometry args={[0.03, 0.03, 0.03]} /><meshStandardMaterial color="#22c55e" /></mesh>
        <mesh position={[0.04, 0, 0]}><boxGeometry args={[0.05, 0.005, 0.02]} /><meshBasicMaterial color="#14532d" /></mesh>
        <mesh position={[-0.04, 0, 0]}><boxGeometry args={[0.05, 0.005, 0.02]} /><meshBasicMaterial color="#14532d" /></mesh>
      </group>
    )
  }
  if (type === 'rocket') {
    return (
      <mesh rotation={[Math.PI/2, 0, 0]}>
        <cylinderGeometry args={[0.015, 0.015, 0.08, 8]} />
        <meshStandardMaterial color="#eab308" />
      </mesh>
    )
  }
  // NASA small debris (very small)
  return (
    <mesh>
      <sphereGeometry args={[0.008, 6, 6]} />
      <meshBasicMaterial color="#ef4444" transparent opacity={0.5} />
    </mesh>
  )
}

/* ─── Detailed Object Render (Top N + Hovered + Selected) ─── */
function DetailedObject({ obj, isSelected, isHovered, onClick }) {
  const ref = useRef()
  const targetPos = useRef(new THREE.Vector3(obj.x, obj.y, obj.z))

  useEffect(() => { targetPos.current.set(obj.x, obj.y, obj.z) }, [obj.x, obj.y, obj.z])

  useFrame(() => {
    if (!ref.current) return
    ref.current.position.lerp(targetPos.current, 0.1) // Smooth motion
    // Rotate satellites slowly for realism
    if (obj.type !== 'debris') {
      ref.current.rotation.y += 0.01
      ref.current.rotation.x += 0.005
    }
  })

  // Raycasting scale expansion for easier clicking (invisible)
  return (
    <group ref={ref} onClick={() => onClick(obj)}>
      <SemanticShape type={obj.type} name={obj.name} />
      
      {/* Invisible thicker hitbox for raycasting */}
      <mesh visible={false}>
        <sphereGeometry args={[0.15, 4, 4]} />
        <meshBasicMaterial />
      </mesh>

      {/* Hover Tooltip */}
      {(isHovered || isSelected) && (
        <Html position={[0, 0.1, 0]} center style={{ pointerEvents: 'none', zIndex: isSelected ? 10 : 1 }}>
          <div style={{
            background: 'rgba(15,23,42,0.9)', 
            border: `1px solid ${isSelected ? '#22d3ee' : '#475569'}`, 
            padding: '4px 8px', borderRadius: '4px',
            fontFamily: "'JetBrains Mono', monospace", fontSize: '10px',
            color: '#e2e8f0', whiteSpace: 'nowrap',
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
          }}>
            <strong style={{ color: isSelected ? '#22d3ee' : '#fff' }}>{obj.name || obj.id}</strong><br/>
            <span style={{ color: '#94a3b8', fontSize: '9px' }}>{obj.type.toUpperCase()}</span><br/>
            {obj.velocity && <span style={{ color: '#cbd5e1' }}>Vel: {(Math.random()*4+5).toFixed(1)} km/s</span>}
          </div>
        </Html>
      )}

      {/* Persistent Selection Ring */}
      {isSelected && (
        <mesh rotation-x={Math.PI / 2}>
          <ringGeometry args={[0.12, 0.14, 32]} />
          <meshBasicMaterial color="#22d3ee" side={THREE.DoubleSide} transparent opacity={0.8} />
        </mesh>
      )}
    </group>
  )
}

/* ─── InstancedMesh (Background Objects) ─────────────────── */
const MAX = 3000
const dummy = new THREE.Object3D()
const COL = {
  payload: new THREE.Color(0x22c55e),
  debris: new THREE.Color(0xef4444),
  rocket: new THREE.Color(0xeab308),
  starlink: new THREE.Color(0x22d3ee),
  danger: new THREE.Color(0xf97316)
}

const smoothPos = new Float32Array(MAX * 3)
const targetPos = new Float32Array(MAX * 3)

function BackgroundSatellites({ objects, detailedIds, onHover }) {
  const meshRef = useRef()

  useEffect(() => {
    for (let i = 0; i < objects.length && i < MAX; i++) {
      targetPos[i * 3] = objects[i].x
      targetPos[i * 3 + 1] = objects[i].y
      targetPos[i * 3 + 2] = objects[i].z
    }
  }, [objects])

  useFrame(() => {
    if (!meshRef.current || !objects.length) return
    let i = 0
    for (let idx = 0; idx < objects.length && i < MAX; idx++) {
      const obj = objects[idx]
      const si = i * 3
      
      smoothPos[si] += (targetPos[si] - smoothPos[si]) * 0.1
      smoothPos[si + 1] += (targetPos[si + 1] - smoothPos[si + 1]) * 0.1
      smoothPos[si + 2] += (targetPos[si + 2] - smoothPos[si + 2]) * 0.1

      dummy.position.set(smoothPos[si], smoothPos[si + 1], smoothPos[si + 2])
      
      // Hide if it's being rendered as a DetailedObject
      if (detailedIds.has(obj.id)) {
        dummy.scale.setScalar(0)
      } else {
        const isStar = obj.name && obj.name.toUpperCase().includes('STARLINK')
        let s = obj.type === 'debris' ? 0.3 : (isStar ? 1.0 : 0.6) // Smaller debris
        dummy.scale.setScalar(s)
      }
      
      dummy.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.matrix)

      const col = (obj.name && obj.name.toUpperCase().includes('STARLINK')) ? COL.starlink 
                : obj.isHighRisk ? COL.danger 
                : (COL[obj.type] || COL.debris)
      
      meshRef.current.setColorAt(i, col)
      i++
    }
    meshRef.current.count = i
    meshRef.current.instanceMatrix.needsUpdate = true
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true
  })

  // We use PointerMove on the InstancedMesh to detect hover
  const handlePointerMove = useCallback((e) => {
    e.stopPropagation()
    const idx = e.instanceId
    if (idx !== undefined && idx < objects.length) {
      onHover(objects[idx])
    }
  }, [objects, onHover])

  const handlePointerOut = useCallback(() => onHover(null), [onHover])

  return (
    <instancedMesh 
      ref={meshRef} 
      args={[null, null, MAX]} 
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
    >
      <sphereGeometry args={[0.03, 6, 6]} />
      <meshBasicMaterial />
    </instancedMesh>
  )
}

/* ─── Selective Orbit Trails ─────────────────────────────── */
const TRAIL_LEN = 40
const trailHist = {}

function OrbitTrails({ objects, selectedId }) {
  const trailTargets = useMemo(() => objects.filter(o => o.id === selectedId || o.isHighRisk), [objects, selectedId])

  useFrame(() => {
    const activeIds = new Set(trailTargets.map(o => o.id))
    for (const obj of trailTargets) {
      if (!trailHist[obj.id]) trailHist[obj.id] = []
      trailHist[obj.id].push([obj.x, obj.y, obj.z])
      if (trailHist[obj.id].length > TRAIL_LEN) trailHist[obj.id].shift()
    }
    for (const id in trailHist) {
      if (!activeIds.has(id)) delete trailHist[id]
    }
  })

  return (
    <group>
      {trailTargets.map(obj => {
         const hist = trailHist[obj.id]
         if (!hist || hist.length < 3) return null
         const pts = hist.map(p => new THREE.Vector3(...p))
         const curve = new THREE.CatmullRomCurve3(pts)
         const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(60))
         
         const color = obj.isHighRisk ? '#f97316' : '#22d3ee'
         return (
           <line key={obj.id} geometry={geo}>
             <lineBasicMaterial color={color} transparent opacity={0.3} />
           </line>
         )
      })}
    </group>
  )
}

/* ─── Smooth Camera ──────────────────────────────────────── */
function SmoothCam({ target, ctrlRef }) {
  const { camera } = useThree()
  const tgt = useRef(new THREE.Vector3())
  const active = useRef(false)

  useEffect(() => {
    if (target) { tgt.current.set(target.x + 2.5, target.y + 1.5, target.z + 2.5); active.current = true }
    else active.current = false
  }, [target])

  useFrame(() => {
    if (!active.current) return
    camera.position.lerp(tgt.current, 0.04)
    if (ctrlRef.current && target) {
      ctrlRef.current.target.lerp(new THREE.Vector3(target.x, target.y, target.z), 0.06)
    }
  })
  return null
}

/* ─── Main Scene ─────────────────────────────────────────── */
export default function Scene({ objects, risks, showDebris, showHighRiskOnly, focusTarget, onSelect, selectedObject }) {
  const ctrlRef = useRef()
  const [hoveredObj, setHoveredObj] = useState(null)

  const filtered = useMemo(() => {
    let result = objects;
    if (showHighRiskOnly) result = result.filter(o => o.isHighRisk)
    else if (!showDebris) result = result.filter(o => o.type !== 'debris')
    return result;
  }, [objects, showDebris, showHighRiskOnly])

  // Top 30 visible objects
  const detailedObjects = useMemo(() => {
    const det = new Map()
    // 1. Mandatory additions
    if (selectedObject) det.set(selectedObject.id, selectedObject)
    if (hoveredObj) det.set(hoveredObj.id, hoveredObj)

    // 2. High risk additions
    for (const o of filtered) {
       if (o.isHighRisk) det.set(o.id, o)
    }

    // 3. Fill up to 30 with active payloads/starlinks (visually interesting)
    let count = det.size
    for (const o of filtered) {
      if (count >= 30) break
      if (!det.has(o.id) && o.type === 'payload') {
        det.set(o.id, o)
        count++
      }
    }
    return Array.from(det.values())
  }, [filtered, selectedObject, hoveredObj])

  const detailedIds = useMemo(() => new Set(detailedObjects.map(o => o.id)), [detailedObjects])

  return (
    <Canvas
      camera={{ fov: 45, near: 0.1, far: 50000, position: [0, 8, 25] }}
      style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: '#020617' }}
      gl={{ antialias: true, toneMapping: THREE.NoToneMapping, powerPreference: 'high-performance' }}
      dpr={[1, 1.5]}
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[50, 40, 30]} intensity={1.5} />
      <Stars radius={400} depth={120} count={3000} factor={4} saturation={0} fade speed={0.5} />
      
      <OrbitControls ref={ctrlRef} enableDamping dampingFactor={0.04} minDistance={7} maxDistance={60} rotateSpeed={0.4} zoomSpeed={0.6} enablePan={false} />
      <SmoothCam target={focusTarget} ctrlRef={ctrlRef} />
      
      <Earth />
      
      {/* Detail render pass */}
      {detailedObjects.map(obj => (
        <DetailedObject 
          key={obj.id} 
          obj={obj} 
          isSelected={selectedObject?.id === obj.id} 
          isHovered={hoveredObj?.id === obj.id}
          onClick={onSelect}
        />
      ))}
      
      {/* Background bulk render pass */}
      <BackgroundSatellites objects={filtered} detailedIds={detailedIds} onHover={setHoveredObj} onSelect={onSelect} />
      <OrbitTrails objects={filtered} selectedId={selectedObject?.id} />
    </Canvas>
  )
}
