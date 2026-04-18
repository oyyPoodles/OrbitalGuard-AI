import React, { useState, useMemo, useRef, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Stars, Html } from '@react-three/drei'
import * as THREE from 'three'

const EARTH_RADIUS = 5
const EARTH_TEXTURE_URL = 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg'

/* ─── Earth (textured, rotating) ────────────────────── */
function Earth() {
  const meshRef  = useRef()
  const cloudRef = useRef()
  const loader   = useMemo(() => {
    const l = new THREE.TextureLoader()
    l.setCrossOrigin('anonymous')
    return l
  }, [])
  const [textures, setTextures] = useState({ day: null })

  useEffect(() => {
    loader.load(
      EARTH_TEXTURE_URL, 
      t => setTextures({ day: t }),
      undefined,
      err => {
        console.error("Earth texture failed to load, falling back to blue globe:", err);
      }
    )
  }, [loader])

  useFrame((_, delta) => {
    if (meshRef.current)  meshRef.current.rotation.y  += delta * 0.03
    if (cloudRef.current) cloudRef.current.rotation.y += delta * 0.036
  })

  return (
    <group>
      <mesh ref={meshRef}>
        <sphereGeometry args={[EARTH_RADIUS, 64, 64]} />
        {textures.day ? (
          <meshPhongMaterial map={textures.day} specular={new THREE.Color(0x333333)} shininess={15} />
        ) : (
          <meshStandardMaterial color="#0c356a" roughness={0.7} metalness={0.1} />
        )}
      </mesh>
      {/* Cloud layer */}
      <mesh ref={cloudRef} scale={[1.012, 1.012, 1.012]}>
        <sphereGeometry args={[EARTH_RADIUS, 48, 48]} />
        <meshPhongMaterial color="#ffffff" transparent opacity={0.06} />
      </mesh>
      {/* Atmosphere glow */}
      <mesh scale={[1.03, 1.03, 1.03]}>
        <sphereGeometry args={[EARTH_RADIUS, 32, 32]} />
        <meshBasicMaterial color="#22d3ee" transparent opacity={0.07} side={THREE.BackSide} />
      </mesh>
    </group>
  )
}

/* ─── Color map ─────────────────────────────────────── */
const COL = {
  payload:  new THREE.Color(0x10b981),   // green
  debris:   new THREE.Color(0xef4444),   // red
  rocket:   new THREE.Color(0xf97316),   // orange
  starlink: new THREE.Color(0x22d3ee),   // cyan
  danger:   new THREE.Color(0xff6b00),   // hot orange
}
const MAX_INST = 6000
const dummy    = new THREE.Object3D()

/* ─── Instanced Debris Cloud ────────────────────────── */
function DebrisGroup({ subset, maxInst, riskSet, selectedId, geometry, baseScale, onSelect, onHover }) {
  const meshRef = useRef()

  useFrame(() => {
    if (!meshRef.current || !subset.length) return
    const count = Math.min(subset.length, maxInst)
    for (let i = 0; i < count; i++) {
      const o = subset[i]
      dummy.position.set(o.x, o.y, o.z)
      let finalScale = baseScale
      const isStarlink = o.name?.toUpperCase().includes('STARLINK')
      if (isStarlink) finalScale = baseScale * 1.4
      dummy.scale.setScalar(o.id === selectedId ? finalScale * 3 : finalScale)
      
      // Face velocity vector so cylinders/rockets point forward
      if (geometry.type === 'CylinderGeometry' && o.vx !== undefined) {
        dummy.lookAt(o.x + o.vx, o.y + o.vy, o.z + o.vz)
        dummy.rotateX(Math.PI / 2) // align cylinder along velocity
      } else {
        dummy.rotation.set(0, 0, 0)
      }

      dummy.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.matrix)

      const col = isStarlink ? COL.starlink : riskSet.has(o.id?.toString()) ? COL.danger : (COL[o.type] || COL.debris)
      meshRef.current.setColorAt(i, col)
    }
    meshRef.current.count = count
    meshRef.current.instanceMatrix.needsUpdate = true
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true
  })

  // Pointer events map directly to instanceId which corresponds to subset array index
  const triggerMap = (e, callback) => {
    e.stopPropagation()
    if (e.instanceId !== undefined && e.instanceId < subset.length) {
      callback(subset[e.instanceId])
    }
  }

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, null, maxInst]}
      onClick={e => triggerMap(e, onSelect)}
      onPointerOver={e => triggerMap(e, onHover)}
      onPointerOut={() => onHover(null)}
    >
      <meshPhongMaterial />
    </instancedMesh>
  )
}

function DebrisCloud({ objects, selectedId, riskSet, onSelect, onHover }) {
  const { payloads, rockets, debris } = useMemo(() => {
    const p = [], r = [], d = []
    for (const o of objects) {
      if (o.type === 'payload' || o.name?.toUpperCase().includes('STARLINK')) p.push(o)
      else if (o.type === 'rocket') r.push(o)
      else d.push(o)
    }
    return { payloads: p, rockets: r, debris: d }
  }, [objects])

  const boxGeo = useMemo(() => {
    const g = new THREE.BoxGeometry(1.2, 0.8, 0.8)
    g.computeBoundingSphere()
    g.boundingSphere.radius = 4000 // Huge radius ensures raycaster checks all instances
    return g
  }, [])
  const cylGeo = useMemo(() => {
    const g = new THREE.CylinderGeometry(0.5, 0.5, 2.5, 8)
    g.computeBoundingSphere()
    g.boundingSphere.radius = 4000
    return g
  }, [])
  const dodGeo = useMemo(() => {
    const g = new THREE.DodecahedronGeometry(0.9, 0)
    g.computeBoundingSphere()
    g.boundingSphere.radius = 4000
    return g
  }, [])

  return (
    <group>
      <DebrisGroup subset={payloads} maxInst={3000} riskSet={riskSet} selectedId={selectedId} 
                   geometry={boxGeo} baseScale={0.08} onSelect={onSelect} onHover={onHover} />
      <DebrisGroup subset={rockets} maxInst={1000} riskSet={riskSet} selectedId={selectedId} 
                   geometry={cylGeo} baseScale={0.07} onSelect={onSelect} onHover={onHover} />
      <DebrisGroup subset={debris} maxInst={4000} riskSet={riskSet} selectedId={selectedId} 
                   geometry={dodGeo} baseScale={0.06} onSelect={onSelect} onHover={onHover} />
    </group>
  )
}

/* ─── Selection Ring ────────────────────────────────── */
function SelectedDot({ obj, riskSet }) {
  const ringRef = useRef()
  useFrame((_, delta) => {
    if (ringRef.current) ringRef.current.rotation.z += delta * 2
  })
  if (!obj) return null

  const isStarlink = obj.name?.toUpperCase().includes('STARLINK')
  const isRisk = riskSet?.has(obj.id?.toString())
  const color = isRisk ? '#ff6b00'
    : isStarlink        ? '#22d3ee'
    : obj.type === 'debris' ? '#ef4444'
    : obj.type === 'rocket' ? '#f97316'
    : '#10b981'

  return (
    <group position={[obj.x, obj.y, obj.z]}>
      <mesh>
        <sphereGeometry args={[0.22, 16, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.42, 0.025, 8, 48]} />
        <meshBasicMaterial color={color} transparent opacity={0.75} />
      </mesh>
      {/* Pulse ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.7, 0.01, 6, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} />
      </mesh>
    </group>
  )
}

/* ─── Orbit Trails ──────────────────────────────────── */
const TRAIL_LEN = 80
const trailCache = {}

function OrbitTrails({ objects, selectedId, riskSet }) {
  const targets = useMemo(() => {
    const sel = objects.filter(o => o.id === selectedId)
    const hr  = objects.filter(o => riskSet.has(o.id?.toString())).slice(0, 5)
    return [...new Map([...sel, ...hr].map(o => [o.id, o])).values()]
  }, [objects, selectedId, riskSet])

  useFrame(() => {
    const activeIds = new Set(targets.map(o => o.id))
    for (const o of targets) {
      if (!trailCache[o.id]) trailCache[o.id] = []
      trailCache[o.id].push([o.x, o.y, o.z])
      if (trailCache[o.id].length > TRAIL_LEN) trailCache[o.id].shift()
    }
    for (const id in trailCache) {
      if (!activeIds.has(id)) delete trailCache[id]
    }
  })

  return (
    <group>
      {targets.map(obj => {
        const hist = trailCache[obj.id]
        if (!hist || hist.length < 4) return null
        const pts  = hist.map(p => new THREE.Vector3(...p))
        const geo  = new THREE.BufferGeometry().setFromPoints(
          new THREE.CatmullRomCurve3(pts).getPoints(100)
        )
        const isRisk = riskSet.has(obj.id?.toString())
        return (
          <line key={obj.id} geometry={geo}>
            <lineBasicMaterial
              color={isRisk ? '#ff6b00'
                : obj.type === 'debris'  ? '#ef4444'
                : obj.type === 'rocket'  ? '#f97316'
                : '#22d3ee'}
              transparent opacity={0.4}
            />
          </line>
        )
      })}
    </group>
  )
}

/* ─── Avoidance Path (Glowing Line) ─────────────────── */
function AvoidancePath({ path }) {
  if (!path) return null
  
  const pts = useMemo(() => {
    return path.points.map(p => new THREE.Vector3(p[0], p[1], p[2]))
  }, [path])

  const geo = useMemo(() => {
    if (pts.length < 2) return null
    return new THREE.BufferGeometry().setFromPoints(
      new THREE.CatmullRomCurve3(pts).getPoints(200)
    )
  }, [pts])

  if (!geo) return null

  return (
    <group>
      <line geometry={geo}>
        <lineBasicMaterial color="#10b981" linewidth={2} transparent opacity={0.9} />
      </line>
      {/* Outer Glow */}
      <line geometry={geo}>
        <lineBasicMaterial color="#10b981" linewidth={4} transparent opacity={0.3} />
      </line>
    </group>
  )
}

/* ─── Smart Camera ──────────────────────────────────── */
function SmartCamera({ target, ctrlRef }) {
  const { camera } = useThree()
  const tgt    = useRef(new THREE.Vector3())
  const active = useRef(false)

  useEffect(() => {
    if (target) {
      tgt.current.set(target.x, target.y, target.z)
      active.current = true
    } else {
      active.current = false
    }
  }, [target])

  useFrame(() => {
    if (!active.current) return
    if (ctrlRef.current && target) {
      // ONLY lerp the pivot target, allowing the user to freely rotate/zoom around it!
      ctrlRef.current.target.lerp(tgt.current, 0.08)
    }
  })
  return null
}

/* ─── Main Scene ─────────────────────────────────────── */
export default function Scene({ objects, risks, focusTarget, onSelect, selectedObject, avoidancePath }) {
  const ctrlRef = useRef()
  const [hovered, setHovered] = useState(null)

  const riskSet = useMemo(() => {
    const s = new Set()
    risks.forEach(r => { if (r.risk === 'HIGH') { s.add(r.a); s.add(r.b) } })
    return s
  }, [risks])

  return (
    <Canvas
      camera={{ fov: 45, near: 0.1, far: 60000, position: [0, 6, 22] }}
      style={{ width: '100%', height: '100%', background: 'transparent' }}
      gl={{ antialias: true, toneMapping: THREE.NoToneMapping, powerPreference: 'high-performance' }}
      dpr={[1, 1.5]}
    >
      <color attach="background" args={['#000000']} />
      <ambientLight intensity={0.9} />
      <directionalLight position={[60, 40, 30]} intensity={2.2} />
      <pointLight position={[-60, -30, -40]} intensity={0.4} color="#0044ff" />

      <Stars radius={500} depth={200} count={7000} factor={3} saturation={0} fade speed={0.4} />

      <OrbitControls
        ref={ctrlRef}
        enableDamping dampingFactor={0.05}
        minDistance={6.5} maxDistance={90}
        rotateSpeed={0.4} zoomSpeed={0.6}
        enablePan={false}
      />
      <SmartCamera target={focusTarget} ctrlRef={ctrlRef} />

      <Earth />
      
      <DebrisCloud
        objects={objects}
        selectedId={selectedObject?.id}
        riskSet={riskSet}
        onSelect={onSelect}
        onHover={(obj) => setHovered(obj)}
      />
      
      {hovered && (
        <Html position={[hovered.x, hovered.y, hovered.z]} center style={{ pointerEvents: 'none', zIndex: 10 }}>
          <div className="glass" style={{ padding: '6px 10px', borderRadius: 8, color: '#fff', fontSize: '11px', whiteSpace: 'nowrap', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            <span style={{ color: 'var(--cyan)', fontWeight: 700 }}>{hovered.name || hovered.id}</span>
            <br/>
            <span style={{ color: 'var(--text-muted)' }}>{hovered.type}</span>
            {hovered.altitude_km && <span> · {(hovered.altitude_km).toFixed(1)} km</span>}
          </div>
        </Html>
      )}

      <SelectedDot obj={selectedObject} riskSet={riskSet} />
      <OrbitTrails objects={objects} selectedId={selectedObject?.id} riskSet={riskSet} />
      <AvoidancePath path={avoidancePath} />
    </Canvas>
  )
}
