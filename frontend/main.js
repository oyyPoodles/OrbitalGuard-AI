// OrbitalGuard AI — Production Three.js Frontend
// Connects to FastAPI WebSocket for real-time orbital visualization

const container = document.getElementById('canvas-container');

// ─── Scene ────────────────────────────────────────────────
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 50000);
camera.position.set(0, 5, 25);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 8;
controls.maxDistance = 100;
controls.rotateSpeed = 0.5;

// ─── Starfield ────────────────────────────────────────────
(function createStars() {
    const geo = new THREE.BufferGeometry();
    const verts = new Float32Array(6000);
    for (let i = 0; i < 6000; i++) {
        verts[i] = (Math.random() - 0.5) * 2000;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    scene.add(new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.15, sizeAttenuation: true })));
})();

// ─── Earth ────────────────────────────────────────────────
const loader = new THREE.TextureLoader();
const earthGeo = new THREE.SphereGeometry(6.371, 64, 64);
const earthMat = new THREE.MeshPhongMaterial({
    map: loader.load('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg'),
    bumpMap: loader.load('https://unpkg.com/three-globe/example/img/earth-topology.png'),
    bumpScale: 0.08,
    specular: new THREE.Color(0x222222),
    shininess: 25
});
const earth = new THREE.Mesh(earthGeo, earthMat);
scene.add(earth);

// Atmosphere
const atmosGeo = new THREE.SphereGeometry(6.55, 64, 64);
const atmosMat = new THREE.ShaderMaterial({
    vertexShader: `
        varying vec3 vNormal;
        void main() {
            vNormal = normalize(normalMatrix * normal);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        varying vec3 vNormal;
        void main() {
            float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
            gl_FragColor = vec4(0.3, 0.6, 1.0, 1.0) * intensity;
        }
    `,
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide,
    transparent: true
});
scene.add(new THREE.Mesh(atmosGeo, atmosMat));

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.15));
const sun = new THREE.DirectionalLight(0xffffff, 1.8);
sun.position.set(50, 20, 30);
scene.add(sun);

// ─── InstancedMesh for Objects ────────────────────────────
const MAX_OBJECTS = 3000;
const objGeo = new THREE.SphereGeometry(0.035, 6, 6);
const objMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
const mesh = new THREE.InstancedMesh(objGeo, objMat, MAX_OBJECTS);
mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(mesh);

const dummy = new THREE.Object3D();
const COL_PAYLOAD = new THREE.Color(0x22c55e);
const COL_DEBRIS  = new THREE.Color(0xef4444);
const COL_ROCKET  = new THREE.Color(0xeab308);

// ─── Danger Spheres Group ─────────────────────────────────
const dangerGroup = new THREE.Group();
scene.add(dangerGroup);
const dangerPool = [];
const DANGER_MAT = new THREE.MeshBasicMaterial({ color: 0xff3333, transparent: true, opacity: 0.25, depthWrite: false });
const DANGER_GEO = new THREE.SphereGeometry(0.15, 12, 12);

function getDangerSphere(i) {
    if (i < dangerPool.length) return dangerPool[i];
    const m = new THREE.Mesh(DANGER_GEO, DANGER_MAT);
    dangerPool.push(m);
    dangerGroup.add(m);
    return m;
}

// ─── State ────────────────────────────────────────────────
let objectCache = [];
let riskCache = [];
let wsLatency = 0;
let lastMsgTime = Date.now();

// ─── WebSocket ────────────────────────────────────────────
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
let ws;

function connect() {
    ws = new WebSocket('ws://localhost:8000/ws/live');

    ws.onopen = () => {
        statusDot.classList.add('connected');
        statusText.textContent = 'Live · Streaming';
    };

    ws.onmessage = (e) => {
        const now = Date.now();
        wsLatency = now - lastMsgTime;
        lastMsgTime = now;
        document.getElementById('latency').textContent = wsLatency + ' ms';

        try {
            const data = JSON.parse(e.data);
            if (data.objects) objectCache = data.objects;
            if (data.risks) riskCache = data.risks;
        } catch (err) { /* skip bad frame */ }
    };

    ws.onclose = () => {
        statusDot.classList.remove('connected');
        statusText.textContent = 'Reconnecting...';
        setTimeout(connect, 2000);
    };

    ws.onerror = () => ws.close();
}

// ─── Render Loop ──────────────────────────────────────────
function updateScene() {
    const showPayload = document.getElementById('togglePayload').checked;
    const showDebris = document.getElementById('toggleDebris').checked;
    const showZones = document.getElementById('toggleZones').checked;

    let count = 0;
    let highRiskCount = 0;

    for (let i = 0; i < objectCache.length && count < MAX_OBJECTS; i++) {
        const obj = objectCache[i];
        if (obj.type === 'payload' && !showPayload) continue;
        if (obj.type === 'debris' && !showDebris) continue;

        dummy.position.set(obj.x, obj.y, obj.z);
        dummy.updateMatrix();
        mesh.setMatrixAt(count, dummy.matrix);

        const col = obj.type === 'payload' ? COL_PAYLOAD : obj.type === 'rocket' ? COL_ROCKET : COL_DEBRIS;
        mesh.setColorAt(count, col);
        count++;
    }

    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

    document.getElementById('obj-count').textContent = count.toLocaleString();

    // Danger spheres for HIGH risks
    if (showZones && riskCache) {
        for (let i = 0; i < riskCache.length; i++) {
            if (riskCache[i].risk === 'HIGH') highRiskCount++;
        }
        // Position danger spheres at first few high-risk object positions
        let di = 0;
        for (let i = 0; i < riskCache.length && di < 20; i++) {
            if (riskCache[i].risk !== 'HIGH') continue;
            // Find the object position by ID
            const target = objectCache.find(o => o.id === riskCache[i].a);
            if (target) {
                const sphere = getDangerSphere(di);
                sphere.position.set(target.x, target.y, target.z);
                sphere.visible = true;
                sphere.material.opacity = 0.15 + 0.15 * Math.sin(Date.now() * 0.005);
                di++;
            }
        }
        // Hide unused
        for (let j = di; j < dangerPool.length; j++) {
            dangerPool[j].visible = false;
        }
    } else {
        for (let j = 0; j < dangerPool.length; j++) dangerPool[j].visible = false;
    }

    document.getElementById('risk-count').textContent = highRiskCount;
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    earth.rotation.y += 0.0003;
    updateScene();
    renderer.render(scene, camera);
}

// ─── Resize ───────────────────────────────────────────────
window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
});

// ─── Init ─────────────────────────────────────────────────
connect();
animate();
