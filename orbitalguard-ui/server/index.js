/**
 * OrbitalGuard AI — Node.js WebSocket Proxy
 * Connects to FastAPI backend (ws://localhost:8000/ws/live)
 * and re-broadcasts to React frontend clients on port 3001.
 */
const WebSocket = require('ws');

const FASTAPI_WS = 'ws://localhost:8000/ws/live';
const PROXY_PORT = 3001;

// ─── Proxy Server ─────────────────────────────────────────
const wss = new WebSocket.Server({ port: PROXY_PORT });
console.log(`🚀 OrbitalGuard proxy listening on ws://localhost:${PROXY_PORT}`);

let latestData = null;
let upstreamConnected = false;

// ─── Connect to FastAPI upstream ──────────────────────────
function connectUpstream() {
    const upstream = new WebSocket(FASTAPI_WS);

    upstream.on('open', () => {
        upstreamConnected = true;
        console.log('✅ Connected to FastAPI backend');
    });

    upstream.on('message', (data) => {
        const str = data.toString();
        latestData = str;

        // Broadcast to all connected React clients
        wss.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(str);
            }
        });
    });

    upstream.on('close', () => {
        upstreamConnected = false;
        console.log('⚠️  Upstream disconnected. Reconnecting in 3s...');
        setTimeout(connectUpstream, 3000);
    });

    upstream.on('error', (err) => {
        console.error('Upstream error:', err.message);
        upstream.close();
    });
}

// ─── Handle React client connections ──────────────────────
wss.on('connection', (ws) => {
    console.log(`📡 React client connected (total: ${wss.clients.size})`);

    // Send latest cached data immediately so client isn't blank
    if (latestData) {
        ws.send(latestData);
    }

    ws.on('close', () => {
        console.log(`📡 React client disconnected (total: ${wss.clients.size})`);
    });
});

// ─── Start ────────────────────────────────────────────────
connectUpstream();
