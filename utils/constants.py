"""
Shared constants for the OrbitalGuard AI system.
"""

# Earth parameters
EARTH_RADIUS_KM = 6371.0
EARTH_MU = 398600.4418  # km^3/s^2 (standard gravitational parameter)

# Collision thresholds (km)
HIGH_RISK_THRESHOLD = 1.0
MEDIUM_RISK_THRESHOLD = 5.0
DETECTION_RADIUS = 10.0

# Simulation defaults
MAX_OBJECTS = 1500
DEFAULT_TICK_RATE_HZ = 10
WEBSOCKET_PORT = 8000
FRONTEND_PORT = 8001

# Coordinate scale (km → render units for Three.js)
RENDER_SCALE = 1.0 / 1000.0

# Object type constants
TYPE_PAYLOAD = "payload"
TYPE_DEBRIS = "debris"
TYPE_ROCKET = "rocket"
