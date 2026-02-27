from core.state_manager import StateManager
from collision.detector import CollisionDetector
from collision.predictor import CollisionPredictor

class SimulationEngine:
    """The central loop combining state physics and collision mathematics."""
    def __init__(self, propagator):
        self.state_manager = StateManager(propagator)
        self.detector = CollisionDetector(self.state_manager)
        self.predictor = CollisionPredictor(self.state_manager)
        
        # Ensure initial state propagation
        self.state_manager.step(0)
        
    def tick(self, dt_seconds):
        """Advances the simulation by one frame."""
        # 1. Update Physics
        self.state_manager.step(dt_seconds)
        
        # 2. Detect Collisions using real mathematical Euclidean distance
        current_risks = self.detector.detect_risks()
        
        # 3. Compile Frame Data
        frame_data = {
            'time': self.state_manager.current_time,
            'positions': self.state_manager.positions,
            'velocities': self.state_manager.velocities,
            'history': self.state_manager.history,
            'objects': self.state_manager.objects,
            'active_risks': current_risks
        }
        
        return frame_data
