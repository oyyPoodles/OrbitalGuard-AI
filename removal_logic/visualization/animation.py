import time
from datetime import datetime, timedelta
import numpy as np

class SimulationAnimator:
    def __init__(self, environment, target_selector, trajectory_planner_class):
        self.env = environment
        self.selector = target_selector
        self.PlannerClass = trajectory_planner_class
        
        self.trajectory = None
        self.target_idx = None
        self.mission_active = False
        
        self.current_time = datetime.utcnow()
        self.time_delta = timedelta(seconds=0.5) 
        self.env.rocket_pos = np.array([6600.0, 0.0, 0.0]) 
        
    def engage_target(self):
        if self.mission_active:
            return False, "Mission already in progress."
            
        state = self.env.get_state()
        self.selector.debris_positions = state["debris"]
        self.selector.rocket_pos = self.env.rocket_pos
        
        idx, min_dist = self.selector.select_highest_risk()
        if idx is None:
            return False, "No debris left in orbital parameters."
            
        self.target_idx = idx
        self.env.rocket_target_idx = idx
        
        target_pos = state["debris"][idx]
        self.trajectory = self.PlannerClass(self.env.rocket_pos, target_pos, speed=250.0) 
        
        self.env.rocket_active = True
        self.mission_active = True
        
        return True, f"AI Locked on Debris [{idx}]. Intercept Distance: {min_dist:.2f} km"
        
    def step_frame(self):
        self.env.step_simulation(self.current_time)
        self.current_time += self.time_delta
        msg = "Scanning LEO Environment..."
        
        if self.mission_active and self.trajectory is not None:
            state = self.env.get_state()
            if self.target_idx < len(state["debris"]):
                live_target_pos = state["debris"][self.target_idx]
                self.trajectory.target_pos = live_target_pos
                
                new_pos, capture_complete = self.trajectory.step(self.time_delta.total_seconds())
                self.env.rocket_pos = new_pos
                
                if capture_complete:
                    self.env.remove_debris(self.target_idx)
                    self.mission_active = False
                    self.env.rocket_active = False
                    self.target_idx = None
                    self.env.rocket_target_idx = None
                    self.trajectory = None
                    msg = "⚠️ TARGET SECURED AND REMOVED FROM ORBIT."
                else:
                    msg = "🚀 Rocket En Route... Closing Distance."
            else:
                self.mission_active = False
                msg = "Target Lost."
                
        return msg
