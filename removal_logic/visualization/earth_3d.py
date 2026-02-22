import plotly.graph_objects as go
import numpy as np

class PlanetRenderer:
    def __init__(self):
        self.figure = go.Figure()
        self._draw_earth()
        
    def _draw_earth(self):
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
        x = 6371.0 * np.cos(u) * np.sin(v)
        y = 6371.0 * np.sin(u) * np.sin(v)
        z = 6371.0 * np.cos(v)
        self.figure.add_trace(go.Surface(
            x=x, y=y, z=z, colorscale='Blues', opacity=0.4, showscale=False, name='Earth', hoverinfo='none'
        ))
        
    def draw_scene(self, active_pos, debris_pos, rocket_active, rocket_pos, target_idx=None):
        self.figure.data = [self.figure.data[0]] 
        
        # Ensure points are visible: size >= 3, opacity >= 0.7
        if len(active_pos) > 0:
            print(f"[DEBUG-PLOTLY] Plotting {len(active_pos)} Active Satellites.")
            self.figure.add_trace(go.Scatter3d(
                x=active_pos[:,0], y=active_pos[:,1], z=active_pos[:,2],
                mode='markers', marker=dict(size=3, color='lime', opacity=0.7), name='Active Satellites'
            ))
            
        if len(debris_pos) > 0:
            print(f"[DEBUG-PLOTLY] Plotting {len(debris_pos)} Space Debris.")
            if target_idx is not None and target_idx < len(debris_pos):
                target_pos = debris_pos[target_idx]
                mask = np.ones(len(debris_pos), dtype=bool)
                mask[target_idx] = False
                other = debris_pos[mask]
                
                if len(other) > 0:
                    self.figure.add_trace(go.Scatter3d(
                        x=other[:,0], y=other[:,1], z=other[:,2],
                        mode='markers', marker=dict(size=3, color='red', opacity=0.8), name='Space Debris'
                    ))
                self.figure.add_trace(go.Scatter3d(
                    x=[target_pos[0]], y=[target_pos[1]], z=[target_pos[2]],
                    mode='markers', marker=dict(size=8, color='yellow', symbol='diamond', line=dict(width=2, color='white')), name='Hunted Target'
                ))
            else:
                self.figure.add_trace(go.Scatter3d(
                    x=debris_pos[:,0], y=debris_pos[:,1], z=debris_pos[:,2],
                    mode='markers', marker=dict(size=3, color='red', opacity=0.8), name='Space Debris'
                ))

        if rocket_active:
            self.figure.add_trace(go.Scatter3d(
                x=[rocket_pos[0]], y=[rocket_pos[1]], z=[rocket_pos[2]],
                mode='markers', marker=dict(size=8, color='cyan', symbol='cross', line=dict(width=2, color='white')), name='AI Interceptor'
            ))
            
        self.figure.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            paper_bgcolor="black", plot_bgcolor="black"
        )
        return self.figure
