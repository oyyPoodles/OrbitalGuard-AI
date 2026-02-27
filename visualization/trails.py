import plotly.graph_objects as go
import numpy as np

def get_trail_scatter(frame_data):
    """Extracts object history and generates multi-segment line traces for orbital trails."""
    history = frame_data['history']
    objs = frame_data['objects']
    
    traces = []
    # Plotly struggles rendering 300 individual line traces simultaneously.
    # We will combine them into a single trace using None-separated coordinates.
    
    x_lines, y_lines, z_lines = [], [], []
    
    for obj in objs:
        hist = history[obj['id']]
        if len(hist) > 1:
            arr = np.array(hist)
            x_lines.extend(arr[:, 0].tolist() + [None])
            y_lines.extend(arr[:, 1].tolist() + [None])
            z_lines.extend(arr[:, 2].tolist() + [None])
            
    if x_lines:
        traces.append(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=1),
            hoverinfo='skip', showlegend=False, name="Trails"
        ))
        
    return traces
