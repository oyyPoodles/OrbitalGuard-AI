import plotly.graph_objects as go
import numpy as np

def get_object_scatter(frame_data):
    """Parses raw physics frame data and outputs separate Scatter3d traces for active sats and debris."""
    pos = frame_data['positions']
    objs = frame_data['objects']
    
    if pos is None or len(pos) == 0:
        return []
        
    names = np.array([o['name'] for o in objs])
    ids = np.array([o['id'] for o in objs])
    types = np.array([o['type'] for o in objs])
    
    # Check for NaNs derived from SGp4 errors
    valid_mask = ~np.isnan(pos[:, 0])
    
    active_mask = (types == 'SAT') & valid_mask
    debris_mask = (types == 'DEB') & valid_mask
    
    active_pos = pos[active_mask]
    debris_pos = pos[debris_mask]
    
    traces = []
    
    if len(active_pos) > 0:
        traces.append(go.Scatter3d(
            x=active_pos[:, 0], y=active_pos[:, 1], z=active_pos[:, 2],
            mode='markers', marker=dict(size=2, color='#00ff00', opacity=0.9, symbol='square'),
            text=ids[active_mask], name="Satellites", hoverinfo='text'
        ))
        
    if len(debris_pos) > 0:
        traces.append(go.Scatter3d(
            x=debris_pos[:, 0], y=debris_pos[:, 1], z=debris_pos[:, 2],
            mode='markers', marker=dict(size=1.5, color='#ff3333', opacity=0.8, symbol='circle'),
            text=ids[debris_mask], name="Debris", hoverinfo='text'
        ))
        
    return traces
