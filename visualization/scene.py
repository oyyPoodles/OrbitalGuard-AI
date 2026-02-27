import plotly.graph_objects as go
from visualization.earth import get_earth_surface
from visualization.objects import get_object_scatter
from visualization.trails import get_trail_scatter
import numpy as np

def build_scene(frame_data, show_trails=True):
    """Compiles Earth, Objects, and Trails into a final LeoLabs-styled Plotly Figure."""
    fig = go.Figure()
    
    # 1. Add Earth
    fig.add_trace(get_earth_surface())
    
    # 2. Add Objects
    for trace in get_object_scatter(frame_data):
        fig.add_trace(trace)
        
    # 3. Add Trails
    if show_trails:
        for trace in get_trail_scatter(frame_data):
            fig.add_trace(trace)
            
    # Add Stars Background
    np.random.seed(42)
    stars_x = np.random.uniform(-18000, 18000, 500)
    stars_y = np.random.uniform(-18000, 18000, 500)
    stars_z = np.random.uniform(-18000, 18000, 500)
    r_sq = stars_x**2 + stars_y**2 + stars_z**2
    stars_x, stars_y, stars_z = stars_x[r_sq > 7000**2], stars_y[r_sq > 7000**2], stars_z[r_sq > 7000**2]
    fig.add_scatter3d(x=stars_x, y=stars_y, z=stars_z, mode='markers', marker=dict(size=1.0, color='#888', opacity=0.8), hoverinfo='skip')

    # Formatting
    fig.update_layout(
        height=750, 
        margin=dict(l=0, r=0, b=0, t=0), 
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False), 
            aspectmode='cube', 
            bgcolor='#020617' # Deep slate dark
        ),
        paper_bgcolor='#020617',
        plot_bgcolor='#020617',
        showlegend=False
    )
    
    # Time Overlay
    t_str = frame_data['time'].strftime('%Y-%m-%d %H:%M:%S UTC')
    obj_count = len([x for x in frame_data['positions'] if not np.isnan(x[0])])
    
    fig.add_annotation(
        text=f"<b>{t_str}</b><br><i>{obj_count} objects tracking</i>",
        align='right', showarrow=False, xref='paper', yref='paper', x=0.98, y=0.98, 
        font=dict(color="white", size=14, family="sans-serif"),
        bgcolor="rgba(0,0,0,0.5)", bordercolor="rgba(255,255,255,0.2)", borderwidth=1, borderpad=6
    )
    
    return fig
