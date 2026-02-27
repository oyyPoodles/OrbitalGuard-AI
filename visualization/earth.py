import os
import numpy as np
import plotly.graph_objects as go
from PIL import Image

def get_earth_surface():
    """Generates the main Earth globe, attempting to use a texture map."""
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = 6371.0 * np.cos(u) * np.sin(v)
    y = 6371.0 * np.sin(u) * np.sin(v)
    z = 6371.0 * np.cos(v)
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    texture_path = os.path.join(BASE_DIR, 'data', 'earth_texture.jpg')
    
    try:
        img = Image.open(texture_path)
        surface = go.Surface(
            x=x, y=y, z=z,
            surfacecolor=np.zeros_like(x), 
            cmin=0, cmax=1,
            colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']],
            showscale=False, opacity=0.9, hoverinfo='skip'
        )
        surface.update(
            surfacecolor=np.flipud(np.array(img.resize((20, 40)))),
            colorscale='Earth',
        )
    except Exception:
        surface = go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.6, showscale=False, hoverinfo='skip')
        
    return surface
