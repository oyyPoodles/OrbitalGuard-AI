"""
STEP 8: 3D Plotly Visualization Dashboard
Renders satellites, debris, predicted paths, and collision events.
"""
import numpy as np
import plotly.graph_objects as go


def create_earth_mesh(radius=6371, resolution=50):
    """Generate a 3D Earth sphere surface."""
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, '#1a3a5c'], [0.5, '#2d6a9f'], [1, '#4da6d1']],
        showscale=False,
        opacity=0.9,
        hoverinfo='skip',
        name='Earth'
    )


def create_object_scatter(objects, obj_type='payload', color='green', symbol='circle', size=3):
    """Create a Scatter3d trace for a given object type."""
    filtered = [o for o in objects if o.get('type') == obj_type and not np.any(np.isnan(o['position']))]
    if not filtered:
        return None

    positions = np.array([o['position'] for o in filtered])
    names = [o.get('name', 'Unknown') for o in filtered]

    return go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode='markers',
        marker=dict(size=size, color=color, symbol=symbol),
        text=names,
        hoverinfo='text',
        name=obj_type.capitalize()
    )


def create_collision_markers(risks, objects_lookup):
    """Create danger sphere markers for high-risk conjunctions."""
    high_risks = [r for r in risks if r.get('risk_level') == 'HIGH']
    if not high_risks:
        return None

    # Use midpoint between the two objects
    xs, ys, zs, texts = [], [], [], []
    for r in high_risks:
        id1, id2 = r['obj1_id'], r['obj2_id']
        if id1 in objects_lookup and id2 in objects_lookup:
            p1 = np.array(objects_lookup[id1]['position'])
            p2 = np.array(objects_lookup[id2]['position'])
            mid = (p1 + p2) / 2
            xs.append(mid[0])
            ys.append(mid[1])
            zs.append(mid[2])
            texts.append(f"⚠️ {r['distance_km']:.2f} km | {r['risk_level']}")

    if not xs:
        return None

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=12, color='red', opacity=0.4, symbol='diamond'),
        text=texts,
        hoverinfo='text',
        name='⚠️ Collision Risk'
    )


def build_dashboard_figure(objects, risks=None):
    """
    Assemble the full 3D visualization.
    
    Args:
        objects: list of dicts with 'name', 'position', 'velocity', 'type', 'id'
        risks: list of risk dicts from the collision pipeline
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Earth
    fig.add_trace(create_earth_mesh())

    # Satellites
    sat_trace = create_object_scatter(objects, 'payload', '#00ff88', 'circle', 3)
    if sat_trace:
        fig.add_trace(sat_trace)

    # Debris
    deb_trace = create_object_scatter(objects, 'debris', '#ff4444', 'circle', 2)
    if deb_trace:
        fig.add_trace(deb_trace)

    # Rocket bodies
    rkt_trace = create_object_scatter(objects, 'rocket', '#ffff00', 'diamond', 4)
    if rkt_trace:
        fig.add_trace(rkt_trace)

    # Collision risk markers
    if risks:
        lookup = {o.get('id', o.get('name')): o for o in objects}
        risk_trace = create_collision_markers(risks, lookup)
        if risk_trace:
            fig.add_trace(risk_trace)

    # Scene styling
    fig.update_layout(
        scene=dict(
            bgcolor='#020617',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        paper_bgcolor='#020617',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(font=dict(color='white'))
    )

    return fig
