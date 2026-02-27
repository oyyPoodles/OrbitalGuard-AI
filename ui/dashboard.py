import streamlit as st
from visualization.scene import build_scene

def render_dashboard(frame_data):
    """Renders the three main tabs of the Streamlit dashboard."""
    tab1, tab2, tab3 = st.tabs(["🌍 Live Space Map", "🚨 Collision Monitor", "⚙️ System Info"])
    
    with tab1:
        auto_refresh, sim_speed = render_live_map(frame_data)
        
    with tab2:
        render_collision_monitor(frame_data)
        
    with tab3:
        render_system_info(frame_data)
        
    return auto_refresh, sim_speed

def render_live_map(frame_data):
    col_menu, col_map = st.columns([1, 4])
    
    with col_menu:
        with st.container(border=True):
            st.markdown("### 🌌 LEOLABS")
            st.caption("Orbital Guard AI Engine")
            st.markdown("---")
            
            show_trails = st.checkbox("Show Trails", value=True)
            auto_refresh = st.checkbox("Live Simulation", value=True, key="live_sim_toggle")
            
            sim_speed = st.slider("Sim Speed (x)", min_value=1, max_value=100, value=10)
            
            st.markdown("---")
            st.metric("Total Objects", len(frame_data['objects']))
            
            
    with col_map:
        fig = build_scene(frame_data, show_trails=show_trails)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    return auto_refresh, sim_speed

def render_collision_monitor(frame_data):
    st.subheader("🚨 Active Collision Risks")
    risks = frame_data['active_risks']
    
    if not risks:
        st.success("No collision risks detected within 5km.")
        return
        
    high_risks = [r for r in risks if r['risk_level'] == 'HIGH']
    med_risks = [r for r in risks if r['risk_level'] == 'MEDIUM']
    
    c1, c2 = st.columns(2)
    c1.metric("HIGH Risks (< 1km)", len(high_risks))
    c2.metric("MEDIUM Risks (< 5km)", len(med_risks))
    
    st.markdown("---")
    
    for r in risks[:20]: # Show top 20
        color = "red" if r['risk_level'] == 'HIGH' else "orange"
        with st.container(border=True):
            cols = st.columns([3, 1, 1, 1])
            cols[0].markdown(f"**{r['obj1_name']}** ↔ **{r['obj2_name']}**")
            cols[1].markdown(f"**Distance:** {r['distance_km']:.2f} km")
            cols[2].markdown(f"**Rel Vel:** {r['relative_velocity_km_s']:.2f} km/s")
            cols[3].markdown(f":{color}[{r['risk_level']}]")

def render_system_info(frame_data):
    st.subheader("⚙️ Simulation Engine Telemetry")
    
    objs = frame_data['objects']
    sat_count = sum(1 for o in objs if o['type'] == 'SAT')
    deb_count = sum(1 for o in objs if o['type'] == 'DEB')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Processed", len(objs))
    col2.metric("Active Satellites", sat_count)
    col3.metric("Debris Tracked", deb_count)
    
    st.markdown("---")
    st.markdown("""
    ### System Architecture
    - **Propagator**: SGP4 (Vectorized via `sgp4.api.SatrecArray`)
    - **Collision Detector**: Euclidean distance with 50km Altitude Banding filter.
    - **Visualization**: Plotly 3D WebGL Engine
    """)
