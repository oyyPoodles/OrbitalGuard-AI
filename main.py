import streamlit as st
import time
import os
from core.tle_parser import TLEParser
from core.propagator import Propagator
from simulation.engine import SimulationEngine
from ui.dashboard import render_dashboard

# Setup Page Layout immediately
st.set_page_config(page_title="Space Debris Simulation", layout="wide", page_icon="🛰️")

# Clean UI Styling
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #1e1e24; }
    header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
    h1, h2, h3, h4 { color: #111827 !important; font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 15px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<h1>🛰️ Space Debris Detection and Collision Prevention System</h1>', unsafe_allow_html=True)
st.caption("Real-Time Low Earth Orbit Physics Simulation (SGP4)")

# --- Initialize Simulation Engine (Run Once) ---
@st.cache_resource
def get_simulation_engine():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'tle_data.txt')
    parser = TLEParser(data_path)
    objects = parser.load_objects(max_objects=300)
    
    if not objects:
        st.error("Failed to load TLE data. Please check data/tle_data.txt")
        return None
        
    propagator = Propagator(objects)
    engine = SimulationEngine(propagator)
    return engine

engine = get_simulation_engine()

if engine:
    # --- UI Configuration State ---
    if 'sim_speed' not in st.session_state:
        st.session_state.sim_speed = 10
        
    # --- Step Physics Engine ---
    # Advance time frame based on configured simulation speed
    frame_data = engine.tick(dt_seconds=st.session_state.sim_speed)
    
    # --- Render Dashboard ---
    # render_dashboard now returns state changes from UI inputs
    auto_refresh, speed = render_dashboard(frame_data)
    
    # Store settings for next tick
    st.session_state.sim_speed = speed
    
    # --- Continuous Simulation Loop ---
    if auto_refresh:
        time.sleep(0.1) # Small delay to prevent immediate DOM lockup
        st.rerun()
