import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_processor import DataProcessor
from simulation.propagator import OrbitalPropagator
from datetime import datetime

processor = DataProcessor(
    tle_path="data/tle_data.txt",
    conjunction_path="data/conjuction_data.csv"
)
sats = processor.load_tles()
print("Sats loaded:", len(sats))

propagator = OrbitalPropagator(sats)
now = datetime.utcnow()
ids, pos, vel = propagator.propagate(now)
print("Valid IDs:", len(ids))
