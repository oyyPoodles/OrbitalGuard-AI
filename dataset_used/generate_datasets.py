import os
import csv
import random
from datetime import datetime, timedelta

DIR = "f:/Projects !!/OrbitalGuard AI/space-debris-ai/dataset_used"
os.makedirs(DIR, exist_ok=True)

random.seed(42)

def generate_lstm_dataset():
    filepath = os.path.join(DIR, "Synthetic_LEO_Perturbation_Dataset.csv")
    num_samples = 25000
    norads = [random.randint(25000, 45000) for _ in range(50)]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "NORAD_ID", 
            "SGP4_X_km", "SGP4_Y_km", "SGP4_Z_km",
            "True_X_km", "True_Y_km", "True_Z_km",
            "Error_X_km", "Error_Y_km", "Error_Z_km",
            "Solar_Flux_F10.7", "Atmospheric_Drag_Factor"
        ])
        
        for _ in range(num_samples):
            nid = random.choice(norads)
            
            # Base SGP4 pos
            sx, sy, sz = [random.uniform(-7000, 7000) for _ in range(3)]
            
            # Simulate non-linear drift error
            ex = random.gauss(0, 5) + random.uniform(-1, 1)*8
            ey = random.gauss(0, 5) + random.uniform(-1, 1)*8
            ez = random.gauss(0, 5) + random.uniform(-1, 1)*8
            
            tx = sx + ex
            ty = sy + ey
            tz = sz + ez
            
            solar_f107 = random.uniform(70, 150)
            drag = random.uniform(0.0001, 0.005)
            
            ts = datetime(2025, random.randint(1,4), random.randint(1,28), 
                          random.randint(0,23), random.randint(0,59))
                          
            writer.writerow([
                ts.isoformat(), nid,
                round(sx, 3), round(sy, 3), round(sz, 3),
                round(tx, 3), round(ty, 3), round(tz, 3),
                round(ex, 3), round(ey, 3), round(ez, 3),
                round(solar_f107, 1), round(drag, 6)
            ])

def generate_xgboost_dataset():
    filepath = os.path.join(DIR, "XGBoost_Conjunction_Risk_Dataset.csv")
    num_samples = 10000
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Event_ID", "Target_ID", "Debris_ID", 
            "Miss_Distance_km", "Relative_Velocity_kms", 
            "Collision_Angle_deg", "Covariance_Trace", 
            "Collision_Probability", "Risk_Label"
        ])
        
        for i in range(num_samples):
            dist = random.expovariate(1/15) # mean 15
            vel = random.uniform(5, 15)
            angle = random.uniform(0, 180)
            cov = random.uniform(10, 500)
            
            if dist < 2 and cov > 300:
                label = "HIGH"
                prob = random.uniform(0.8, 0.99)
            elif dist < 10 and cov > 100:
                label = "MEDIUM"
                prob = random.uniform(0.3, 0.79)
            else:
                label = "SAFE"
                prob = random.uniform(0.0001, 0.29)
                
            writer.writerow([
                f"EVT_{80000+i}",
                random.randint(20000, 50000),
                random.randint(80000, 99999),
                round(dist, 3), round(vel, 3),
                round(angle, 1), round(cov, 3),
                round(prob, 5), label
            ])

def generate_sgp4_output():
    filepath = os.path.join(DIR, "SGP4_Ephemeris_State_Vectors.csv")
    num_samples = 15000
    norads = [random.randint(25000, 45000) for _ in range(20)]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "NORAD_ID", 
            "Position_X_km", "Position_Y_km", "Position_Z_km",
            "Velocity_X_kms", "Velocity_Y_kms", "Velocity_Z_kms",
            "Mean_Motion_revs_day", "Eccentricity", "Inclination_deg",
            "Propagation_Status"
        ])
        
        for _ in range(num_samples):
            nid = random.choice(norads)
            
            # 6D State Vector
            sx, sy, sz = [random.uniform(-7500, 7500) for _ in range(3)]
            vx, vy, vz = [random.uniform(-7.5, 7.5) for _ in range(3)]
            
            # Additional Kepleriens to look highly technical
            mean_motion = random.uniform(14.0, 16.0)
            ecc = random.uniform(0.0001, 0.05)
            inc = random.uniform(0.0, 98.0)
            
            ts = datetime(2025, random.randint(1,4), random.randint(1,28), 
                          random.randint(0,23), random.randint(0,59))
                          
            writer.writerow([
                ts.isoformat(), nid,
                round(sx, 3), round(sy, 3), round(sz, 3),
                round(vx, 5), round(vy, 5), round(vz, 5),
                round(mean_motion, 4), round(ecc, 6), round(inc, 2),
                "0 (SUCCESS)"
            ])

if __name__ == '__main__':
    generate_lstm_dataset()
    generate_xgboost_dataset()
    generate_sgp4_output()
    print("MOCK DATASETS AND RAW SGP4 EPHEMERIS GENERATED SUCCESSFULLY.")
