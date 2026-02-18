import os
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd

def mine_anomalies(processed_dir):
    files = glob.glob(os.path.join(processed_dir, "validation", "**", "*.npz"), recursive=True)
    results = []

    print(f"Mining {len(files)} validation scenarios for anomalies...")
    for f in tqdm(files):
        data = np.load(f)
        sc_id = str(data['scenario_id'])
        
        # Heuristic 1: Hard Braking (Deceleration > 5.0 m/s^2)
        # Velocity is channel 2 & 3. 
        # Acceleration = delta_v / delta_t
        v = data['agents'][0, :, 2:4] # [91, 2]
        speed = np.linalg.norm(v, axis=1)
        accel = np.diff(speed) / 0.1 # 10Hz sampling
        
        is_hard_brake = np.any(accel < -5.0)
        
        # Heuristic 2: Large Yaw Rate (Swerving)
        yaw = data['agents'][0, :, 4]
        yaw_rate = np.abs(np.diff(yaw) / 0.1)
        is_swerve = np.any(yaw_rate > 1.5) # rad/s
        
        if is_hard_brake or is_swerve:
            results.append({
                'scenario_id': sc_id,
                'hard_brake': is_hard_brake,
                'swerve': is_swerve,
                'path': f
            })

    # Create the folder if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv("data/golden_test_set.csv", index=False)
    print(f"✅ Found {len(df)} candidate anomalies. Saved to data/golden_test_set.csv")

if __name__ == "__main__":
    mine_anomalies("/mnt/d/waymo_datasets/Deep-Flow_Dataset/processed_npz")