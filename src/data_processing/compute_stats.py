import os
import glob
import numpy as np
import json
import random
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../../configs", config_name="main_config", version_base=None)
def main(cfg: DictConfig):
    # 1. Find all processed training files
    search_path = os.path.join(cfg.data.processed_dir, "training", "**", "*.npz")
    all_files = glob.glob(search_path, recursive=True)
    
    if not all_files:
        print(f"❌ No processed files found in {search_path}")
        return

    # 2. Sample scenarios
    sample_size = min(len(all_files), 5000) 
    sampled_files = random.sample(all_files, sample_size)
    print(f"📊 Computing stats over {sample_size} sampled scenarios...")

    # Accumulators
    agent_vals = [] # To store [x, y, vx, vy, yaw, length, width]
    map_vals = []   # To store [x, y, z]

    for f in tqdm(sampled_files):
        try:
            data = np.load(f)
            
            # --- Agent Stats ---
            # agents shape: [32, 91, 10]
            # mask shape: [32, 91]
            agents = data['agents']
            mask = data['agent_mask'].astype(bool)
            
            # Only take valid steps for physics features (first 7 channels)
            # channels: x, y, vx, vy, yaw, length, width
            valid_agents = agents[mask][:, :7] 
            agent_vals.append(valid_agents)
            
            # --- Map Stats ---
            # map shape: [256, 20, 7]
            # mask shape: [256]
            map_pts = data['map']
            m_mask = data['map_mask'].astype(bool)
            
            # Only take valid polylines for geometry (first 3 channels: x, y, z)
            valid_map = map_pts[m_mask][:, :, :3].reshape(-1, 3)
            map_vals.append(valid_map)
            
        except Exception as e:
            print(f"Skipping {f}: {e}")

    # 3. Concatenate and Compute
    print("Flattening and computing Z-scores...")
    agent_vals = np.concatenate(agent_vals, axis=0)
    map_vals = np.concatenate(map_vals, axis=0)

    stats = {
        "agents": {
            "mean": agent_vals.mean(axis=0).tolist(),
            "std": agent_vals.std(axis=0).tolist(),
            "columns": ["x", "y", "vx", "vy", "yaw", "length", "width"]
        },
        "map": {
            "mean": map_vals.mean(axis=0).tolist(),
            "std": map_vals.std(axis=0).tolist(),
            "columns": ["x", "y", "z"]
        },
        "metadata": {
            "sample_size": sample_size,
            "total_files_searched": len(all_files)
        }
    }

    # 4. Save to JSON
    output_path = os.path.join(os.path.dirname(cfg.data.processed_dir), "stats.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"✅ Statistics saved to: {output_path}")
    
    # Print a quick summary for the user
    print("\nSummary (Mean | Std):")
    for i, col in enumerate(stats['agents']['columns']):
        print(f"Agent {col:8}: {stats['agents']['mean'][i]:.3f} | {stats['agents']['std'][i]:.3f}")

if __name__ == "__main__":
    main()