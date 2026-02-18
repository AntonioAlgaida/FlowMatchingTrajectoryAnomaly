# src/dataset/waymo_dataset.py
import os
import glob
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def _load_worker(file_path, stats, pca_comp, pca_mean, pca_stds):
    try:
        raw = np.load(file_path)
        
        # 1. Validity Check
        target_mask = raw['agent_mask'][0, 11:91].astype(bool)
        if not np.all(target_mask): return None 

        # Handcoded Scaling (Must match compute_pca.py)
        SCALE_POS = 50.0
        SCALE_VEL = 20.0
        SCALE_DIM = 5.0
        SCALE_YAW = np.pi

        # 2. Normalize History & Map (Fixed Scaling)
        a_raw = raw['agents'][:, :11, :].astype(np.float32)
        a_phys = np.zeros_like(a_raw[:, :, :7])
        a_phys[:, :, 0:2] = a_raw[:, :, 0:2] / SCALE_POS
        a_phys[:, :, 2:4] = a_raw[:, :, 2:4] / SCALE_VEL
        a_phys[:, :, 4]   = a_raw[:, :, 4]   / SCALE_YAW
        a_phys[:, :, 5:7] = a_raw[:, :, 5:7] / SCALE_DIM
        agent_ctx = np.concatenate([a_phys, a_raw[:, :, 7:]], axis=-1)
        
        m_raw = raw['map'].astype(np.float32)
        m_phys = m_raw[:, :, :3] / SCALE_POS
        map_ctx = np.concatenate([m_phys, m_raw[:, :, 3:]], axis=-1)

        # 3. Target Action (Whitened PCA)
        ego_future = raw['agents'][0, 11:91, :2].astype(np.float32)
        target_norm = (ego_future / SCALE_POS).flatten()
        
        # Project and Whiten
        # coeffs = ((target - mean) @ components.T) / stds
        target_centered = torch.from_numpy(target_norm) - pca_mean
        target_pca = torch.matmul(target_centered, pca_comp.t()) 
        # target_whitened = target_pca / (pca_stds + 1e-6)
        
        # 4. Goal Extraction
        goal_pos_norm = raw['agents'][0, 90, :2].astype(np.float32) / SCALE_POS
        
        # --- NEW: CRITICALITY WEIGHTING ---
        # ---  Kinematic Complexity Weighting ---
        
        # Extract Ego Future (Meters)
        # Shape: [80, 2]
        traj = ego_future
        
        # A. Path Tortuosity (Geometry Score)
        # Calculate step-wise distances (Euclidean norm of delta positions)
        deltas = np.linalg.norm(np.diff(traj, axis=0), axis=1) # [79]
        arc_length = np.sum(deltas)
        # Displacement: Distance from start to end
        displacement = np.linalg.norm(traj[-1] - traj[0])
        
        # Tortuosity = Arc / Chord. 
        # Add epsilon to displacement to handle stationary cars (0/0)
        # If car moves < 1.0m, tortuosity is irrelevant (set to 1.0)
        if displacement > 1.0:
            tortuosity = arc_length / (displacement + 1e-6)
        else:
            tortuosity = 1.0
            
        # Map Tortuosity (usually 1.0 to 1.5) to a weight (1.0 to 3.0)
        # Roundabouts often have tortuosity ~1.5 (half circle)
        w_geometry = 1.0 + 2.0 * np.clip(tortuosity - 1.0, 0.0, 1.0)

        # B. Jerk Energy (Dynamics Score)
        # Velocity ~ 1st diff, Accel ~ 2nd diff, Jerk ~ 3rd diff
        # We use 10Hz sampling (dt=0.1), but for relative weighting, raw diffs are fine.
        velocity = np.diff(traj, axis=0)      # [79, 2]
        accel = np.diff(velocity, axis=0)     # [78, 2]
        jerk = np.diff(accel, axis=0)         # [77, 2]
        
        # Mean magnitude of jerk vectors
        mean_jerk = np.mean(np.linalg.norm(jerk, axis=1))
        
        # High jerk implies complex interaction or aggressive driving.
        # Normalize: A jerk of 0.5 m/s^3 is significant.
        w_dynamics = 1.0 + np.clip(mean_jerk / 0.5, 0.0, 4.0)

        # C. Combined Sample Weight
        # We multiply them: A complex shape (Roundabout) driven aggressively (High Jerk) 
        # gets the highest weight.
        sample_weight = w_geometry * w_dynamics
        
        # Soft-Clip to prevent exploding gradients (Max weight ~12.0)
        sample_weight = np.clip(sample_weight, 0.5, 10.0)
        # -----------------------------------------------------------
        
        # --- NEW: LANE-AWARE GOAL ---
        # 1. Get the goal position (t=90)
        goal_pos_raw = raw['agents'][0, 90, :2]
        
        # 2. Find the Map Feature (Lane) closest to this goal
        map_feats = raw['map'] # [256, 20, 7]
        map_mask = raw['map_mask']
        
        # We only care about Lanes (Type 1, 2, 3)
        lane_mask = (map_feats[:, 0, 3] >= 1) & (map_feats[:, 0, 3] <= 3) & map_mask
        if np.any(lane_mask):
            lanes = map_feats[lane_mask] # [N_lanes, 20, 7]
            # Distances from goal to the first point of each lane
            dists = np.linalg.norm(lanes[:, 0, :2] - goal_pos_raw, axis=1)
            best_lane_idx = np.argmin(dists)
            target_lane = lanes[best_lane_idx, :, :2] # [20, 2]
        else:
            # Fallback if no lanes are nearby
            target_lane = np.tile(goal_pos_raw, (20, 1))

        # Normalize the lane coordinates
        target_lane_norm = target_lane / 50.0 
        
        # return {
        #     "agent_context": agent_ctx,
        #     "agent_mask": raw['agent_mask'][:, :11].astype(bool),
        #     "map_context": map_ctx,
        #     "map_mask": raw['map_mask'].astype(bool),
        #     "target_action": target_pca.numpy(), # [12] PCA coeffs
        #     "goal_pos": goal_pos_norm,
        #     "scenario_id": str(raw['scenario_id'])
        # }
                # RETURN A TUPLE, NOT A DICTIONARY
        # Order: 0:agent_ctx, 1:agent_mask, 2:map_ctx, 3:map_mask, 4:target, 5:goal, 6:id
        return (
            agent_ctx.astype(np.float32), 
            raw['agent_mask'][:, :11].astype(bool),
            map_ctx.astype(np.float32),
            raw['map_mask'].astype(bool),
            target_pca.numpy().astype(np.float32), 
            goal_pos_norm.astype(np.float32),
            np.array([sample_weight], dtype=np.float32), # Return as a 1-element array
            target_lane_norm.astype(np.float32), # Lane-aware goal
            str(raw['scenario_id'])
        )
    except Exception: return None
    
class WaymoDataset(Dataset):
    def __init__(self, cfg, split='training', in_memory=True):
        """
        Args:
            cfg: Hydra config
            split: 'training' or 'validation'
            in_memory: If True, load all data into memory (useful for small datasets)
        """
        self.cfg = cfg
        self.base_path = os.path.join(cfg.data.processed_dir, split)
        self.stats_path = os.path.join(os.path.dirname(cfg.data.processed_dir), "stats.json")
        self.pca_path = os.path.join(os.path.dirname(cfg.data.processed_dir), "pca_basis.json")
        self.in_memory = in_memory
        
        with open(self.stats_path, 'r') as f:
            self.stats = json.load(f)
        with open(self.pca_path, 'r') as f:
            pca_data = json.load(f)
            self.pca_comp = torch.tensor(pca_data['components']).float()
            self.pca_mean = torch.tensor(pca_data['mean']).float()
            self.pca_stds = torch.tensor(pca_data['stds']).float()
            
        self.file_list = glob.glob(os.path.join(self.base_path, "**", "*.npz"), recursive=True)
        
        self.file_list = list(np.random.choice(self.file_list, size=min(200000, len(self.file_list)), replace=False))
        
        self.data_cache = []
        if self.in_memory:
            self._load_all_data_parallel()
        
        print(f"✅ {split} split loaded. Count: {len(self.data_cache if self.in_memory else self.file_list)}")

    def _load_all_data_parallel(self):
        print(f"🚀 Parallel Eager Load ({mp.cpu_count()} workers)...")
        
        # Use partial to bake the stats and PCA data into the worker function
        worker_func = partial(_load_worker, stats=self.stats, pca_comp=self.pca_comp, pca_mean=self.pca_mean, pca_stds=self.pca_stds)
        
        # Use imap_unordered for better memory efficiency and progress tracking
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # We use a chunksize to reduce the overhead of passing 100k tasks
            results = list(tqdm(pool.imap_unordered(worker_func, self.file_list, chunksize=10), 
                                total=len(self.file_list)))
        
        # Filter out any None results from failed loads
        self.data_cache = [r for r in results if r is not None]
        
    def __len__(self):
        return len(self.data_cache) if self.in_memory else len(self.file_list)

    # def normalize(self, data, key):
    #     """ Applies Z-score normalization using global stats. """
    #     mean = np.array(self.stats[key]['mean'])
    #     std = np.array(self.stats[key]['std'])
    #     # Add epsilon to std to avoid div by zero
    #     return (data - mean) / (std + 1e-6)

    def __getitem__(self, idx):
        if self.in_memory:
            tup = self.data_cache[idx]
        else:
            # Re-use worker for consistent scaling if not in memory
            tup = _load_worker(self.file_list[idx], self.stats, self.pca_comp, self.pca_mean, self.pca_stds)
            if tup is None: 
                print(f"⚠️  Failed to load {self.file_list[idx]}. Skipping.")
                return self.__getitem__((idx + 1) % len(self))

        # return {
        #     "agent_context": torch.from_numpy(d["agent_context"]),
        #     "agent_mask": torch.from_numpy(d["agent_mask"]),
        #     "map_context": torch.from_numpy(d["map_context"]),
        #     "map_mask": torch.from_numpy(d["map_mask"]),
        #     "target_action": torch.from_numpy(d["target_action"]),
        #     "goal_pos": torch.from_numpy(d["goal_pos"]),
        #     "scenario_id": d["scenario_id"]
        # }
        return {
            "agent_context": torch.from_numpy(tup[0]),
            "agent_mask": torch.from_numpy(tup[1]),
            "map_context": torch.from_numpy(tup[2]),
            "map_mask": torch.from_numpy(tup[3]),
            "target_action": torch.from_numpy(tup[4]),
            "goal_pos": torch.from_numpy(tup[5]),
            "sample_weight": torch.from_numpy(tup[6]).squeeze(), # [1] -> scalar
            "goal_lane": torch.from_numpy(tup[7]), # Lane-aware goal
            "scenario_id": tup[8]
        }
        