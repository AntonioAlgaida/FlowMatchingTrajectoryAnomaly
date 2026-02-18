import os
import glob
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from functools import partial
import traceback
import hydra
from omegaconf import DictConfig

# Waymo Imports
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import map_pb2

class WaymoParser:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t_curr = cfg.data.current_time_index # 10

    def get_rotation_matrix(self, yaw):
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([[c, -s], [s, c]])

    def wrap_angle(self, angle):
        """Standardize angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def extract_agents(self, scenario, ego_id):
        # 1. Find Ego Track
        try:
            # Note: We assume ego_id passed here is the REAL ID, not the index
            ego_track = next(t for t in scenario.tracks if t.id == ego_id)
        except StopIteration:
            print(f"Ego ID {ego_id} not found in scenario {scenario.scenario_id}")
            return None, None, None, None, None

        ego_state = ego_track.states[self.t_curr]
        if not ego_state.valid:
            print(f"Ego state invalid at t={self.t_curr} in scenario {scenario.scenario_id}")
            return None, None, None, None, None

        # Ego Global Pose
        ego_pos = np.array([ego_state.center_x, ego_state.center_y])
        ego_yaw = self.wrap_angle(ego_state.heading)
        R = self.get_rotation_matrix(-ego_yaw) # Inverse rotation for global->local

        # 2. Filter Neighbors
        relevant_agents = []
        for track in scenario.tracks:
            if not track.states[self.t_curr].valid: continue
            
            s = track.states[self.t_curr]
            pos = np.array([s.center_x, s.center_y])
            dist = np.linalg.norm(pos - ego_pos)
            relevant_agents.append({'track': track, 'dist': dist})

        relevant_agents.sort(key=lambda x: x['dist'])
        relevant_agents = relevant_agents[:self.cfg.data.max_agents]

        # 3. Construct Tensor
        num_agents = self.cfg.data.max_agents
        total_steps = self.cfg.data.total_steps
        
        agent_tensor = np.zeros((num_agents, total_steps, 10), dtype=np.float32)
        valid_mask = np.zeros((num_agents, total_steps), dtype=bool)
        track_ids = np.zeros((num_agents,), dtype=np.int32)
        agent_types = np.zeros((num_agents,), dtype=np.int32)
        
        for i, item in enumerate(relevant_agents):
            track = item['track']
            track_ids[i] = track.id
            agent_types[i] = track.object_type
            
            # One-Hots
            is_veh = 1.0 if track.object_type == scenario_pb2.Track.TYPE_VEHICLE else 0.0
            is_ped = 1.0 if track.object_type == scenario_pb2.Track.TYPE_PEDESTRIAN else 0.0
            is_cyc = 1.0 if track.object_type == scenario_pb2.Track.TYPE_CYCLIST else 0.0
            
            for t in range(total_steps):
                state = track.states[t]
                if not state.valid: continue
                
                g_pos = np.array([state.center_x, state.center_y])
                g_vel = np.array([state.velocity_x, state.velocity_y])
                g_yaw = self.wrap_angle(state.heading)
                
                # Transform to Ego Frame
                l_pos = (g_pos - ego_pos) @ R.T
                l_vel = g_vel @ R.T
                l_yaw = self.wrap_angle(g_yaw - ego_yaw)

                agent_tensor[i, t, 0:2] = l_pos
                agent_tensor[i, t, 2:4] = l_vel
                agent_tensor[i, t, 4] = l_yaw
                agent_tensor[i, t, 5:7] = [state.length, state.width]
                agent_tensor[i, t, 7] = is_veh
                agent_tensor[i, t, 8] = is_ped
                agent_tensor[i, t, 9] = is_cyc
                
                valid_mask[i, t] = True

        return agent_tensor, valid_mask, (ego_pos, ego_yaw, R), track_ids, agent_types

    def extract_map(self, scenario, ego_transform):
        ego_pos, ego_yaw, R = ego_transform
        
        # TL Lookup
        tl_states = {}
        if scenario.dynamic_map_states:
            dynamic_state = scenario.dynamic_map_states[self.t_curr]
            for ls in dynamic_state.lane_states:
                s = ls.state
                is_red = 1.0 if s in [1, 4, 7] else 0.0 
                is_yellow = 1.0 if s in [2, 5, 8] else 0.0
                is_green = 1.0 if s in [3, 6, 9] else 0.0
                tl_states[ls.lane] = [is_red, is_yellow, is_green]

        map_elements = []

        for feature in scenario.map_features:
            # We must use .WhichOneof to robustly identify the type
            feature_type_str = feature.WhichOneof('feature_data')
            if feature_type_str is None: 
                print(f"Unknown map feature in scenario {scenario.scenario_id}")
                continue
            
            feature_data = getattr(feature, feature_type_str)
            poly = None
            type_id = -1 # Filtered out by default

            # --- MAPPING LOGIC (Based on your Dictionary) ---
            if feature_type_str == 'lane':
                # 1-9: LaneCenter
                type_id = feature_data.type # 1, 2, 3
                poly = feature_data.polyline
            
            elif feature_type_str == 'road_line':
                # 10-19: RoadLine
                if feature_data.type != map_pb2.RoadLine.TYPE_UNKNOWN:
                    type_id = feature_data.type + 10 
                    poly = feature_data.polyline

            elif feature_type_str == 'road_edge':
                # 20-29: RoadEdge
                if feature_data.type != map_pb2.RoadEdge.TYPE_UNKNOWN:
                    type_id = feature_data.type + 20
                    poly = feature_data.polyline
            
            elif feature_type_str == 'stop_sign':
                # 31: StopSign (Point)
                type_id = 31
                p = feature_data.position
                # Normalize point to a list for uniform handling
                poly = [p] 
            
            elif feature_type_str == 'crosswalk':
                # 41: Crosswalk
                type_id = 41
                poly = feature_data.polygon
            
            elif feature_type_str == 'speed_bump':
                # 51: SpeedBump
                type_id = 51
                poly = feature_data.polygon
                
            elif feature_type_str == 'driveway':
                # 61: Driveway
                type_id = 61
                poly = feature_data.polygon
            
            else:
                # Unhandled feature type
                print(f"Unhandled map feature type: {feature_type_str} in scenario {scenario.scenario_id}")
                continue

            # Validation
            if type_id == -1 or poly is None or len(poly) == 0:
                print(f"Skipping invalid map feature of type {feature_type_str} in scenario {scenario.scenario_id}")
                continue
            
            # Special case: Stop Sign is 1 point, others need at least 2
            if type_id != 31 and len(poly) < 2:
                # print(f"Skipping short polyline for feature type {feature_type_str} in scenario {scenario.scenario_id}")
                continue

            # Calculate Distance for sorting
            p0 = np.array([poly[0].x, poly[0].y])
            dist = np.linalg.norm(p0 - ego_pos)
            tl_vec = tl_states.get(feature.id, [0.0, 0.0, 0.0])
            
            map_elements.append({
                'poly': poly, 
                'type': type_id, 
                'dist': dist, 
                'tl': tl_vec
            })

        # Sort and Crop
        map_elements.sort(key=lambda x: x['dist'])
        map_elements = map_elements[:self.cfg.data.max_map_elements]

        # Tensorize
        num_map = self.cfg.data.max_map_elements
        pts = self.cfg.data.map_points_per_element
        map_tensor = np.zeros((num_map, pts, 7), dtype=np.float32)
        valid_mask = np.zeros((num_map), dtype=bool)

        for i, item in enumerate(map_elements):
            raw_pts = np.array([[p.x, p.y, p.z] for p in item['poly']])
            
            # Transform Logic
            if item['type'] == 31:
                # STOP SIGN: Repeat the single point
                xy_local = (raw_pts[0, :2] - ego_pos) @ R.T
                map_tensor[i, :, 0] = xy_local[0]
                map_tensor[i, :, 1] = xy_local[1]
                map_tensor[i, :, 2] = raw_pts[0, 2]
                map_tensor[i, :, 3] = 31
            else:
                # LINES: Resample
                diffs = np.linalg.norm(raw_pts[1:] - raw_pts[:-1], axis=1)
                cum_dist = np.concatenate(([0], np.cumsum(diffs)))
                total_len = cum_dist[-1]
                
                if total_len == 0: continue
                
                target_dists = np.linspace(0, total_len, pts)
                new_x = np.interp(target_dists, cum_dist, raw_pts[:,0])
                new_y = np.interp(target_dists, cum_dist, raw_pts[:,1])
                new_z = np.interp(target_dists, cum_dist, raw_pts[:,2])
                
                xy_local = (np.stack([new_x, new_y], axis=1) - ego_pos) @ R.T
                
                map_tensor[i, :, 0] = xy_local[:, 0]
                map_tensor[i, :, 1] = xy_local[:, 1]
                map_tensor[i, :, 2] = new_z
                map_tensor[i, :, 3] = item['type']
            
            map_tensor[i, :, 4:] = item['tl']
            valid_mask[i] = True
            
        return map_tensor, valid_mask

    def parse_scenario(self, raw_data):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw_data)
        
        if len(scenario.tracks) == 0: return None

        # Resolve SDC ID
        try:
            ego_id = scenario.tracks[scenario.sdc_track_index].id
        except IndexError:
            return None

        # Extract Agents
        agents, agent_mask, transform, track_ids, agent_types = self.extract_agents(scenario, ego_id)
        if agents is None: return None

        # Global Pose for Reconstruction
        global_pose = np.array([transform[0][0], transform[0][1], transform[1]], dtype=np.float32)

        # Extract Map with Granular Types
        map_feats, map_mask = self.extract_map(scenario, transform)
        
        return {
            "scenario_id": scenario.scenario_id,
            "agents": agents,         # [32, 91, 10]
            "agent_mask": agent_mask, # [32, 91]
            "track_ids": track_ids,   # [32]
            "object_types": agent_types, # [32]
            "map": map_feats,         # [256, 20, 7]
            "map_mask": map_mask,     # [256]
            "global_pose": global_pose
        }

# --- PROCESS FUNCTION ---
def process_single_file(file_path, cfg):
    parser = WaymoParser(cfg)
    
    # 1. Determine base output directory
    rel_path = os.path.relpath(file_path, cfg.data.raw_dir)
    subdir = os.path.dirname(rel_path) 
    base_output_dir = os.path.join(cfg.data.processed_dir, subdir)
    
    raw_dataset = tf.data.TFRecordDataset(file_path, compression_type='')
    count = 0
    
    for raw_bytes in raw_dataset:
        try:
            data = parser.parse_scenario(raw_bytes.numpy())
            if data is None: continue
            
            sc_id = data['scenario_id']
            sub_shard = sc_id[:2] # e.g., 'a1'
            
            # Robust Directory Creation for WSL
            final_dir = os.path.join(base_output_dir, sub_shard)
            if not os.path.exists(final_dir):
                try:
                    os.makedirs(final_dir, exist_ok=True)
                except FileExistsError:
                    pass # Race condition solved
                except FileNotFoundError:
                    # Fallback: Sometimes WSL needs a retry on deep paths
                    import time
                    time.sleep(0.1)
                    os.makedirs(final_dir, exist_ok=True)
            
            output_path = os.path.join(final_dir, f"{sc_id}.npz")
            
            if os.path.exists(output_path): continue

            np.savez_compressed(output_path, **data)
            count += 1
            
        except Exception as e:
            # Suppress non-critical errors to keep logs clean
            # print(f"Error in {sc_id}: {e}")
            print(traceback.format_exc())
            continue

    return f"Processed {os.path.basename(file_path)}: Saved {count}"

@hydra.main(config_path="../../configs", config_name="main_config", version_base=None)
def main(cfg: DictConfig):
    input_pattern = os.path.join(cfg.data.raw_dir, "**", "*.tfrecord*")
    files = glob.glob(input_pattern, recursive=True)
    
    if not files:
        print(f"No files found in {cfg.data.raw_dir}")
        return

    print(f"Found {len(files)} files.")
    
    # --- ROBUST PRE-CREATION LOOP ---
    print("Pre-creating output directories...")
    
    # 1. Identify all 'split' folders (training/validation)
    subdirs = set(os.path.dirname(os.path.relpath(f, cfg.data.raw_dir)) for f in files)
    
    for s in subdirs:
        base = os.path.join(cfg.data.processed_dir, s)
        
        # Create the base training/val folder
        if not os.path.exists(base):
            try:
                os.makedirs(base, exist_ok=True)
            except FileExistsError:
                pass
        
        # 2. Pre-create shards for '00' through 'ff' (and '0' through '9' just in case)
        # We iterate 0..255 to cover all hex possibilities (00, 01 ... fe, ff)
        # This covers Waymo IDs that start with 'a1', 'b2', '10', etc.
        for i in range(256):
            shard_name = f"{i:02x}" # Hex string (e.g., '10', 'a5')
            shard_path = os.path.join(base, shard_name)
            
            # DEFENSIVE CHECK: Only try to create if python thinks it's missing
            if not os.path.exists(shard_path):
                try:
                    os.makedirs(shard_path, exist_ok=True)
                except FileExistsError:
                    # If we get here, it means it DOES exist (race condition or file system lag)
                    # We check if it is a directory. If so, we are good.
                    if os.path.isdir(shard_path):
                        pass
                    else:
                        print(f"⚠️ WARNING: Name collision at {shard_path}. A file exists with this name?")
                        
    print("Directories ready. Starting Parallel Processing...")

    worker_func = partial(process_single_file, cfg=cfg)

    # Use tqdm for progress bar
    from tqdm import tqdm
    with mp.Pool(processes=cfg.processing.n_workers) as pool:
        list(tqdm(pool.imap_unordered(worker_func, files), total=len(files)))

if __name__ == "__main__":
    main()