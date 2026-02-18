import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon, Circle

# --- CONFIG ---
COLORS = {
    'background': '#ffffff',  # Pure White
    'lane': '#bdc3c7',        # Darker Grey (Concrete)
    'road_edge': '#000000',   # Pure Black (High Contrast)
    'crosswalk': '#9b59b6',   # Purple (Standard "Pedestrian Zone" color in AV tools)
    'ego_body': '#e74c3c',    # Red
    'ego_roof': '#c0392b',    # Darker Red
    'veh_body': '#3498db',    # Blue
    'veh_roof': '#2980b9',    # Darker Blue
    'pedestrian': '#f39c12',  # Orange
    'cyclist': '#27ae60',     # Green
    'headlight': '#f1c40f',   # Yellow
    
    # Traffic Lights (Neon)
    'tl_red': '#ff0000',
    'tl_yellow': '#f39c12',
    'tl_green': '#00ff00',
    
    # Stop Signs
    'stop_sign': '#c0392b',   # Deep Red for Stop Signs
    'stop_text': '#ffffff',   # White for the text (optional)
}

def get_corners(x, y, length, width, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    l2, w2 = length / 2, width / 2
    corners = np.array([[l2, w2], [l2, -w2], [-l2, -w2], [-l2, w2]])
    return (corners @ R.T) + np.array([x, y])

def plot_scenario(data, ax=None, show_future=True, lim=70):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    
    # Unpack Data
    agents = data['agents']
    if agents.ndim == 4: agents = agents[0]; agent_mask = data['agent_mask'][0]; agent_types = data['object_types'][0]; map_feats = data['map'][0]; map_mask = data['map_mask'][0]; sc_id = str(data['scenario_id'][0])
    else: agent_mask = data['agent_mask']; agent_types = data['object_types']; map_feats = data['map']; map_mask = data['map_mask']; sc_id = str(data['scenario_id'])

    # 1. Map
    plot_map(ax, map_feats, map_mask)
    
    # 2. Agents
    plot_agents(ax, agents, agent_mask, agent_types, show_future)
    
    # 3. Polish
    ax.set_aspect('equal')
    ax.set_facecolor(COLORS['background'])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # Faint grid
    ax.grid(True, linestyle=':', alpha=0.3, color='#7f8c8d')
    ax.set_title(f"Scenario: {sc_id}\nEgo-Centric Frame (t=1.1s)", fontsize=14, fontweight='bold', pad=15)
    
    return ax

def plot_map(ax, map_feats, map_mask):
    valid = np.where(map_mask)[0]
    
    col_lanes = []
    col_edges = []
    col_cross = []
    
    # Traffic Light Buckets
    tl_red, tl_green, tl_yellow = [], [], []
    
    # Stop Sign Bucket
    stop_signs = []

    for i in valid:
        poly = map_feats[i]
        xy = poly[:, :2]
        type_idx = int(poly[0, 3])
        
        # Traffic Lights (Check first point)
        # We define them before checking type, because a Lane can HAVE a traffic light
        has_tl = False
        if poly[0, 4] > 0.5: 
            tl_red.append(xy); has_tl = True
        elif poly[0, 6] > 0.5: 
            tl_green.append(xy); has_tl = True
        elif poly[0, 5] > 0.5: 
            tl_yellow.append(xy); has_tl = True
            
        # If it has a TL, we STILL plot the base lane underneath it
        if type_idx == 1: col_lanes.append(xy)
        elif type_idx == 2: col_edges.append(xy)
        elif type_idx == 3: col_cross.append(xy)
        elif type_idx == 4: stop_signs.append(xy[0]) # Take just the first point

    # 1. Base Map Layers
    # Crosswalks: Purple, thick, semi-transparent (Easier to see than white)
    ax.add_collection(LineCollection(col_cross, colors=COLORS['crosswalk'], linewidths=5, alpha=0.5, zorder=1, capstyle='round'))
    
    # Lanes: Dashed Grey
    ax.add_collection(LineCollection(col_lanes, colors=COLORS['lane'], linewidths=1, linestyle='--', zorder=2))
    
    # Road Edges: Solid Black, Thicker (The "Walls")
    ax.add_collection(LineCollection(col_edges, colors=COLORS['road_edge'], linewidths=2.0, zorder=3, capstyle='round'))
    
    # 2. Traffic Lights (The "Neon Glow" Effect)
    # Layer A: Thick, transparent glow
    ax.add_collection(LineCollection(tl_red, colors=COLORS['tl_red'], linewidths=6, alpha=0.3, zorder=4))
    ax.add_collection(LineCollection(tl_green, colors=COLORS['tl_green'], linewidths=6, alpha=0.3, zorder=4))
    ax.add_collection(LineCollection(tl_yellow, colors=COLORS['tl_yellow'], linewidths=6, alpha=0.3, zorder=4))
    
    # Layer B: Thin, solid core
    ax.add_collection(LineCollection(tl_red, colors=COLORS['tl_red'], linewidths=2, alpha=1.0, zorder=5))
    ax.add_collection(LineCollection(tl_green, colors=COLORS['tl_green'], linewidths=2, alpha=1.0, zorder=5))
    ax.add_collection(LineCollection(tl_yellow, colors=COLORS['tl_yellow'], linewidths=2, alpha=1.0, zorder=5))
    
    # NEW: Plot Stop Signs
    # We use a Scatter plot with marker='8' (Octagon)
    if len(stop_signs) > 0:
        stop_signs = np.array(stop_signs)
        # 1. The Red Octagon
        ax.scatter(stop_signs[:,0], stop_signs[:,1], s=150, c=COLORS['stop_sign'], marker='8', zorder=6, edgecolor='white', linewidth=1)
        # 2. (Optional) The Pole (Small black dot under it)
        ax.scatter(stop_signs[:,0], stop_signs[:,1], s=20, c='black', marker='o', zorder=5)

def plot_agents(ax, agents, agent_mask, agent_types, show_future):
    t_curr = 10
    
    for i in range(len(agents)):
        if not agent_mask[i, t_curr]: continue
        
        # Physics
        x, y = agents[i, t_curr, 0:2]
        yaw = agents[i, t_curr, 4]
        l, w = agents[i, t_curr, 5:7]
        obj_type = agent_types[i]
        is_ego = (i == 0)

        # Style Selection
        if is_ego:
            body_col, roof_col = COLORS['ego_body'], COLORS['ego_roof']
            z_order = 20
        elif obj_type == 2: # Ped
            body_col = COLORS['pedestrian']
            z_order = 15
        elif obj_type == 3: # Cyc
            body_col = COLORS['cyclist']
            z_order = 15
        else: # Veh
            body_col, roof_col = COLORS['veh_body'], COLORS['veh_roof']
            z_order = 10

        # --- Trajectories ---
        traj = agents[i, :, :2]
        mask = agent_mask[i]
        hist = traj[:t_curr+1][mask[:t_curr+1]]
        fut = traj[t_curr:][mask[t_curr:]]

        # History: Solid fade line
        ax.plot(hist[:, 0], hist[:, 1], color=body_col, alpha=0.4, linewidth=1.5, zorder=z_order-1)
        
        # Future: Dots
        if show_future and len(fut) > 1:
            ax.scatter(fut[::5, 0], fut[::5, 1], s=8, color=body_col, alpha=0.6, zorder=z_order-1, edgecolor='none')

        # --- The Agent Body ---
        if obj_type == 2: # Pedestrian (Circle)
            ax.add_patch(Circle((x, y), radius=0.6, color=body_col, zorder=z_order, ec='white', lw=1))
        
        else: # Vehicle / Cyclist (Detailed Box)
            corners = get_corners(x, y, l, w, yaw)
            ax.add_patch(Polygon(corners, closed=True, color=body_col, zorder=z_order, ec='black', lw=0.5, alpha=0.9))
            
            # Roof
            roof_corners = get_corners(x, y, l*0.6, w*0.8, yaw)
            ax.add_patch(Polygon(roof_corners, closed=True, color=roof_col, zorder=z_order+1, alpha=0.8))

            # Headlights
            fl, fr = corners[0], corners[1]
            beam_length = 4.0
            beam_angle = np.deg2rad(20)
            
            hl_poly_l = np.array([fl, fl + beam_length * np.array([np.cos(yaw + beam_angle), np.sin(yaw + beam_angle)]), fl + beam_length * np.array([np.cos(yaw), np.sin(yaw)])])
            hl_poly_r = np.array([fr, fr + beam_length * np.array([np.cos(yaw), np.sin(yaw)]), fr + beam_length * np.array([np.cos(yaw - beam_angle), np.sin(yaw - beam_angle)])])

            ax.add_patch(Polygon(hl_poly_l, closed=True, color=COLORS['headlight'], alpha=0.25, zorder=z_order-2, lw=0))
            ax.add_patch(Polygon(hl_poly_r, closed=True, color=COLORS['headlight'], alpha=0.25, zorder=z_order-2, lw=0))