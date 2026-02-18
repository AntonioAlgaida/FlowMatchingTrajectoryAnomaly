import torch
import numpy as np
import sys
import os

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.encoder import SceneEncoder

def verify():
    # 1. Setup Dummy Config & Model
    class DummyCfg:
        pass
    cfg = DummyCfg()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SceneEncoder(cfg).to(device)
    model.eval()

    B, N_agents, T_hist, D_agent = 4, 32, 11, 10
    N_map, T_pts, D_map = 256, 20, 7
    
    # 2. Create Dummy Tensors
    agent_context = torch.randn(B, N_agents, T_hist, D_agent).to(device)
    map_context = torch.randn(B, N_map, T_pts, D_map).to(device)
    
    # 3. Create Mask with "Empty" Slots
    # Let's say only 5 agents are valid and only 50 map segments are valid
    agent_mask = torch.zeros(B, N_agents, T_hist, dtype=torch.bool).to(device)
    agent_mask[:, :5, :] = True # Only first 5 agents are valid
    
    map_mask = torch.zeros(B, N_map, dtype=torch.bool).to(device)
    map_mask[:, :50] = True # Only first 50 map segments are valid

    # --- THE "STRESS TEST" FOR NaNs ---
    # We inject INFINITY into the padded regions. 
    # If our masking is correct, these Inf values will be ignored 
    # and the output will be finite.
    agent_context[:, 5:, :, :] = float('inf') 
    map_context[:, 50:, :, :] = float('nan')

    print(f"Running Forward Pass with B={B}...")
    
    try:
        with torch.no_grad():
            output = model(agent_context, agent_mask, map_context, map_mask)
        
        # 4. Check Shape
        print(f"✅ Output Shape: {output.shape}") # Expected: [4, 256]
        
        # 5. Check for NaNs/Infs
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        
        if not has_nan and not has_inf:
            print("✅ SANITY CHECK PASSED: No NaNs or Infs in output despite poisoned input.")
        else:
            print("❌ SANITY CHECK FAILED: NaNs or Infs leaked through the mask!")
            if has_nan: print("   -> Found NaNs")
            if has_inf: print("   -> Found Infs")

    except Exception as e:
        print(f"❌ CRASHED during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()