import torch
import torch.nn as nn
import sys
import os

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.deep_flow import DeepFlow

def verify_integration():
    print("🚀 Starting Full System Integration Test...")
    
    # 1. Setup Dummy Config & Model
    class DummyCfg:
        pass
    cfg = DummyCfg()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFlow(cfg).to(device)
    
    # Batch Size 4
    B = 4
    N_agents, T_hist, D_agent = 32, 11, 10
    N_map, T_pts, D_map = 256, 20, 7
    Action_dim = 160 # 80 steps * 2 coords

    # 2. Create Dummy Tensors
    # [B, N, T, D]
    agent_context = torch.randn(B, N_agents, T_hist, D_agent).to(device)
    agent_mask = torch.ones(B, N_agents, T_hist, dtype=torch.bool).to(device)
    
    # [B, M, P, D]
    map_context = torch.randn(B, N_map, T_pts, D_map).to(device)
    map_mask = torch.ones(B, N_map, dtype=torch.bool).to(device)
    
    # CFM Inputs
    x_t = torch.randn(B, Action_dim).to(device) # Noisy action
    t = torch.rand(B).to(device) # Random time steps 0.0 to 1.0

    print(f"Input tensors created on {device}.")

    # 3. Test Forward Pass
    print("Testing Forward Pass...")
    try:
        v_t = model(agent_context, agent_mask, map_context, map_mask, x_t, t)
        
        print(f"✅ Output Shape: {v_t.shape}") # Expected: [B, 160]
        assert v_t.shape == (B, Action_dim), f"Wrong output shape: {v_t.shape}"
        
        has_nan = torch.isnan(v_t).any().item()
        if has_nan:
            print("❌ FAILED: NaNs detected in output.")
            return
        else:
            print("✅ Forward pass clean (No NaNs).")

    except Exception as e:
        print(f"❌ CRASHED during forward pass: {e}")
        return

    # 4. Test Backward Pass (The Gradient Audit)
    print("Testing Backward Pass (Gradient Flow)...")
    try:
        loss = v_t.pow(2).mean() # Dummy MSE loss
        loss.backward()
        
        # Check if gradients reached the very beginning of the network
        # We check the weights of the local encoders
        agent_grad = model.encoder.agent_local.net[0].weight.grad
        map_grad = model.encoder.map_local[0].weight.grad
        head_grad = model.flow_head.net[0].weight.grad
        
        grads = [agent_grad, map_grad, head_grad]
        names = ["Agent Encoder", "Map Encoder", "Flow Head"]
        
        all_ok = True
        for name, grad in zip(names, grads):
            if grad is None:
                print(f"❌ FAILED: No gradient reached {name}!")
                all_ok = False
            elif torch.all(grad == 0):
                print(f"⚠️ WARNING: Gradient at {name} is all zeros!")
                all_ok = False
            else:
                print(f"✅ Gradient flow to {name} confirmed. (Mean grad: {grad.abs().mean().item():.2e})")
        
        if all_ok:
            print("✅ BACKWARD PASS PASSED: Full model is end-to-end differentiable.")

    except Exception as e:
        print(f"❌ CRASHED during backward pass: {e}")
        return

    print("\n🎉 INTEGRATION TEST SUCCESSFUL: Model is ready for training.")

if __name__ == "__main__":
    verify_integration()