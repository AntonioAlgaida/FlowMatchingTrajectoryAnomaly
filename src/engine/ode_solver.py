import torch
import numpy as np

class FlowEvaluator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def get_divergence(self, xt, t, context, goal_emb):
        """
        Calculates the exact Trace of the Jacobian (div v) for the 12-dim latent space.
        """
        with torch.set_grad_enabled(True):
            xt = xt.detach().requires_grad_(True)
            t = t.detach().requires_grad_(True)
            
            # Predict velocity (using the new architecture signature)
            v = self.model.flow_head(xt, t, context, goal_emb)
            
            # Exact Trace Calculation for 12 dimensions
            # This is very fast (loop runs only 12 times)
            divergence = 0.0
            for i in range(xt.shape[1]):
                # Compute gradients of the i-th output w.r.t. the input xt
                grad_i = torch.autograd.grad(
                    outputs=v[:, i].sum(), 
                    inputs=xt, 
                    create_graph=True, 
                    retain_graph=True
                )[0]
                # Add the i-th diagonal element (dv_i / dx_i)
                divergence += grad_i[:, i]
                
        return v.detach(), divergence.detach()

    @torch.no_grad()
    def compute_log_likelihood(self, batch, steps=20, return_path=False):
        """
        Integrates backwards from Data (t=1) to Noise (t=0).
        
        Args:
            batch: Data dictionary
            steps: Number of integration steps
            return_path (bool): If True, returns (log_likelihood, path_tensor)
                                path_tensor shape: [Steps+1, B, 12]
        """
        # 1. Setup Data (Target Action is the PCA coefficients)
        x1 = batch['target_action'] # [B, 12]
        device = x1.device
        B = x1.shape[0]
        
        # 2. Encode Context ONCE
        context, goal_emb = self.model.encoder(
            batch['agent_context'], 
            batch['agent_mask'], 
            batch['map_context'], 
            batch['map_mask'],
            batch['goal_pos'],
            batch['goal_lane']
        )
        
        # 3. Initialize ODE at t=1 (The Data)
        curr_x = x1
        curr_logp = torch.zeros(B, device=device)
        dt = 1.0 / steps
        
        # Storage for visualization
        trajectory_path = []
        if return_path:
            trajectory_path.append(curr_x.cpu().clone())

        # 4. Integrate Backwards (t=1 -> t=0)
        for i in range(steps):
            # Time goes 1.0 -> 0.0
            t_val = 1.0 - (i * dt)
            t_tensor = torch.ones(B, device=device) * t_val
            
            # Get velocity and how much the volume expands/contracts (divergence)
            v, div = self.get_divergence(curr_x, t_tensor, context, goal_emb)
            
            # Euler Step backwards
            # dx = v * dt
            # But since we go backwards, it's x_{new} = x - v*dt
            curr_x = curr_x - v * dt
            
            # Update Likelihood
            # Change of variable formula: log p(x1) = log p(x0) - Integral(Trace)
            # Accumulate trace:
            curr_logp = curr_logp + div * dt
            
            if return_path:
                trajectory_path.append(curr_x.cpu().clone())
            
        # 5. Base Probability (Standard Normal at t=0)
        # log N(x; 0, I) = -0.5 * (x^2 + log(2pi))
        # Sum over dimensions
        logp_x0 = -0.5 * (curr_x.pow(2) + torch.log(torch.tensor(2 * torch.pi))).sum(dim=-1)
        
        # Total Log-Likelihood
        # Note: The integral in CNF is usually defined t0->t1. 
        # When integrating backwards t1->t0, the sign matches the formula:
        # log p(x1) = log p(x0) - Integral_0^1 Trace(J) dt
        # Our loop sums (Trace * dt), effectively approximating the integral.
        # Since we want -Integral, and we summed positive terms, we subtract.
        total_log_likelihood = logp_x0 - curr_logp
        
        if return_path:
            # Stack into [Steps+1, B, 12]
            full_path = torch.stack(trajectory_path) 
            return total_log_likelihood, full_path

        return total_log_likelihood