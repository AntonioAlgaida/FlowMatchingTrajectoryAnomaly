import torch
import pandas as pd
import hydra
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import sys
from pathlib import Path
import os
# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.deep_flow import DeepFlow
from src.dataset.waymo_dataset import WaymoDataset
from src.engine.ode_solver import FlowEvaluator

@hydra.main(config_path="../configs", config_name="main_config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Evaluation on {device}...")

    # 1. Load Model
    model = DeepFlow(cfg).to(device)
    checkpoint_path = "checkpoints/best_model.pth"
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 2. Load Data (Validation Split)
    # Ensure in_memory is False to handle large validation sets safely
    val_set = WaymoDataset(cfg, split='validation', in_memory=True) 
    loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
    
    evaluator = FlowEvaluator(model)
    
    results = []
    
    print("Starting Likelihood Estimation...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Compute Likelihoods
            # steps=20 is a good trade-off for accuracy/speed
            log_likelihoods = evaluator.compute_log_likelihood(batch, steps=20)
            
            # Store Results
            ids = batch['scenario_id']
            ll_values = log_likelihoods.cpu().numpy()
            
            for s_id, score in zip(ids, ll_values):
                results.append({
                    "scenario_id": s_id,
                    "log_likelihood": score
                })

    # 3. Save to CSV
    df = pd.DataFrame(results)
    output_path = "data/val_scores.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Scores saved to {output_path}. Total Scenarios: {len(df)}")

if __name__ == "__main__":
    main()