import os
import glob
import numpy as np
import json
from sklearn.decomposition import PCA
from tqdm import tqdm
import hydra

@hydra.main(config_path="../../configs", config_name="main_config", version_base=None)
def main(cfg):
    # 1. Collect Trajectories
    files = glob.glob(os.path.join(cfg.data.processed_dir, "training", "**", "*.npz"), recursive=True)
    # Sampling 10k scenarios is statistically sufficient for PCA
    files = files[:10000] 
    
    # Handcoded scaling to match the Dataset loader
    SCALE_POS = 50.0

    trajs = []
    print("Collecting expert trajectories for PCA manifold...")
    for f in tqdm(files):
        data = np.load(f)
        # Ego future: [80, 2]
        future = data['agents'][0, 11:91, :2]
        # Fixed Scaling: Meters -> [-2, 2] approx
        future_norm = future / SCALE_POS
        trajs.append(future_norm.flatten()) # [160]

    trajs = np.stack(trajs)

    # 2. Fit PCA
    # You chose 6 components: This is high-compression, very smooth maneuvers.
    n_components = 12 
    pca = PCA(n_components=n_components)
    pca.fit(trajs)
    
    # 3. Compute Coefficient Standard Deviations
    # This is the 'Whitening' step. It ensures each PCA dimension 
    # has a similar scale for the Flow Matching loss.
    coeffs = pca.transform(trajs) # [N, 6]
    stds = np.std(coeffs, axis=0) # [6]
    
    # 4. Save the Basis
    pca_data = {
        "components": pca.components_.tolist(), # [6, 160]
        "mean": pca.mean_.tolist(),             # [160]
        "stds": stds.tolist(),                  # [6]
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "scale_pos": SCALE_POS
    }
    
    output_path = os.path.join(os.path.dirname(cfg.data.processed_dir), "pca_basis.json")
    with open(output_path, 'w') as f:
        json.dump(pca_data, f, indent=4)
        
    print(f"✅ PCA Basis saved to {output_path}")
    print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  - PC{i}: {var*100:.2f}%")

if __name__ == "__main__":
    main()