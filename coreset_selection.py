import numpy as np
import os
from collections import Counter

def load_feats(feat_folder:str):
    feat_files = [f for f in os.listdir(feat_folder) if "feat.npy" in f]
    feats = []
    for feat_file in sorted(feat_files):
        feat:np.ndarray = np.load(os.path.join(feat_folder, feat_file))
        feats.append(feat / np.mean(feat))
    return np.array(feats)

if __name__ == "__main__":
    feat_folder = "/path/to/observation_data"
    feats = load_feats(feat_folder)
    dataset = "cifar10" # cifar10, isic2019, kvasir_capsule, indoor
    save_folder = "./data/coreset/ours" # cdal, deepfool, uncertainty, ours
    
    os.makedirs(save_folder, exist_ok=True)
    
    # mask by thresholds, and determine if a sample should be selected in certain epoch
    print(feats.shape[1], feats.shape[1] * 0.3)
    mask_lower = feats < 0.1
    mask_higher = feats > 40
    mask_mid = (feats > 0.1) & (feats < 40)
    num_per_pt = np.sum(mask_mid, axis=0).reshape(-1,)
    print(Counter(num_per_pt))
    print(feats[0, :].min(), feats[0, :].max())
    
    # set the threshold and select the coreset
    thresh = 4
    indices = np.where(num_per_pt >= thresh)[0]
    if len(np.where(num_per_pt >= thresh)[0]) / feats.shape[1] > 0.3:
        # if the budget is exceeded, randomly sample from the selected samples
        import random
        indices = random.sample(indices.tolist(), int(feats.shape[1] * 0.3))
    np.save(os.path.join(save_folder, f"{dataset}.npy"), indices)
    print(len(np.where(num_per_pt >= thresh)[0]), len(np.where(num_per_pt >= thresh)[0]) / feats.shape[1], len(indices) / feats.shape[1])
    
    