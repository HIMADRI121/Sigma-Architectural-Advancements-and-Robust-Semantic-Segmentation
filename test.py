import os
import torch
import numpy as np
from PIL import Image

# -------- Paths --------
pt_file = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_rgb/00001D_flip_sigma0.00_rgb_class.pt'
save_path = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/ensemble_visuals/00001D_ensemble_vis.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# -------- Color Palette for MFNet (9 classes) --------
palette = [
    (0, 0, 0),          # 0: unlabeled
    (128, 64, 128),     # 1: car
    (220, 20, 60),      # 2: person
    (255, 0, 0),        # 3: bike
    (255, 255, 0),      # 4: curve
    (0, 255, 0),        # 5: car stop
    (0, 128, 255),      # 6: guardrail
    (255, 128, 0),      # 7: color cone
    (128, 0, 255),      # 8: bump
]

# -------- Load and Convert --------
class_mask = torch.load(pt_file).numpy()  # shape: [H, W]

# Convert to RGB
rgb_mask = np.zeros((class_mask.shape[0], class_mask.shape[1], 3), dtype=np.uint8)
for class_id, color in enumerate(palette):
    rgb_mask[class_mask == class_id] = color

# Save as PNG
Image.fromarray(rgb_mask).save(save_path)
print(f"✅ Saved visualized PNG to: {save_path}")
