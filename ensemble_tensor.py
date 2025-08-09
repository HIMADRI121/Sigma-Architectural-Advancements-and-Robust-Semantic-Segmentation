# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image

# ---------- Paths ----------
ensemble_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/ensemble_output'  # folder with *_ensemble.png files
output_pt_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/ensemble_output_pt'
os.makedirs(output_pt_dir, exist_ok=True)

# ---------- Color Palette ----------
palette = [
    (0, 0, 0), (128, 64, 128), (220, 20, 60), (255, 0, 0),
    (255, 255, 0), (0, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255)
]
color2class = {color: idx for idx, color in enumerate(palette)}

# ---------- RGB to Class Index ----------
def rgb_to_class(rgb_img):
    arr = np.array(rgb_img)
    class_mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    for color, cls in color2class.items():
        match = np.all(arr == color, axis=-1)
        class_mask[match] = cls
    return torch.tensor(class_mask)

# ---------- Convert All ----------
for fname in sorted(os.listdir(ensemble_dir)):
    if not fname.endswith('_ensemble.png'):
        continue
    path = os.path.join(ensemble_dir, fname)
    rgb_img = Image.open(path).convert('RGB')
    class_mask = rgb_to_class(rgb_img)  # shape: [H, W]
    
    image_id = fname.replace('_ensemble.png', '')
    torch.save(class_mask, os.path.join(output_pt_dir, "{}_ensemble.pt".format(image_id)))


print("All ensemble PNGs converted to .pt in: {}".format(output_pt_dir))

