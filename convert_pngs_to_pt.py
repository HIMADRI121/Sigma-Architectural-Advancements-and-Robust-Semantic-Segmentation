import os
import torch
import numpy as np
from PIL import Image

# ---------- Configuration ----------
input_png_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions_noisy_rgb'  # e.g., predictions like 00001D_sigma0.05_modal.png
output_pt_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_rgb'
os.makedirs(output_pt_dir, exist_ok=True)

# Class color palette
palette = [
    (0, 0, 0),         # 0: unlabeled
    (128, 64, 128),    # 1: car
    (220, 20, 60),     # 2: person
    (255, 0, 0),       # 3: bike
    (255, 255, 0),     # 4: curve
    (0, 255, 0),       # 5: car stop
    (0, 128, 255),     # 6: guardrail
    (255, 128, 0),     # 7: color cone
    (128, 0, 255),     # 8: bump
]
color2class = {color: idx for idx, color in enumerate(palette)}

# ---------- Function to convert RGB mask to class index map ----------
def rgb_to_class(mask):
    mask = np.array(mask)
    class_map = np.zeros(mask.shape[:2], dtype=np.uint8)
    for color, class_id in color2class.items():
        match = np.all(mask == color, axis=-1)
        class_map[match] = class_id
    return class_map

# ---------- Start Conversion ----------
all_files = sorted(f for f in os.listdir(input_png_dir) if f.endswith('.png'))

for fname in all_files:
    path = os.path.join(input_png_dir, fname)
    image = Image.open(path).convert('RGB')
    
    class_map = rgb_to_class(image)  # shape: [H, W]
    tensor = torch.from_numpy(class_map).to(torch.uint8)  # class indices

    # Optional: convert to one-hot (if needed for softmax-based ensemble)
    one_hot = torch.nn.functional.one_hot(tensor.long(), num_classes=len(palette))  # [H, W, C]

    one_hot = one_hot.permute(2, 0, 1).float()  # [C, H, W]

    # Save both (or comment out one of them)
    torch.save(tensor, os.path.join(output_pt_dir, fname.replace('.png', '_class.pt')))   # [H, W]
    torch.save(one_hot, os.path.join(output_pt_dir, fname.replace('.png', '_onehot.pt'))) # [C, H, W]

    print(f"✅ Converted: {fname}")

print(f"\n✅ All PNGs converted to .pt and saved to: {output_pt_dir}")
