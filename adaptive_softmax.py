# -*- coding: utf-8 -*-
import os
import re
import torch
import numpy as np

# ---------- Configuration ----------
num_classes = 9
sigma_values = ['0.01', '0.05', '0.10', '0.20']
sigma_labels = ['σ=0.01', 'σ=0.05', 'σ=0.1', 'σ=0.2']

# ---------- Paths to IoU files ----------
iou_rgb_path = 'iou_rgb.txt'
iou_modal_path = 'iou_modal.txt'

# ---------- Function to parse IoU files ----------
def parse_iou_file(path):
    class_ious = {i: [] for i in range(num_classes)}
    with open(path, 'r') as f:
        lines = f.readlines()
        class_id = 0
        for line in lines:
            if any(lbl in line for lbl in sigma_labels):
                values = re.findall(r'(\d+\.\d+)%', line)
                if len(values) >= len(sigma_values):
                    class_ious[class_id] = list(map(float, values[1:]))  # skip 'Clean'
                    class_id += 1
    return class_ious

# ---------- Parse IoUs ----------
rgb_ious = parse_iou_file(iou_rgb_path)
modal_ious = parse_iou_file(iou_modal_path)

# ---------- Strategy based on robustness ----------
class_strategy = {}
for cls in range(num_classes):
    rgb_vals = rgb_ious.get(cls, [])
    modal_vals = modal_ious.get(cls, [])

    if len(rgb_vals) != len(sigma_values) or len(modal_vals) != len(sigma_values):
        print(f"⚠️ Missing IoUs for class {cls}, defaulting to 'modal'")
        class_strategy[cls] = 'modal'
        continue

    rgb_std = np.std(rgb_vals)
    modal_std = np.std(modal_vals)
    class_strategy[cls] = 'rgb' if rgb_std < modal_std else 'modal'

# ---------- Show class strategies ----------
print("\n📌 Class strategy:")
for k, v in class_strategy.items():
    print(f"Class {k}: use {v}")

# ---------- Dummy logits (replace with real ones during inference) ----------
modal_softmax = {
    '0.01': torch.tensor([0.01, 0.02, 0.30, 0.05, 0.01, 0.20, 0.02, 0.02, 0.37]),
    '0.05': torch.tensor([0.01, 0.01, 0.80, 0.02, 0.01, 0.04, 0.01, 0.01, 0.09]),
    '0.10': torch.tensor([0.02, 0.02, 0.50, 0.10, 0.01, 0.05, 0.02, 0.01, 0.27]),
    '0.20': torch.tensor([0.01, 0.01, 0.30, 0.03, 0.01, 0.10, 0.01, 0.01, 0.52]),
}
rgb_clean_softmax = torch.tensor([0.01, 0.05, 0.90, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

# ---------- Build final softmax ----------
final_softmax = torch.zeros(num_classes)

for cls in range(num_classes):
    if class_strategy[cls] == 'rgb':
        final_softmax[cls] = rgb_clean_softmax[cls]
    else:
        avg = torch.stack([modal_softmax[s][cls] for s in sigma_values]).mean()
        final_softmax[cls] = avg

# ---------- Normalize ----------
final_softmax /= final_softmax.sum()
final_tensor = final_softmax.unsqueeze(0)  # [1, 9]

# ---------- Print and save ----------
print("\n🎯 Final ensembled softmax tensor:")
print(final_tensor)
torch.save(final_tensor, "final_ensemble_tensor.pt")
print("\n💾 Saved to final_ensemble_tensor.pt")
