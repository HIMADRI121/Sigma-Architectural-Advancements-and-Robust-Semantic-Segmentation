# import os
# import numpy as np
# from PIL import Image
# import re
# import time

# # ---------- Paths ----------
# pred_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions_noisy_modal'
# gt_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions'

# # ---------- Class Info ----------
# class_names = [
#     "unlabeled", "car", "person", "bike", "curve",
#     "car stop", "guardrail", "color cone", "bump"
# ]
# num_classes = len(class_names)

# # ---------- Palette and Mapping ----------
# palette = [
#     (0, 0, 0),         # unlabeled
#     (128, 64, 128),    # car
#     (220, 20, 60),     # person
#     (255, 0, 0),       # bike
#     (255, 255, 0),     # curve
#     (0, 255, 0),       # car stop
#     (0, 128, 255),     # guardrail
#     (255, 128, 0),     # color cone
#     (128, 0, 255),     # bump
# ]
# color2class = {color: idx for idx, color in enumerate(palette)}
# color_keys = np.array(list(color2class.keys()))  # shape (9, 3)
# color_values = np.array(list(color2class.values()))

# # ---------- Fast RGB to Class Map ----------
# def rgb_to_class(mask):
#     mask = np.array(mask)
#     reshaped = mask.reshape(-1, 3)  # (H*W, 3)
#     class_map = np.zeros((reshaped.shape[0],), dtype=np.uint8)

#     for color, class_id in zip(color_keys, color_values):
#         matches = np.all(reshaped == color, axis=1)
#         class_map[matches] = class_id

#     return class_map.reshape(mask.shape[:2])

# # ---------- IoU Evaluation ----------
# intersection = np.zeros(num_classes, dtype=np.float64)
# union = np.zeros(num_classes, dtype=np.float64)

# pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])

# # Optional: Limit number of images for testing
# # pred_files = pred_files[:50]

# start_time = time.time()

# for i, fname in enumerate(pred_files):
#     t0 = time.time()
#     gt_base = re.sub(r'_sigma[0-9.]+_modal', '', os.path.splitext(fname)[0])
#     gt_path = os.path.join(gt_dir, gt_base + '.png')

#     if not os.path.exists(gt_path):
#         print(f"❌ Ground truth not found: {gt_path}")
#         continue

#     pred_mask = rgb_to_class(Image.open(os.path.join(pred_dir, fname)).convert('RGB'))
#     gt_mask = rgb_to_class(Image.open(gt_path).convert('RGB'))

#     for cls in range(num_classes):
#         pred_cls = (pred_mask == cls)
#         gt_cls = (gt_mask == cls)

#         inter = np.logical_and(pred_cls, gt_cls).sum()
#         uni = np.logical_or(pred_cls, gt_cls).sum()

#         intersection[cls] += inter
#         union[cls] += uni

#     print(f"✅ Processed {fname} ({i+1}/{len(pred_files)}) in {time.time() - t0:.2f}s")

# # ---------- Compute IoUs ----------
# ious = intersection / (union + 1e-10)
# valid_classes = union > 0
# mean_iou = ious[valid_classes].mean()

# # ---------- Display ----------
# print("\n📊 Per-Class IoU:")
# for i, (cls_name, iou, valid) in enumerate(zip(class_names, ious, valid_classes)):
#     if valid:
#         print(f"{cls_name:12s} : {iou * 100:.2f}%")
#     else:
#         print(f"{cls_name:12s} : N/A (not present in ground truth)")

# print(f"\n📈 Mean IoU: {mean_iou * 100:.2f}%")
# print(f"\n⏱️ Total Time: {time.time() - start_time:.2f} seconds")

from PIL import Image
import numpy as np

pred_path = "/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions_noisy_rgb/00001D_sigma0.00_modal.png"
img = Image.open(pred_path).convert("RGB")
np_img = np.array(img)

# Show all unique RGB tuples
unique_colors = np.unique(np_img.reshape(-1, 3), axis=0)
print("Unique RGB colors in prediction image:")
print(unique_colors)
