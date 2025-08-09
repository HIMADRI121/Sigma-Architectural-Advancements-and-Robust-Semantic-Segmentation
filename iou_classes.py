
#for rgb
# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import re
# import time

# # ---------- Paths ----------
# gt_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/Label_Colored'
# pred_noisy_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions_noisy_rgb'

# # ---------- Class Info ----------
# class_names = [
#     "unlabeled", "car", "person", "bike", "curve",
#     "car stop", "guardrail", "color cone", "bump"
# ]
# num_classes = len(class_names)

# # ---------- Palette ----------
# palette = [
#     (0, 0, 0),         # 0: unlabeled
#     (128, 64, 128),    # 1: car
#     (220, 20, 60),     # 2: person
#     (255, 0, 0),       # 3: bike
#     (255, 255, 0),     # 4: curve
#     (0, 255, 0),       # 5: car stop
#     (0, 128, 255),     # 6: guardrail
#     (255, 128, 0),     # 7: color cone
#     (128, 0, 255),     # 8: bump
# ]
# color2class = {color: idx for idx, color in enumerate(palette)}
# color_keys = np.array(list(color2class.keys()))
# color_values = np.array(list(color2class.values()))

# def rgb_to_class(mask):
#     mask = np.array(mask)
#     reshaped = mask.reshape(-1, 3)
#     class_map = np.zeros((reshaped.shape[0],), dtype=np.uint8)
#     for color, class_id in zip(color_keys, color_values):
#         matches = np.all(reshaped == color, axis=1)
#         class_map[matches] = class_id
#     return class_map.reshape(mask.shape[:2])

# # ---------- Sigma Config ----------
# sigma_values = ['0.00', '0.01', '0.05', '0.10', '0.20']
# sigma_labels = ['Clean', 'σ=0.01', 'σ=0.05', 'σ=0.1', 'σ=0.2']
# ious_by_sigma = {s: np.zeros(num_classes, dtype=np.float64) for s in sigma_values}
# union_by_sigma = {s: np.zeros(num_classes, dtype=np.float64) for s in sigma_values}

# # ---------- Start Evaluation ----------
# start_time = time.time()
# all_files = sorted([f for f in os.listdir(pred_noisy_dir) if f.endswith('.png')])

# for fname in all_files:
#     match = re.match(r'(.+?)_sigma([0-9.]+)_rgb\.png', fname)
#     if not match:
#         continue

#     base_name, sigma = match.groups()
#     if sigma not in sigma_values:
#         continue

#     pred_path = os.path.join(pred_noisy_dir, fname)
#     gt_filename = f"{base_name}.png"
#     gt_path = os.path.join(gt_dir, gt_filename)

#     if not os.path.exists(gt_path):
#         print(f"❌ GT missing for: {gt_filename}")
#         continue

#     pred_mask = rgb_to_class(Image.open(pred_path).convert('RGB'))
#     gt_mask = rgb_to_class(Image.open(gt_path).convert('RGB'))

#     for cls in range(num_classes):
#         pred_cls = (pred_mask == cls)
#         gt_cls = (gt_mask == cls)
#         ious_by_sigma[sigma][cls] += np.logical_and(pred_cls, gt_cls).sum()
#         union_by_sigma[sigma][cls] += np.logical_or(pred_cls, gt_cls).sum()

#     print(f"✅ Processed σ={sigma}: {fname}")

# # ---------- Compute IoUs ----------
# final_ious = {}
# for sigma in sigma_values:
#     final_ious[sigma] = ious_by_sigma[sigma] / (union_by_sigma[sigma] + 1e-10)

# # ---------- Plot ----------
# x = np.arange(num_classes)
# bar_width = 0.15
# plt.figure(figsize=(14, 6))

# for i, sigma in enumerate(sigma_values):
#     plt.bar(x + i * bar_width, final_ious[sigma] * 100, width=bar_width, label=sigma_labels[i])

# plt.xticks(x + 2 * bar_width, class_names, rotation=30)
# plt.xlabel("Classes")
# plt.ylabel("IoU (%)")
# plt.title("Per-Class IoU across Noise Levels (RGB corrupted)")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.savefig("iou_rgb_noise_comparison.png")
# plt.show()

# # ---------- Print Table ----------
# print("\n📊 Per-Class IoUs (%):")
# for cls_id, cls_name in enumerate(class_names):
#     print(f"{cls_name:12s} :", end=" ")
#     for sigma in sigma_values:
#         val = final_ious[sigma][cls_id] * 100
#         print(f"{sigma_labels[sigma_values.index(sigma)]:>8s}: {val:6.2f}%", end=" ")
#     print()

# # ---------- Save IoU Report to Text File ----------
# report_path = "iou_rgb.txt"
# with open(report_path, "w") as f:
#     f.write("📊 Per-Class IoUs (%)\n")
#     for cls_id, cls_name in enumerate(class_names):
#         f.write(f"{cls_name:12s} : ")
#         for sigma in sigma_values:
#             val = final_ious[sigma][cls_id] * 100
#             label = sigma_labels[sigma_values.index(sigma)]
#             f.write(f"{label:>8s}: {val:6.2f}% ")
#         f.write("\n")
#     f.write(f"\n⏱️ Evaluation Time: {time.time() - start_time:.2f}s\n")

# print(f"\n✅ IoU report saved to: {report_path}")
# print(f"\n⏱️ Evaluation Time: {time.time() - start_time:.2f}s")

#for modal -x 
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re
import time

# ---------- Paths ----------
gt_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/Label_Colored'
pred_noisy_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions_noisy_modal'

# ---------- Class Info ----------
class_names = [
    "unlabeled", "car", "person", "bike", "curve",
    "car stop", "guardrail", "color cone", "bump"
]
num_classes = len(class_names)

# ---------- Palette ----------
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
color_keys = np.array(list(color2class.keys()))
color_values = np.array(list(color2class.values()))

def rgb_to_class(mask):
    mask = np.array(mask)
    reshaped = mask.reshape(-1, 3)
    class_map = np.zeros((reshaped.shape[0],), dtype=np.uint8)
    for color, class_id in zip(color_keys, color_values):
        matches = np.all(reshaped == color, axis=1)
        class_map[matches] = class_id
    return class_map.reshape(mask.shape[:2])

# ---------- Sigma Config ----------
sigma_values = ['0.00', '0.01', '0.05', '0.10', '0.20']
sigma_labels = ['Clean', 'σ=0.01', 'σ=0.05', 'σ=0.1', 'σ=0.2']
ious_by_sigma = {s: np.zeros(num_classes, dtype=np.float64) for s in sigma_values}
union_by_sigma = {s: np.zeros(num_classes, dtype=np.float64) for s in sigma_values}

# ---------- Start Evaluation ----------
start_time = time.time()
all_files = sorted([f for f in os.listdir(pred_noisy_dir) if f.endswith('.png')])

for fname in all_files:
    match = re.match(r'(.*)_sigma([0-9.]+)_modal\.png', fname)
    if not match:
        continue

    base_name, sigma = match.groups()
    if sigma not in sigma_values:
        continue

    pred_path = os.path.join(pred_noisy_dir, fname)
    gt_filename = f"{base_name}.png"
    gt_path = os.path.join(gt_dir, gt_filename)

    if not os.path.exists(gt_path):
        print(f"❌ GT missing for: {gt_filename}")
        continue

    pred_mask = rgb_to_class(Image.open(pred_path).convert('RGB'))
    gt_mask = rgb_to_class(Image.open(gt_path).convert('RGB'))

    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        ious_by_sigma[sigma][cls] += np.logical_and(pred_cls, gt_cls).sum()
        union_by_sigma[sigma][cls] += np.logical_or(pred_cls, gt_cls).sum()

    print(f"✅ Processed σ={sigma}: {fname}")

# ---------- Compute IoUs ----------
final_ious = {}
for sigma in sigma_values:
    final_ious[sigma] = ious_by_sigma[sigma] / (union_by_sigma[sigma] + 1e-10)

# ---------- Plot ----------
x = np.arange(num_classes)
bar_width = 0.15
plt.figure(figsize=(14, 6))

for i, sigma in enumerate(sigma_values):
    plt.bar(x + i * bar_width, final_ious[sigma] * 100, width=bar_width, label=sigma_labels[i])

plt.xticks(x + 2 * bar_width, class_names, rotation=30)
plt.xlabel("Classes")
plt.ylabel("IoU (%)")
plt.title("Per-Class IoU across Noise Levels (Modal corrupted)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("iou_modal_noise_comparison.png")
plt.show()

# ---------- Print Table ----------
print("\n📊 Per-Class IoUs (%):")
for cls_id, cls_name in enumerate(class_names):
    print(f"{cls_name:12s} :", end=" ")
    for sigma in sigma_values:
        val = final_ious[sigma][cls_id] * 100
        print(f"{sigma_labels[sigma_values.index(sigma)]:>8s}: {val:6.2f}%", end=" ")
    print()

# ---------- Save IoU Report to Text File ----------
report_path = "iou_modal.txt"
with open(report_path, "w") as f:
    f.write("📊 Per-Class IoUs (%)\n")
    for cls_id, cls_name in enumerate(class_names):
        f.write(f"{cls_name:12s} : ")
        for sigma in sigma_values:
            val = final_ious[sigma][cls_id] * 100
            label = sigma_labels[sigma_values.index(sigma)]
            f.write(f"{label:>8s}: {val:6.2f}% ")
        f.write("\n")
    f.write(f"\n⏱️ Evaluation Time: {time.time() - start_time:.2f}s\n")

print(f"\n✅ IoU report saved to: {report_path}")
print(f"\n⏱️ Evaluation Time: {time.time() - start_time:.2f}s")


