# import os
# import re
# import torch
# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# # -------- Configuration --------
# num_classes = 9
# sigma_values = ['0.00', '0.01', '0.05', '0.10', '0.20']

# # Paths
# iou_rgb_path = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/iou_rgb.txt'
# iou_modal_path = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/iou_modal.txt'
# image_list_path = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/train2.txt'
# rgb_pred_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions_noisy_rgb'
# modal_pred_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions_noisy_modal'
# output_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/ensemble_output'
# os.makedirs(output_dir, exist_ok=True)

# # RGB color palette
# palette = [
#     (0, 0, 0), (128, 64, 128), (220, 20, 60), (255, 0, 0),
#     (255, 255, 0), (0, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255)
# ]

# # -------- Helper Functions --------
# def parse_iou_file(iou_path):
#     class_ious = {}
#     class_id = 0
#     with open(iou_path, 'r') as f:
#         for line in f:
#             if line.strip() == "" or "Evaluation Time" in line or "📊" in line:
#                 continue
#             values = re.findall(r'([0-9.]+)%', line)
#             if len(values) == 5:
#                 class_ious[class_id] = list(map(float, values))
#                 class_id += 1
#     return class_ious

# def decide_class_strategy(rgb_ious, modal_ious):
#     strategy = {}
#     for cls in range(num_classes):
#         drop_rgb = rgb_ious[cls][0] - np.mean(rgb_ious[cls][1:])
#         drop_modal = modal_ious[cls][0] - np.mean(modal_ious[cls][1:])
#         strategy[cls] = 'rgb' if drop_rgb < drop_modal else 'modal'
#     return strategy

# def load_rgb_mask(path):
#     return np.array(Image.open(path).convert('RGB'))

# def rgb_to_class(mask):
#     class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
#     for cls_id, color in enumerate(palette):
#         class_mask[np.all(mask == color, axis=-1)] = cls_id
#     return class_mask

# def class_to_rgb(class_mask):
#     rgb = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
#     for cls_id, color in enumerate(palette):
#         rgb[class_mask == cls_id] = color
#     return rgb

# def ensemble_softmax(image_id, class_strategy):
#     rgb_mask_path = os.path.join(rgb_pred_dir, f"{image_id}_sigma0.00_rgb.png")
#     rgb_class_mask = rgb_to_class(load_rgb_mask(rgb_mask_path))

#     modal_logits_all = []
#     for sigma in sigma_values:
#         pt_path = os.path.join(modal_pred_dir, f"{image_id}_sigma{sigma}_modal.pt")
#         logits = torch.load(pt_path)  # shape: (C, H, W)
#         softmax = torch.nn.functional.softmax(logits, dim=0)
#         modal_logits_all.append(softmax)

#     avg_modal = torch.stack(modal_logits_all[1:]).mean(dim=0)  # exclude clean
#     C, H, W = avg_modal.shape
#     final_mask = torch.zeros((H, W), dtype=torch.uint8)

#     for h in range(H):
#         for w in range(W):
#             rgb_cls = rgb_class_mask[h, w]
#             strategy = class_strategy[rgb_cls]
#             if strategy == 'rgb':
#                 final_mask[h, w] = rgb_cls
#             else:  # 'modal'
#                 final_mask[h, w] = torch.argmax(avg_modal[:, h, w]).item()

#     return final_mask

# # -------- Main Pipeline --------
# rgb_ious = parse_iou_file(iou_rgb_path)
# modal_ious = parse_iou_file(iou_modal_path)
# class_strategy = decide_class_strategy(rgb_ious, modal_ious)

# with open(image_list_path, 'r') as f:
#     image_ids = [line.strip() for line in f]

# for image_id in tqdm(image_ids, desc="Ensembling validation predictions"):
#     out_mask = ensemble_softmax(image_id, class_strategy)
#     out_rgb = class_to_rgb(out_mask.numpy())
#     Image.fromarray(out_rgb).save(os.path.join(output_dir, f"{image_id}_ensemble.png"))

# print(f"\n✅ All ensemble predictions saved to: {output_dir}")


import os
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# -------- Paths --------
num_classes = 9
sigma_values = ['0.01', '0.05', '0.10', '0.20']
iou_rgb_path = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/iou_rgb.txt'
iou_modal_path = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/iou_modal.txt'
image_list_path = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/train2.txt'
rgb_pred_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions_noisy_rgb'
modal_pred_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions_noisy_modal'
output_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/ensemble_output'
# modal_pred_dir = '/mnt/data/mfnet_predictions_noisy_modal'
# rgb_pred_dir = '/mnt/data/rgb_preds'
# output_dir = '/mnt/data/ensemble_output'
# image_list_path = '/mnt/data/val_image_ids.txt'
# iou_rgb_path = '/mnt/data/iou_rgb.txt'
# iou_modal_path = '/mnt/data/iou_modal.txt'
os.makedirs(output_dir, exist_ok=True)

# -------- Palette --------
palette = [
    (0, 0, 0), (128, 64, 128), (220, 20, 60), (255, 0, 0),
    (255, 255, 0), (0, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255)
]

# -------- Utilities --------
def rgb_to_class(mask):
    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for i, color in enumerate(palette):
        class_mask[np.all(mask == color, axis=-1)] = i
    return class_mask

def class_to_rgb(class_mask):
    rgb = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(palette):
        rgb[class_mask == cls_id] = color
    return rgb

def parse_iou_file(iou_path):
    class_ious = {}
    class_id = 0
    with open(iou_path, 'r') as f:
        for line in f:
            if line.strip() == "" or "Evaluation Time" in line or "📊" in line:
                continue
            values = re.findall(r'([0-9.]+)%', line)
            if len(values) == 5:
                class_ious[class_id] = list(map(float, values))
                class_id += 1
    return class_ious

def decide_class_strategy(rgb_ious, modal_ious):
    strategy = {}
    for cls in range(num_classes):
        drop_rgb = rgb_ious[cls][0] - np.mean(rgb_ious[cls][1:])
        drop_modal = modal_ious[cls][0] - np.mean(modal_ious[cls][1:])
        strategy[cls] = 'rgb' if drop_rgb < drop_modal else 'modal'
    return strategy

def ensemble_prediction(image_id, class_strategy):
    # Load RGB clean prediction
    rgb_path = os.path.join(rgb_pred_dir, f"{image_id}_sigma0.00_rgb.png")
    rgb_mask = rgb_to_class(np.array(Image.open(rgb_path)))

    # Load modal predictions at noisy sigmas
    modal_preds = []
    for sigma in sigma_values:
        modal_path = os.path.join(modal_pred_dir, f"{image_id}_sigma{sigma}_modal.png")
        modal_class = rgb_to_class(np.array(Image.open(modal_path)))
        modal_preds.append(modal_class)

    H, W = rgb_mask.shape
    onehot_modal = torch.zeros((len(sigma_values), num_classes, H, W))
    for i, modal_cls in enumerate(modal_preds):
        onehot = torch.nn.functional.one_hot(torch.tensor(modal_cls).long(), num_classes).permute(2, 0, 1)
        onehot_modal[i] = onehot

    modal_avg_probs = onehot_modal.mean(dim=0)  # shape: (C, H, W)

    final_mask = torch.zeros((H, W), dtype=torch.uint8)
    for h in range(H):
        for w in range(W):
            cls = rgb_mask[h, w]
            if class_strategy[cls] == 'rgb':
                final_mask[h, w] = cls
            else:
                final_mask[h, w] = torch.argmax(modal_avg_probs[:, h, w])

    return final_mask

# -------- Run --------
rgb_ious = parse_iou_file(iou_rgb_path)
modal_ious = parse_iou_file(iou_modal_path)
class_strategy = decide_class_strategy(rgb_ious, modal_ious)

with open(image_list_path, 'r') as f:
    image_ids = [line.strip() for line in f]

for image_id in tqdm(image_ids, desc="Adaptive ensemble prediction"):
    out_mask = ensemble_prediction(image_id, class_strategy)
    out_rgb = class_to_rgb(out_mask.numpy())
    Image.fromarray(out_rgb).save(os.path.join(output_dir, f"{image_id}_ensemble.png"))

print(f"\n✅ Ensemble results saved in: {output_dir}")
