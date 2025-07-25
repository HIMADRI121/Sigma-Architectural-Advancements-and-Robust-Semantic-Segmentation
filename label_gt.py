import os
import sys
import torch
from dataloader.RGBXDataset import RGBXDataset

# Add root directory to sys.path (only needed if imports fail)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dataset settings
setting = {
    'rgb_root': '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/RGB/',
    'rgb_format': '.png',
    
    # Change this from GT to Label
    'gt_root': '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/Label/',
    'gt_format': '.png',
    
    'transform_gt': True,
    'x_root': '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/Modal/',
    'x_format': '.png',
    'x_single_channel': True,
    'train_source': '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/train.txt',
    'eval_source': '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/train.txt',
    'class_names': ['background', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']
}

# setting = {
#     'rgb_root': '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/RGB/',
#     'rgb_format': '.png',
#     'gt_root': '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/GT/',
#     'gt_format': '.png',
#     'transform_gt': True,
#     'x_root': '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/Modal/',
#     'x_format': '.png',
#     'x_single_channel': True,
#     'train_source': '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/train.txt',
#     'eval_source': '/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/test.txt',
#     'class_names': ['background', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']
# }

split_name = 'val'  # or 'train'
dataset = RGBXDataset(setting, split_name=split_name)
output_txt = f"1gt_labels_{split_name}.txt"

with open(output_txt, 'w') as f:
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            fn = sample['fn']
            label = sample['label']

            if label is None:
                print(f"[WARN] Label is None at index {i}")
                continue

            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(label)

            unique_labels = torch.unique(label)
            label_str = ','.join(str(l.item()) for l in unique_labels)
            f.write(f"{fn}: {label_str}\n")

        except Exception as e:
            print(f"[ERROR] Could not process index {i}: {e}")
'''
with open(output_txt, 'w') as f:
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            fn = sample['fn']
            label = sample['label']

            if label is None:
                print(f"[WARN] Skipped {fn} due to missing GT.")
                continue

            unique_labels = torch.unique(label)
            label_str = ','.join(str(l.item()) for l in unique_labels)
            f.write(f"{fn}: {label_str}\n")

        except Exception as e:
            print(f"[ERROR] Could not process index {i}: {e}")
            continue

print(f"[✔] Ground truth labels saved to: {os.path.abspath(output_txt)}")
'''