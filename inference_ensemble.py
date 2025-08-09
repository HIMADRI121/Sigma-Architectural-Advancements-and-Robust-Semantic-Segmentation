# # # import os
# # # import torch

# # # # --- SETTINGS ---
# # # rgb_logits_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions_noisy_rgb'
# # # modal_logits_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/mfnet_predictions_noisy_modal'
# # # output_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/ensemble_class_tensors'
# # # os.makedirs(output_dir, exist_ok=True)

# # # sigma_list = ['0.01', '0.05', '0.10', '0.20']
# # # num_classes = 9

# # # # --- Class-wise strategy (from your output) ---
# # # class_strategy = {
# # #     0: 'rgb',
# # #     1: 'rgb',
# # #     2: 'rgb',
# # #     3: 'modal',
# # #     4: 'rgb',
# # #     5: 'modal',
# # #     6: 'modal',
# # #     7: 'rgb',
# # #     8: 'modal',
# # # }

# # # # --- Helper: load logits from .pt ---
# # # def load_logits(image_id, sigma, modality):
# # #     fname = f"{image_id}_sigma{sigma}_{modality}.pt"
# # #     fpath = os.path.join(modal_logits_dir if modality == 'modal' else rgb_logits_dir, fname)
# # #     return torch.load(fpath)  # shape: (C, H, W)

# # # # --- Ensemble Function ---
# # # def ensemble_image(image_id):
# # #     # Load RGB clean logits
# # #     rgb_logits = load_logits(image_id, '0.00', 'rgb')  # shape: [C, H, W]

# # #     # Load modal logits for each sigma
# # #     modal_logits_list = [load_logits(image_id, s, 'modal') for s in sigma_list]
# # #     modal_logits_avg = torch.stack(modal_logits_list).mean(dim=0)  # avg over sigmas

# # #     # Final output tensor
# # #     C, H, W = rgb_logits.shape
# # #     final_logits = torch.zeros((C, H, W))

# # #     for cls in range(num_classes):
# # #         if class_strategy[cls] == 'rgb':
# # #             final_logits[cls] = rgb_logits[cls]
# # #         else:
# # #             final_logits[cls] = modal_logits_avg[cls]

# # #     # Argmax to get predicted class map
# # #     pred_class_map = final_logits.argmax(dim=0).to(torch.uint8)  # shape: [H, W]

# # #     # Save
# # #     torch.save(pred_class_map, os.path.join(output_dir, f"{image_id}_ensemble.pt"))
# # #     print(f"✅ Saved: {image_id}_ensemble.pt")

# # # # --- Process all image IDs (change if needed) ---
# # # image_ids = ['00001D', '00002D', '00003D']  # Fill with all validation image IDs

# # # for img_id in image_ids:
# # #     ensemble_image(img_id)


# # import os
# # import torch

# # # --- SETTINGS ---
# # rgb_logits_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_rgb'
# # modal_logits_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_modal'
# # output_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/ensemble_class_tensors'
# # os.makedirs(output_dir, exist_ok=True)

# # sigma_list = ['0.01', '0.05', '0.10', '0.20']
# # num_classes = 9

# # # --- Class-wise strategy (from your output) ---
# # class_strategy = {
# #     0: 'rgb',
# #     1: 'rgb',
# #     2: 'rgb',
# #     3: 'modal',
# #     4: 'rgb',
# #     5: 'modal',
# #     6: 'modal',
# #     7: 'rgb',
# #     8: 'modal',
# # }

# # # --- Helper: load logits from .pt ---
# # def load_logits(image_id, sigma, modality):
# #     fname = f"{image_id}_sigma{sigma}_{modality}.pt"
# #     dir_path = modal_logits_dir if modality == 'modal' else rgb_logits_dir
# #     fpath = os.path.join(dir_path, fname)
# #     if not os.path.exists(fpath):
# #         raise FileNotFoundError(f"Missing file: {fpath}")
# #     return torch.load(fpath)  # shape: [C, H, W]

# # # --- Ensemble Function ---
# # def ensemble_image(image_id):
# #     # Load clean logits for both modalities
# #     rgb_clean = load_logits(image_id, '0.00', 'rgb')     # shape: [C, H, W]
# #     modal_clean = load_logits(image_id, '0.00', 'modal') # shape: [C, H, W]

# #     # Load corrupted logits
# #     rgb_corrupt_list = [load_logits(image_id, s, 'rgb') for s in sigma_list]
# #     modal_corrupt_list = [load_logits(image_id, s, 'modal') for s in sigma_list]

# #     # Average corrupted logits
# #     rgb_corrupt_avg = torch.stack(rgb_corrupt_list).mean(dim=0)
# #     modal_corrupt_avg = torch.stack(modal_corrupt_list).mean(dim=0)

# #     # Final tensor
# #     C, H, W = rgb_clean.shape
# #     final_logits = torch.zeros((C, H, W))

# #     # Apply class-specific strategy
# #     for cls in range(num_classes):
# #         if class_strategy[cls] == 'rgb':
# #             # Use RGB clean
# #             final_logits[cls] = rgb_clean[cls]
# #         else:  # 'modal'
# #             # Use averaged corrupted modal
# #             final_logits[cls] = modal_corrupt_avg[cls]

# #     # Final prediction
# #     pred_class_map = final_logits.argmax(dim=0).to(torch.uint8)  # shape: [H, W]

# #     # Save tensor
# #     save_path = os.path.join(output_dir, f"{image_id}_ensemble.pt")
# #     torch.save(pred_class_map, save_path)ad
# #     print(f"✅ Saved: {image_id}_ensemble.pt")

# # # --- Run on image list ---
# # image_ids = ['00001D', '00002D', '00003D']  # Update with full validation set

# # for img_id in image_ids:
# #     ensemble_image(img_id)

# import os
# import torch

# # --- SETTINGS ---
# rgb_logits_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_rgb'
# modal_logits_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_modal'
# output_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/ensemble_class_tensors'
# os.makedirs(output_dir, exist_ok=True)

# sigma_list = ['0.01', '0.05', '0.10', '0.20']
# num_classes = 9

# # --- Class-wise strategy (based on IoU drop across sigma) ---
# class_strategy = {
#     0: 'rgb',
#     1: 'rgb',
#     2: 'rgb',
#     3: 'modal',
#     4: 'rgb',
#     5: 'modal',
#     6: 'modal',
#     7: 'rgb',
#     8: 'modal',
# }

# # --- Helper: load logits from .pt ---
# def load_logits(image_id, sigma, modality):
#     fname = f"{image_id}_sigma{sigma}_{modality}_onehot.pt"
#     dir_path = modal_logits_dir if modality == 'modal' else rgb_logits_dir
#     fpath = os.path.join(dir_path, fname)
#     if not os.path.exists(fpath):
#         raise FileNotFoundError(f"Missing file: {fpath}")
#     return torch.load(fpath)  # shape: [C, H, W]




# # --- Ensemble Function ---
# def ensemble_image(image_id):
#     # Load clean logits for both modalities
#     rgb_clean = load_logits(image_id, '0.00', 'rgb')     # shape: [C, H, W]
#     modal_clean = load_logits(image_id, '0.00', 'modal') # shape: [C, H, W]

#     # Load corrupted logits
#     rgb_corrupt_list = [load_logits(image_id, s, 'rgb') for s in sigma_list]
#     modal_corrupt_list = [load_logits(image_id, s, 'modal') for s in sigma_list]

#     # Average corrupted logits
#     rgb_corrupt_avg = torch.stack([t.float() for t in rgb_corrupt_list]).mean(dim=0)
#     modal_corrupt_avg = torch.stack([t.float() for t in modal_corrupt_list]).mean(dim=0)


#     # Final tensor
#     C, H, W = rgb_clean.shape
#     final_logits = torch.zeros((C, H, W))

#     # Apply class-specific strategy
#     for cls in range(num_classes):
#         if class_strategy[cls] == 'rgb':
#             final_logits[cls] = rgb_clean[cls]
#         else:  # 'modal'
#             final_logits[cls] = modal_corrupt_avg[cls]

#     # Final prediction
#     pred_class_map = final_logits.argmax(dim=0).to(torch.uint8)  # shape: [H, W]

#     # Save tensor
#     save_path = os.path.join(output_dir, f"{image_id}_ensemble.pt")
#     torch.save(pred_class_map, save_path)
#     print(f"✅ Saved: {image_id}_ensemble.pt")

# # --- Run on image list ---
# image_ids = ['00001D', '00002D', '00003D']  # Replace with full list or read from txt

# for img_id in image_ids:
#     ensemble_image(img_id)

import os
import torch

# --- SETTINGS ---
rgb_logits_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_rgb'
modal_logits_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_modal'
output_dir = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/ensemble_class_tensors'
os.makedirs(output_dir, exist_ok=True)

sigma_list = ['0.01', '0.05', '0.10', '0.20']
num_classes = 9

# --- Class-wise strategy ---
class_strategy = {
    0: 'rgb', 1: 'rgb', 2: 'rgb', 3: 'modal',
    4: 'rgb', 5: 'modal', 6: 'modal', 7: 'rgb', 8: 'modal',
}

# --- Helper function ---
def load_logits(image_id, sigma, modality):
    fname = f"{image_id}_sigma{sigma}_{modality}_onehot.pt"
    dir_path = rgb_logits_dir if modality == 'rgb' else modal_logits_dir
    fpath = os.path.join(dir_path, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Missing file: {fpath}")
    return torch.load(fpath).float()  # Ensure float dtype for mean()

# --- Ensemble Function ---
def ensemble_image(image_id):
    try:
        rgb_clean = load_logits(image_id, '0.00', 'rgb')       # [C, H, W]
        modal_clean = load_logits(image_id, '0.00', 'modal')   # [C, H, W]

        rgb_corrupt_list = [load_logits(image_id, s, 'rgb') for s in sigma_list]
        modal_corrupt_list = [load_logits(image_id, s, 'modal') for s in sigma_list]

        rgb_corrupt_avg = torch.stack(rgb_corrupt_list).mean(dim=0)
        modal_corrupt_avg = torch.stack(modal_corrupt_list).mean(dim=0)

        C, H, W = rgb_clean.shape
        final_logits = torch.zeros((C, H, W), dtype=torch.float32)

        for cls in range(num_classes):
            if class_strategy[cls] == 'rgb':
                final_logits[cls] = rgb_clean[cls]
            else:
                final_logits[cls] = modal_corrupt_avg[cls]

        pred_class_map = final_logits.argmax(dim=0).to(torch.uint8)
        torch.save(pred_class_map, os.path.join(output_dir, f"{image_id}_ensemble.pt"))
        print(f"✅ Saved: {image_id}_ensemble.pt")

    except FileNotFoundError as e:
        print(f"⚠️ Skipping {image_id}: {e}")

# --- Auto-detect available image IDs ---
def detect_valid_image_ids(pt_dir, suffix='_rgb_class.pt'):
    return sorted(set([
        fname.split('_')[0] for fname in os.listdir(pt_dir)
        if fname.endswith(suffix)
    ]))

# --- Main ---
image_ids = detect_valid_image_ids(rgb_logits_dir)

for img_id in image_ids:
    ensemble_image(img_id)
