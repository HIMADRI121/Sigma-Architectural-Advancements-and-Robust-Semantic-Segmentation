'''
import os
import torch
import torch.nn.functional as F
from PIL import Image

# --- Config ---
NUM_CLASSES = 9
SIGMAS = ['0.01', '0.05', '0.10', '0.20']
ROOT_RGB = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_rgb'
ROOT_MODAL = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_modal'
ROOT_ENSEMBLE = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/ensemble_class_tensors'
OUT_DIR = 'results/final_predictions'
os.makedirs(OUT_DIR, exist_ok=True)

def to_onehot(pred, num_classes=NUM_CLASSES):
    return F.one_hot(pred.long(), num_classes).permute(2, 0, 1).float()  # [C, H, W]

def load_preds(base_id, folder, prefix):
    # Load clean
    clean_class = torch.load(f"{folder}/{base_id}_sigma0.00_{prefix}_class.pt")  # [H, W]
    clean_onehot = to_onehot(clean_class)  # [C, H, W]

    # Load noisy predictions
    noisy_onehots = []
    for sigma in SIGMAS:
        path = f"{folder}/{base_id}_sigma{sigma}_{prefix}_class.pt"
        noisy_class = torch.load(path)
        noisy_onehots.append(to_onehot(noisy_class))
    avg_noisy_onehot = sum(noisy_onehots) / len(noisy_onehots)  # [C, H, W]
    avg_noisy_class = torch.argmax(avg_noisy_onehot, dim=0)  # [H, W]

    return clean_class, avg_noisy_class, clean_onehot, avg_noisy_onehot

def run_confidence_based_pixel_selection(base_id):
    # Load predictions
    rgb_clean, rgb_noisy_avg, rgb_clean_oh, rgb_noisy_oh = load_preds(base_id, ROOT_RGB, 'rgb')
    modal_clean, modal_noisy_avg, modal_clean_oh, modal_noisy_oh = load_preds(base_id, ROOT_MODAL, 'modal')

    # Compute softmax confidence at each pixel for both
    rgb_logits = torch.where(rgb_clean_oh > 0, rgb_clean_oh, rgb_noisy_oh)
    modal_logits = torch.where(modal_clean_oh > 0, modal_clean_oh, modal_noisy_oh)

    rgb_conf = F.softmax(rgb_logits, dim=0)  # [C, H, W]
    modal_conf = F.softmax(modal_logits, dim=0)

    # Confidence per pixel
    rgb_max_conf = torch.max(rgb_conf, dim=0).values  # [H, W]
    modal_max_conf = torch.max(modal_conf, dim=0).values  # [H, W]

    # Per-pixel decision
    choose_rgb = rgb_max_conf > modal_max_conf  # [H, W]

    # Per-pixel prediction
    final_prediction = torch.where(choose_rgb, rgb_clean, modal_noisy_avg)  # [H, W]

    # Save final prediction
    out_path = os.path.join(OUT_DIR, f"{base_id}_confidence_winner.png")
    Image.fromarray(final_prediction.byte().cpu().numpy()).save(out_path)

    # ---- Reporting ----
    num_total_pixels = choose_rgb.numel()
    num_rgb_pixels = choose_rgb.sum().item()
    num_modal_pixels = num_total_pixels - num_rgb_pixels

    avg_rgb_conf = rgb_max_conf[choose_rgb].mean().item() if num_rgb_pixels > 0 else 0.0
    avg_modal_conf = modal_max_conf[~choose_rgb].mean().item() if num_modal_pixels > 0 else 0.0

    winner = "RGB" if avg_rgb_conf > avg_modal_conf else "Modal"
    winning_score = max(avg_rgb_conf, avg_modal_conf)

    # Confidence vector (mean logits)
    final_logits = torch.where(choose_rgb.unsqueeze(0), rgb_logits, modal_logits)
    final_logits_mean = final_logits.mean(dim=(1, 2), keepdim=True).T  # [1, 9]
    final_conf = F.softmax(final_logits_mean, dim=1)

    print(f"\n🧾 {base_id}")
    print(f"Logits Shape: {final_logits_mean.shape}")
    print(f"Confidence Scores:\n{final_conf}")
    print(f"🟦 RGB Pixels:   {num_rgb_pixels} | Avg Conf: {avg_rgb_conf:.4f}")
    print(f"🟧 Modal Pixels: {num_modal_pixels} | Avg Conf: {avg_modal_conf:.4f}")
    print(f"🏆 Winning Modality: {winner} with Score: {winning_score:.4f}")

# ---- Get all base IDs from ensemble_tensor filenames ----
all_ensemble_files = os.listdir(ROOT_ENSEMBLE)
base_ids = sorted([f.split('_ensemble')[0] for f in all_ensemble_files if f.endswith('.pt')])

# ---- Run batch inference ----
for base_id in base_ids:
    run_confidence_based_pixel_selection(base_id)
'''
# import os
# import torch
# import torch.nn.functional as F
# from PIL import Image

# # --- Config ---
# NUM_CLASSES = 9
# SIGMAS = ['0.01', '0.05', '0.10', '0.20']
# ROOT_RGB = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_rgb'
# ROOT_MODAL = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_modal'
# ROOT_ENSEMBLE = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/ensemble_class_tensorsr'
# OUT_DIR = 'results/final_predictions'
# os.makedirs(OUT_DIR, exist_ok=True)

# def to_onehot(pred, num_classes=NUM_CLASSES):
#     return F.one_hot(pred.long(), num_classes).permute(2, 0, 1).float()

# def load_preds(base_id, folder, prefix):
#     clean_class = torch.load(f"{folder}/{base_id}_sigma0.00_{prefix}_class.pt")
#     clean_onehot = to_onehot(clean_class)

#     noisy_onehots = []
#     for sigma in SIGMAS:
#         path = f"{folder}/{base_id}_sigma{sigma}_{prefix}_class.pt"
#         noisy_class = torch.load(path)
#         noisy_onehots.append(to_onehot(noisy_class))
#     avg_noisy_onehot = sum(noisy_onehots) / len(noisy_onehots)
#     avg_noisy_class = torch.argmax(avg_noisy_onehot, dim=0)

#     return clean_class, avg_noisy_class, clean_onehot, avg_noisy_onehot

# def run_confidence_based_pixel_selection(base_id):
#     rgb_clean, rgb_noisy_avg, rgb_clean_oh, rgb_noisy_oh = load_preds(base_id, ROOT_RGB, 'rgb')
#     modal_clean, modal_noisy_avg, modal_clean_oh, modal_noisy_oh = load_preds(base_id, ROOT_MODAL, 'modal')

#     rgb_logits = torch.where(rgb_clean_oh > 0, rgb_clean_oh, rgb_noisy_oh)
#     modal_logits = torch.where(modal_clean_oh > 0, modal_clean_oh, modal_noisy_oh)

#     rgb_conf = F.softmax(rgb_logits, dim=0)
#     modal_conf = F.softmax(modal_logits, dim=0)

#     rgb_max_conf = torch.max(rgb_conf, dim=0).values
#     modal_max_conf = torch.max(modal_conf, dim=0).values

#     choose_rgb = rgb_max_conf > modal_max_conf
#     final_prediction = torch.where(choose_rgb, rgb_clean, modal_noisy_avg)

#     out_path = os.path.join(OUT_DIR, f"{base_id}_confidence_winner.png")
#     Image.fromarray(final_prediction.byte().cpu().numpy()).save(out_path)

#     # Modal win map
#     choose_modal = ~choose_rgb

#     # Class-wise pixel count maps
#     rgb_class_mask = rgb_clean  # [H, W]
#     modal_class_mask = modal_noisy_avg  # [H, W]

#     # Class-wise counters
#     rgb_class_counts = torch.zeros(NUM_CLASSES, dtype=torch.int32)
#     modal_class_counts = torch.zeros(NUM_CLASSES, dtype=torch.int32)
#     final_class_counts = torch.zeros(NUM_CLASSES, dtype=torch.int32)

#     for cls in range(NUM_CLASSES):
#         rgb_class_counts[cls] = ((rgb_class_mask == cls) & choose_rgb).sum()
#         modal_class_counts[cls] = ((modal_class_mask == cls) & choose_modal).sum()
#         final_class_counts[cls] = (final_prediction == cls).sum()

#     # Print stats
#     num_rgb_pixels = choose_rgb.sum().item()
#     num_modal_pixels = choose_modal.sum().item()
#     avg_rgb_conf = rgb_max_conf[choose_rgb].mean().item() if num_rgb_pixels > 0 else 0.0
#     avg_modal_conf = modal_max_conf[choose_modal].mean().item() if num_modal_pixels > 0 else 0.0

#     winner = "RGB" if avg_rgb_conf > avg_modal_conf else "Modal"
#     winning_score = max(avg_rgb_conf, avg_modal_conf)

#     # Final confidence vector
#     final_logits = torch.where(choose_rgb.unsqueeze(0), rgb_logits, modal_logits)
#     final_logits_mean = final_logits.mean(dim=(1, 2), keepdim=True).T  # [1, 9]
#     final_conf = F.softmax(final_logits_mean, dim=1)

#     print(f"\n🧾 {base_id}")
#     print(f"Logits Shape: {final_logits_mean.shape}")
#     print(f"Confidence Scores:\n{final_conf}")
#     print(f"🟦 RGB Pixels:   {num_rgb_pixels} | Avg Conf: {avg_rgb_conf:.4f}")
#     print(f"🟧 Modal Pixels: {num_modal_pixels} | Avg Conf: {avg_modal_conf:.4f}")
#     print(f"🏆 Winning Modality: {winner} with Score: {winning_score:.4f}")
#     print("\n📊 Class-wise Modality Dominance:")
#     print(f"{'Class':>6} | {'RGB Count':>10} | {'Modal Count':>12} | {'Winner':>8} | {'Total':>6}")
#     print("-" * 50)
#     for cls in range(NUM_CLASSES):
#         rgb_c = rgb_class_counts[cls].item()
#         modal_c = modal_class_counts[cls].item()
#         total_c = final_class_counts[cls].item()
#         winner_cls = "RGB" if rgb_c > modal_c else ("Modal" if modal_c > rgb_c else "Tie")
#         print(f"{cls:6} | {rgb_c:10} | {modal_c:12} | {winner_cls:8} | {total_c:6}")

# # ---- Get all base IDs from ensemble_tensor filenames ----
# all_ensemble_files = os.listdir(ROOT_ENSEMBLE)
# base_ids = sorted([f.split('_ensemble')[0] for f in all_ensemble_files if f.endswith('.pt')])

# # ---- Run batch inference ----
# for base_id in base_ids:
#     run_confidence_based_pixel_selection(base_id)



'''--working script--
import os
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# --- Config ---
NUM_CLASSES = 9
SIGMAS = ['0.01', '0.05', '0.10', '0.20']
ROOT_RGB = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_rgb'
ROOT_MODAL = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_modal'
ROOT_ENSEMBLE = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/ensemble_class_tensors'
OUT_DIR = 'results/final_predictions'
os.makedirs(OUT_DIR, exist_ok=True)

def to_onehot(pred, num_classes=NUM_CLASSES):
    return F.one_hot(pred.long(), num_classes).permute(2, 0, 1).float()  # [C, H, W]

def load_preds(base_id, folder, prefix):
    # Clean
    clean_class = torch.load(f"{folder}/{base_id}_sigma0.00_{prefix}_class.pt")
    clean_onehot = to_onehot(clean_class)

    # Noisy
    noisy_onehots = []
    for sigma in SIGMAS:
        path = f"{folder}/{base_id}_sigma{sigma}_{prefix}_class.pt"
        noisy_class = torch.load(path)
        noisy_onehots.append(to_onehot(noisy_class))
    avg_noisy_onehot = sum(noisy_onehots) / len(noisy_onehots)
    avg_noisy_class = torch.argmax(avg_noisy_onehot, dim=0)

    return clean_class, avg_noisy_class, clean_onehot, avg_noisy_onehot

def save_debug_image(tensor, base_id):
    vis_tensor = (tensor * (255 // NUM_CLASSES)).byte()
    debug_path = os.path.join(OUT_DIR, f"{base_id}_vis_debug.png")
    Image.fromarray(vis_tensor.cpu().numpy()).save(debug_path)

    # Also with matplotlib color map
    cmap_path = os.path.join(OUT_DIR, f"{base_id}_colormap.png")
    plt.imsave(cmap_path, tensor.cpu().numpy(), cmap='tab20')

def run_confidence_based_pixel_selection(base_id):
    # Load predictions
    rgb_clean, rgb_noisy_avg, rgb_clean_oh, rgb_noisy_oh = load_preds(base_id, ROOT_RGB, 'rgb')
    modal_clean, modal_noisy_avg, modal_clean_oh, modal_noisy_oh = load_preds(base_id, ROOT_MODAL, 'modal')

    # Confidence logits
    rgb_logits = torch.where(rgb_clean_oh > 0, rgb_clean_oh, rgb_noisy_oh)
    modal_logits = torch.where(modal_clean_oh > 0, modal_clean_oh, modal_noisy_oh)

    rgb_conf = F.softmax(rgb_logits, dim=0)  # [C, H, W]
    modal_conf = F.softmax(modal_logits, dim=0)

    rgb_max_conf = torch.max(rgb_conf, dim=0).values
    modal_max_conf = torch.max(modal_conf, dim=0).values

    choose_rgb = rgb_max_conf > modal_max_conf
    choose_modal = ~choose_rgb

    final_prediction = torch.where(choose_rgb, rgb_clean, modal_noisy_avg)

    # ---- Debug: Print unique values ----
    print(f"\n🧾 {base_id}")
    print("Unique values in final_prediction:", final_prediction.unique())
    print(f"Final Prediction Shape: {final_prediction.shape}")

    #  ---- Save PNG image (proper format) ----
    # vis_path = os.path.join(OUT_DIR, f"{base_id}_confidence_winner.png")
    # plt.imsave(vis_path.replace('.png', '_colormap.png'),
    #        final_prediction.cpu().numpy(),
    #        cmap='tab20')

    # Image.fromarray(final_prediction.byte().cpu().numpy()).save(vis_path)

    # ---- Also save visual debug image ----
    save_debug_image(final_prediction, base_id)

    # ---- Compute Stats ----
    num_rgb_pixels = choose_rgb.sum().item()
    num_modal_pixels = choose_modal.sum().item()
    avg_rgb_conf = rgb_max_conf[choose_rgb].mean().item() if num_rgb_pixels > 0 else 0.0
    avg_modal_conf = modal_max_conf[choose_modal].mean().item() if num_modal_pixels > 0 else 0.0

    winner = "RGB" if avg_rgb_conf > avg_modal_conf else "Modal"
    winning_score = max(avg_rgb_conf, avg_modal_conf)

    final_logits = torch.where(choose_rgb.unsqueeze(0), rgb_logits, modal_logits)
    final_logits_mean = final_logits.mean(dim=(1, 2), keepdim=True).T  # [1, 9]
    final_conf = F.softmax(final_logits_mean, dim=1)

    print(f"Logits Shape: {final_logits_mean.shape}")
    print(f"Confidence Scores:\n{final_conf}")
    print(f"🟦 RGB Pixels:   {num_rgb_pixels} | Avg Conf: {avg_rgb_conf:.4f}")
    print(f"🟧 Modal Pixels: {num_modal_pixels} | Avg Conf: {avg_modal_conf:.4f}")
    print(f"🏆 Winning Modality: {winner} with Score: {winning_score:.4f}")
    import matplotlib.pyplot as plt

    conf_map = torch.max(rgb_conf, dim=0).values.flatten().cpu().numpy()
    plt.hist(conf_map, bins=50, range=(0,1))
    plt.title("Confidence Distribution (RGB)")
    plt.show()
    print("RGB Logits Stats:")
    print(" - Max:", rgb_logits.max().item())
    print(" - Min:", rgb_logits.min().item())
    print(" - Mean:", rgb_logits.mean().item())
    print(" - Std Dev:", rgb_logits.std().item())

    print("Modal Logits Stats:")
    print(" - Max:", modal_logits.max().item())
    print(" - Min:", modal_logits.min().item())
    print(" - Mean:", modal_logits.mean().item())
    print(" - Std Dev:", modal_logits.std().item())

    # ---- Per-Class Dominance ----
    rgb_class_counts = torch.zeros(NUM_CLASSES, dtype=torch.int32)
    modal_class_counts = torch.zeros(NUM_CLASSES, dtype=torch.int32)
    final_class_counts = torch.zeros(NUM_CLASSES, dtype=torch.int32)

    for cls in range(NUM_CLASSES):
        rgb_class_counts[cls] = ((rgb_clean == cls) & choose_rgb).sum()
        modal_class_counts[cls] = ((modal_noisy_avg == cls) & choose_modal).sum()
        final_class_counts[cls] = (final_prediction == cls).sum()

    print("\n📊 Class-wise Modality Dominance:")
    print(f"{'Class':>6} | {'RGB Count':>10} | {'Modal Count':>12} | {'Winner':>8} | {'Total':>6}")
    print("-" * 50)
    for cls in range(NUM_CLASSES):
        rgb_c = rgb_class_counts[cls].item()
        modal_c = modal_class_counts[cls].item()
        total_c = final_class_counts[cls].item()
        winner_cls = "RGB" if rgb_c > modal_c else ("Modal" if modal_c > rgb_c else "Tie")
        print(f"{cls:6} | {rgb_c:10} | {modal_c:12} | {winner_cls:8} | {total_c:6}")

# ---- Run for All Available Base IDs ----
all_ensemble_files = os.listdir(ROOT_ENSEMBLE)
base_ids = sorted([f.split('_ensemble')[0] for f in all_ensemble_files if f.endswith('.pt')])

for base_id in base_ids:
    run_confidence_based_pixel_selection(base_id)
'''
import os
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# --- Config ---
NUM_CLASSES = 9
SIGMAS = ['0.01', '0.05', '0.10', '0.20']
ROOT_RGB = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_rgb'
ROOT_MODAL = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/pt_files_modal'
ROOT_ENSEMBLE = '/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/ensemble_class_tensors'
OUT_DIR = 'results/final_predictions'
os.makedirs(OUT_DIR, exist_ok=True)

def to_onehot(pred, num_classes=NUM_CLASSES):
    return F.one_hot(pred.long(), num_classes).permute(2, 0, 1).float()  # [C, H, W]

def load_preds(base_id, folder, prefix):
    clean_class = torch.load(f"{folder}/{base_id}_sigma0.00_{prefix}_class.pt")
    clean_onehot = to_onehot(clean_class)

    noisy_onehots = []
    for sigma in SIGMAS:
        path = f"{folder}/{base_id}_sigma{sigma}_{prefix}_class.pt"
        noisy_class = torch.load(path)
        noisy_onehots.append(to_onehot(noisy_class))
    avg_noisy_onehot = sum(noisy_onehots) / len(noisy_onehots)
    avg_noisy_class = torch.argmax(avg_noisy_onehot, dim=0)

    return clean_class, avg_noisy_class, clean_onehot, avg_noisy_onehot

def save_debug_image(tensor, base_id):
    vis_tensor = (tensor * (255 // NUM_CLASSES)).byte()
    debug_path = os.path.join(OUT_DIR, f"{base_id}_vis_debug.png")
    Image.fromarray(vis_tensor.cpu().numpy()).save(debug_path)

    cmap_path = os.path.join(OUT_DIR, f"{base_id}_colormap.png")
    plt.imsave(cmap_path, tensor.cpu().numpy(), cmap='tab20')

def print_logits_stats(logits, label):
    max_val = logits.max().item()
    min_val = logits.min().item()
    mean_val = logits.mean().item()
    std_val = logits.std().item()
    print(f"\n📈 {label} Logits Stats:")
    print(f" - Max:  {max_val:.4f}")
    print(f" - Min:  {min_val:.4f}")
    print(f" - Mean: {mean_val:.4f}")
    print(f" - Std:  {std_val:.4f}")
    if std_val < 1.0:
        print(" ⚠️  WARNING: Very low standard deviation — logits may be flat.")

def print_softmax_stats(conf, label):
    max_conf = torch.max(conf, dim=0).values
    mean_conf = max_conf.mean().item()
    std_conf = max_conf.std().item()
    print(f"\n🧪 {label} Confidence Stats (Softmax Max):")
    print(f" - Mean: {mean_conf:.4f} | Std: {std_conf:.4f}")
    if mean_conf < 0.4:
        print(" ⚠️  Low mean confidence — model may be guessing.")

def run_confidence_based_pixel_selection(base_id):
    rgb_clean, rgb_noisy_avg, rgb_clean_oh, rgb_noisy_oh = load_preds(base_id, ROOT_RGB, 'rgb')
    modal_clean, modal_noisy_avg, modal_clean_oh, modal_noisy_oh = load_preds(base_id, ROOT_MODAL, 'modal')

    rgb_logits = torch.where(rgb_clean_oh > 0, rgb_clean_oh, rgb_noisy_oh)
    modal_logits = torch.where(modal_clean_oh > 0, modal_clean_oh, modal_noisy_oh)

    rgb_conf = F.softmax(rgb_logits, dim=0)
    modal_conf = F.softmax(modal_logits, dim=0)

    rgb_max_conf = torch.max(rgb_conf, dim=0).values
    modal_max_conf = torch.max(modal_conf, dim=0).values

    choose_rgb = rgb_max_conf > modal_max_conf
    choose_modal = ~choose_rgb

    final_prediction = torch.where(choose_rgb, rgb_clean, modal_noisy_avg)
    save_debug_image(final_prediction, base_id)

    num_rgb_pixels = choose_rgb.sum().item()
    num_modal_pixels = choose_modal.sum().item()
    avg_rgb_conf = rgb_max_conf[choose_rgb].mean().item() if num_rgb_pixels > 0 else 0.0
    avg_modal_conf = modal_max_conf[choose_modal].mean().item() if num_modal_pixels > 0 else 0.0

    winner = "RGB" if avg_rgb_conf > avg_modal_conf else "Modal"
    winning_score = max(avg_rgb_conf, avg_modal_conf)

    final_logits = torch.where(choose_rgb.unsqueeze(0), rgb_logits, modal_logits)
    final_logits_mean = final_logits.mean(dim=(1, 2), keepdim=True).T
    final_conf = F.softmax(final_logits_mean, dim=1)

    print(f"\n🧾 {base_id}")
    print("Unique values in final_prediction:", final_prediction.unique())
    print(f"Final Prediction Shape: {final_prediction.shape}")
    print(f"Logits Shape: {final_logits_mean.shape}")
    print(f"Confidence Scores:\n{final_conf}")
    print(f"🟦 RGB Pixels:   {num_rgb_pixels} | Avg Conf: {avg_rgb_conf:.4f}")
    print(f"🟧 Modal Pixels: {num_modal_pixels} | Avg Conf: {avg_modal_conf:.4f}")
    print(f"🏆 Winning Modality: {winner} with Score: {winning_score:.4f}")

    print_logits_stats(rgb_logits, "RGB")
    print_logits_stats(modal_logits, "Modal")
    print_softmax_stats(rgb_conf, "RGB")
    print_softmax_stats(modal_conf, "Modal")

    rgb_class_counts = torch.zeros(NUM_CLASSES, dtype=torch.int32)
    modal_class_counts = torch.zeros(NUM_CLASSES, dtype=torch.int32)
    final_class_counts = torch.zeros(NUM_CLASSES, dtype=torch.int32)

    for cls in range(NUM_CLASSES):
        rgb_class_counts[cls] = ((rgb_clean == cls) & choose_rgb).sum()
        modal_class_counts[cls] = ((modal_noisy_avg == cls) & choose_modal).sum()
        final_class_counts[cls] = (final_prediction == cls).sum()

    print("\n📊 Class-wise Modality Dominance:")
    print(f"{'Class':>6} | {'RGB Count':>10} | {'Modal Count':>12} | {'Winner':>8} | {'Total':>6}")
    print("-" * 50)
    for cls in range(NUM_CLASSES):
        rgb_c = rgb_class_counts[cls].item()
        modal_c = modal_class_counts[cls].item()
        total_c = final_class_counts[cls].item()
        winner_cls = "RGB" if rgb_c > modal_c else ("Modal" if modal_c > rgb_c else "Tie")
        print(f"{cls:6} | {rgb_c:10} | {modal_c:12} | {winner_cls:8} | {total_c:6}")

# ---- Run for All Available Base IDs ----
all_ensemble_files = os.listdir(ROOT_ENSEMBLE)
base_ids = sorted([f.split('_ensemble')[0] for f in all_ensemble_files if f.endswith('.pt')])

for base_id in base_ids:
    run_confidence_based_pixel_selection(base_id)

