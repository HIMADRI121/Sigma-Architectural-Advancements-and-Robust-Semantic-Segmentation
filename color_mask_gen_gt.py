'''from PIL import Image
import numpy as np

# ---------- Load Grayscale Label ----------
label_path = "/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/Label/00001D_flip.png"  # ← path to your grayscale label
label_img = Image.open(label_path).convert("L")
label_array = np.array(label_img)

# ---------- Define MFNet Color Palette ----------
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

# ---------- Convert Grayscale Mask to Color Mask ----------
color_mask = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
for class_idx, color in enumerate(palette):
    color_mask[label_array == class_idx] = color

# ---------- Save or Show Result ----------
color_img = Image.fromarray(color_mask)
color_img.save("00001D_flip_colored.png")  # Save to disk
color_img.show()  # Or show directly
'''
import os
from PIL import Image
import numpy as np

# ---------- Paths ----------
input_dir = "/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/Label"  # grayscale masks (0–8)
output_dir = "/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/Label_Colored"  # save color versions

os.makedirs(output_dir, exist_ok=True)

# ---------- MFNet Color Palette ----------
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

# ---------- Process All Label Files ----------
for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".png"):
        continue

    # Load grayscale label
    label_path = os.path.join(input_dir, fname)
    label_img = Image.open(label_path).convert("L")
    label_array = np.array(label_img)

    # Create empty RGB image
    color_mask = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        color_mask[label_array == class_idx] = color

    # Save colored mask
    out_path = os.path.join(output_dir, fname)
    Image.fromarray(color_mask).save(out_path)
    print(f"✅ Saved: {out_path}")

print("\n🎉 Done coloring all masks!")
