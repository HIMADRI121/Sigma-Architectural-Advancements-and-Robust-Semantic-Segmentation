from PIL import Image
import numpy as np

# ---------- MFNet Palette ----------
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

# ---------- Function: RGB → Class Index ----------
def rgb_to_class(mask):
    mask = np.array(mask)
    class_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for color, class_id in color2class.items():
        match = np.all(mask == color, axis=-1)
        class_map[match] = class_id
    return class_map

# ---------- Load & Convert Prediction ----------
pred_path = "/home/scai/visitor/himlelab.visitor/scratch/Sigma/results/final_predictions/00002D_confidence_winner.png"
rgb_mask = Image.open(pred_path).convert('RGB')
class_mask = rgb_to_class(rgb_mask)

# ---------- Verify Unique Class Indices ----------
unique_vals = np.unique(class_mask)
print("✅ Unique class values in prediction:", unique_vals)

if np.all(np.isin(unique_vals, list(range(9)))):
    print("✅ Prediction correctly maps to class indices (0–8).")
else:
    print("❌ Still invalid values — check your color palette or image.")

# ---------- Optional: Visualize Class Mask as RGB ----------
color_mask = np.zeros((class_mask.shape[0], class_mask.shape[1], 3), dtype=np.uint8)
for class_idx, color in enumerate(palette):
    color_mask[class_mask == class_idx] = color

Image.fromarray(color_mask).show()  # Optional visualization
