from PIL import Image
import numpy as np

# Path to the label image
image_path = "/home/scai/visitor/himlelab.visitor/scratch/Sigma/dataloader/datasets/MFNet/Label/00001D_flip.png"

# Load image
img = Image.open(image_path)

# Convert to NumPy array
img_array = np.array(img)

# Print shape and matrix
print("Image shape:", img_array.shape)
print("Image matrix:\n", img_array)

# Get unique pixel values and their counts
unique_values, counts = np.unique(img_array, return_counts=True)

# Print unique pixel values
print("Unique pixel values in the image:", unique_values)

# Print count of each pixel value
print("Counts for each pixel value:")
for val, count in zip(unique_values, counts):
    print(f"Value {val}: {count} pixels")
