from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# List of image file paths (p125 to p128)
image_files = [
    Path('E:\\Projects\\SAR colorization\\ROIs2017_winter_s1\\s1_116\\ROIs2017_winter_s1_116_p125.tif'),
    Path('E:\\Projects\\SAR colorization\\ROIs2017_winter_s1\\s1_116\\ROIs2017_winter_s1_116_p126.tif'),
    Path('E:\\Projects\\SAR colorization\\ROIs2017_winter_s1\\s1_116\\ROIs2017_winter_s1_116_p127.tif'),
    # Path('E:\\Projects\\SAR colorization\\ROIs2017_winter_s1\\s1_116\\ROIs2017_winter_s1_116_p128.tif')
]

# Create a 3x3 grid
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# Loop through each image file and place it in the grid
for idx, img_file in enumerate(image_files):
    # Read the image stack
    img = io.imread(img_file)
    print(f"Original: {img.shape}")
    img = np.mean(img, axis=2)
    print(f"Combined: {img.shape}")
    # Check the shape (expecting (256, 256, 2))
    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(img_file.name)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
