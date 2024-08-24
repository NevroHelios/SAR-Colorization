import torch
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io # type: ignore
import numpy as np

from utils import load_model

image_path = Path("ROIs2017_winter_s2\s2_25\ROIs2017_winter_s2_25_p33.tif")

model = load_model("generator")

image = io.imread(image_path)
image = (image - image.min()) / (image.max() - image.min())
img_rgb = np.stack([
    image[:,:,3],  # Red channel
    image[:,:,2],  # Green channel
    image[:,:,1] # Blue Channel
], axis=-1)

img = np.mean(img_rgb, axis=2)
img = torch.tensor(img, dtype=torch.float).unsqueeze(0).unsqueeze(0)
print(img.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
img = img.to(device)

model.eval()
with torch.no_grad():
    out = model(img)

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(img.squeeze().cpu().numpy())
axs[0].set_title("Input")
axs[0].axis('off')
axs[1].imshow(img_rgb)
axs[1].set_title("GT")
axs[1].axis('off')
axs[2].imshow(out[0].permute(1, 2, 0).cpu().numpy())
axs[2].set_title("Predicted")
axs[2].axis('off')
plt.show()