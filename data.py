import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from skimage import io # type: ignore

class create_dataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = io.imread(self.image_paths[idx])
            # Normalize the image to 0-1 range for each channel
        img_normalized = (img - img.min()) / (img.max() - img.min())

        # Create a pseudo-color image
        pseudo_color = np.stack([
            img_normalized[:,:,3],  # Red channel
            img_normalized[:,:,2],  # Green channel
            img_normalized[:,:,1] # img_normalized[:,:,1]) / 2  # Blue channel (average of the two)
        ], axis=-1)
        img = np.mean(pseudo_color, axis=2)
        img = torch.tensor(img, dtype=torch.float).unsqueeze(0)
        pseudo_color = torch.tensor(pseudo_color, dtype=torch.float).permute(2, 0, 1)
        return img, pseudo_color
    

def build_dataloader(image_paths, shuffle:bool):
    dataset = create_dataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=shuffle)
    return dataloader