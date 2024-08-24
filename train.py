import torch
import torch.nn as nn
from pathlib import Path

from model import SARModel
from data import build_dataloader

base_dir = Path("ROIs2017_winter_s2")
filepaths = list(base_dir.glob("*/*.tif"))

train_images = filepaths[:int(len(filepaths) * 0.1)]
# val_images = filepaths[int(len(filepaths) * 0.8):]

train_dataloader = build_dataloader(train_images, shuffle=True)
# val_dataloader = build_dataloader(val_images, shuffle=False)

print(len(train_images))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SARModel(device=device)
model.train(train_dataloader)
