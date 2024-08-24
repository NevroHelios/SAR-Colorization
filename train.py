import torch
import torch.nn as nn
from pathlib import Path

from model import SARModel, LR4ColSAR
from data import build_dataloader
from utils import save_model

base_dir = Path("ROIs2017_winter_s2")
filepaths = list(base_dir.glob("*/*.tif"))

train_images = filepaths[:]
# val_images = filepaths[int(len(filepaths) * 0.8):]

train_dataloader = build_dataloader(train_images, shuffle=True)
# val_dataloader = build_dataloader(val_images, shuffle=False)

print(len(train_images))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SARModel(device=device)


# model = LR4ColSAR(device='cuda')
model.train(train_dataloader=train_dataloader, epochs=30, lr=1e-3)
save_model(model.generator, "generator")