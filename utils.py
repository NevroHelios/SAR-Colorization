import torch.nn as nn
from pathlib import Path
import torch
from model import SARModel

def save_model(model:nn.Module, name:str|Path):
    path = Path("models")
    path.mkdir(exist_ok=True)
    model_path = path / f"{name}.pt"
    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)

def load_model(name:str|Path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    path = Path("models")
    model_path = path / f"{name}.pt"
    print(f"Loading model from {model_path}")
    model = SARModel(device=device).generator
    model.load_state_dict(torch.load(model_path))
    return model