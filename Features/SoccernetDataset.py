import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SoccernetDataset(Dataset):
    def __init__(self, frames):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.frames = torch.tensor(frames, dtype=torch.float).permute(0, 3, 1, 2)


    def __len__(self):
        self.count = self.frames.size()[0]
        return self.count

    def __getitem__(self, idx):   
        return self.frames[idx].to(self.device)
