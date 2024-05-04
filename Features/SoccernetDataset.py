import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SoccernetDataset(Dataset):
    def __init__(self, frames):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Format of frames: [#frames, H, W, C]
        self.frames = frames[:, :, :, [2,1,0]] # BGR -> RGB
        self.count = self.frames.shape[0]

        # self.frames = torch.tensor(frames, dtype=torch.float).permute(0, 3, 1, 2)
        # self.frames = self.frames[:, [2,1,0], :, :]

        self.normalization = transforms.Compose([               # normalizing input according to PyTorch https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html#torchvision.models.ResNet152_Weights
            transforms.ToTensor(), # (H x W x C) -> (C x H x W) and rescales values to [0.0, 1.0]
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], # Pytorch pretrained
                                # std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),               # SimCLR
        ])



    def __len__(self):
        # self.count = self.frames.size()[0]
        return self.count

    def __getitem__(self, idx):
        # return self.frames[idx].to(self.device)
        
        image = self.frames[idx]
        image = self.normalization(image)
        return image.to(self.device)
