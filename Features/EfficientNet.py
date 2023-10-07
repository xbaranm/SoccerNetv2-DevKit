import torchvision.models as models
import pytorch_lightning as pl
import torch.nn as nn
import torch
import timm

class EfficientNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model = model.eval()
        

    def forward(self, inputs):
        # return self.model.forward_features(inputs) # dim (<num_frames>, 1280, 7, 7)
        
        # For Avg-pooled features
        self.model.reset_classifier(0)
        return self.model(inputs) # dim (<num_frames, 1280)

    def training_step(self, batch, batch_idx):
        return 0

    def configure_optimizers(self):
        return {}
        # return torch.optim.SGD(self.model.parameters(), lr=0.1)