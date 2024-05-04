import pytorch_lightning as pl
import torch
import timm
import math

from torch import nn
from torchmetrics.regression import MeanSquaredError

class ImageNet_ResNET_152(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.conv = timm.create_model('resnet152', pretrained=True, num_classes=0, global_pool='')


    @staticmethod
    def Name():
        return 'Imagenet_ResNET_152'

    def forward(self, inputs):
        self.conv.reset_classifier(0)
        return self.conv(inputs)

    def configure_optimizers(self):
        return []

    def training_step(self, batch, batch_idx):
        # return self._calculate_loss(batch, mode="train")
        return 1

    def validation_step(self, batch, batch_idx):
        # return self._calculate_loss(batch, mode="valid")
        return 1
    
    def on_train_epoch_end(self):
        pass