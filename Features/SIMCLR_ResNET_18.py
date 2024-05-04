import pytorch_lightning as pl
import torch
import timm
from torch import nn

from NTXentLoss import NTXent_loss_fn

class ResNET_18(pl.LightningModule):
    def __init__(self, max_epochs=1000, temperature=0.7, batch_size=16, lr=1e-3, output="default", **kwargs):
        super().__init__()
        
        self.save_hyperparameters()

        self.lr = lr
        self.model = timm.create_model('resnet18.a1_in1k', pretrained=False, num_classes=0) # Outputs Bx512

        hidden_dim = 512
        out_dim = 128 # According to paper
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    @staticmethod
    def Name():
        return 'ResNET_18'

    def forward(self, inputs):
        if self.hparams.output == "features":
            return self.model(inputs)
        
        out = self.model(inputs)
        out = self.head(out)
        return out
        # inputs.shape = [16, 3, 224, 224] - [batch_size, channels, width, height]
        if self.hparams.output == "unpooled":                   # [len(inputs), 2048, 10, 10]
            return self.forward_features(inputs)
        elif self.hparams.output == "unpooled_no_classifier":   # [len(inputs), 1024, 7, 7] 
            self.model.reset_classifier(0, '')
            return self.model(inputs)
        elif self.hparams.output == "pooled_features":          # [len(inputs), 1024] 
            self.model.reset_classifier(0)
            return self.model(inputs)
        else: # "default"                                       # [len(inputs), 1000]
            return self.model(inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [self.lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        a, b = batch

        a_out = self.forward(a)
        b_out = self.forward(b)

        loss = NTXent_loss_fn(a_out, b_out, self.hparams.temperature)

        # Add sync_dist=True to sync logging across all GPU workers
        self.log(mode + "_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="valid")
    
    def on_train_epoch_end(self):
        if self.lr_scheduler:
            # Add sync_dist=True to sync logging across all GPU workers
            self.log("learning_rate", self.lr_scheduler.get_last_lr()[0], on_step=False, on_epoch=True, sync_dist=True)