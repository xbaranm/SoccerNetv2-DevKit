import pytorch_lightning as pl
import torch
import timm
import math

from torch import nn
from torchmetrics.regression import MeanSquaredError

ALPHA = math.sqrt(8)
BETA  = math.sqrt(2)

class OUR_EfficientNet(pl.LightningModule):
    def __init__(self, max_epochs=1000, temperature=0.7, batch_size=16, lr=1e-3, output="default", mirrored=False, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()

        self.lr = lr
        self.conv = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, global_pool='')
        self.channel = self.conv.num_features
        self.conv_head = nn.Sequential(
            nn.Conv2d(self.channel, self.channel*2, 3, 2),  # 7x7 -> 3x3
            nn.BatchNorm2d(self.channel*2),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel*2, self.channel*2, 3, 1), # 3x3 -> 1x1
            nn.BatchNorm2d(self.channel*2),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(self.channel*2*2, self.channel),
            nn.BatchNorm1d(self.channel),
            nn.LeakyReLU(),
            nn.Linear(self.channel, 1)
        )

        self.loss_fn = nn.MSELoss(reduction='none')

    @staticmethod
    def Name():
        return 'OUR_EfficientNet'

    def forward(self, inputs):
        if self.hparams.output in 'features':
            self.conv.reset_classifier(0)
            return self.conv(inputs)
        
        # Split pair of images that was concatenated in channel dimension
        inputs_1, inputs_2 = inputs.split(3, dim=1)  # Bx6xHxW -> 2x Bx3xHxW
        # inputs_1, inputs_2 = inputs_1.float(), inputs_2.float()
                
        # Split inputs in axis=2 into inputs_1 and inputs_2
        out_1 = self.conv.forward_features(inputs_1)
        out_1 = self.conv_head(out_1)

        out_2 = self.conv.forward_features(inputs_2)
        out_2 = self.conv_head(out_2)    

        # concat out_1 and out_2 into out in axis=0
        out = torch.concatenate((out_1, out_2), dim=1)
        result = self.head(out)

        return result

    # TODO: setup optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.conv.parameters()) +
                                     list(self.conv_head.parameters()) +
                                     list(self.head.parameters())
                                     , lr=self.hparams.lr)
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        # )
        
        # return [optimizer], [self.lr_scheduler]
        return [optimizer], []

    def _calculate_loss(self, batch, mode="train"):
        inputs, targets = batch

        if self.hparams.mirrored:
            inputs, inputs_mir = inputs.split(6, dim=1) # Bx12xHxW -> 2x Bx6xHxW

        out = self.forward(inputs)
        if torch.isnan(out).any():
            print("Warning: out is NaN")
            print(f'out: {out}')
            print(f'targets: {targets}')
            torch.nan_to_num(out, nan=0.0)
            print(f'out after nan_to_num(): {out}')
            if torch.isnan(out).any():
                print(f'out is NaN even after nan_to_num()')
                raise Exception("Warning: out is NaN")

        if self.hparams.mirrored:
            out_mir = self.forward(inputs_mir)
        
        targets = targets.unsqueeze(1)    # [16] -> [16,1]

        if self.hparams.mirrored:
            loss = self.loss_fn(torch.concat((out.float(), out_mir.float()), axis=1), 
                                torch.concat((targets.float(), targets.float()), axis=1))
        else:
            loss = self.loss_fn(out.float(), targets.float())

        loss_before_adjustments = loss
        # Spatial distance coeficient
        coef = ALPHA/(targets + BETA)
        # coef = torch.div(ALPHA, targets + BETA)
        loss = torch.mul(loss, coef).mean()
        torch.nan_to_num(loss, nan=0.0)

        # Add sync_dist=True to sync logging across all GPU workers
        self.log(mode + "_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="valid")
    
    def on_train_epoch_end(self):
        pass
        # if self.lr_scheduler:
        #     # Add sync_dist=True to sync logging across all GPU workers
        #     self.log("learning_rate", self.lr_scheduler.get_last_lr()[0], on_step=False, on_epoch=True, sync_dist=True)