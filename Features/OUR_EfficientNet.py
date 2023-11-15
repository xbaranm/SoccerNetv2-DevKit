import pytorch_lightning as pl
import torch
import timm

from torch import nn
from torchmetrics.regression import MeanSquaredError

class OUR_EfficientNet(pl.LightningModule):
    def __init__(self, max_epochs=1000, temperature=0.7, batch_size=16, lr=1e-3, output="default", **kwargs):
        super().__init__()
        
        self.save_hyperparameters()

        self.lr = lr
        self.conv = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, global_pool='')
        self.head = nn.Sequential(
            nn.Linear(1280*7*7*2, 1280*2),  # Adjust input size
            # nn.BatchNorm(512),
            nn.ReLU(),
            nn.Linear(1280*2, 1)
        )

        self.loss_fn = MeanSquaredError().to('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def Name():
        return 'OUR_EfficientNet'

    def forward(self, inputs):
        if self.hparams.output in 'features':
            self.conv.reset_classifier(0)
            return self.conv(inputs)
        
        # Split pair of images that was concatenated in channel dimension
        inputs_1, inputs_2 = inputs.split(3, dim=1)  # Bx6xWxH -> 2x Bx3xWxH
        inputs_1, inputs_2 = inputs_1.float(), inputs_2.float()
        
        # Split inputs in axis=2 into inputs_1 and inputs_2
        out_1 = self.conv.forward_features(inputs_1).view(-1, 1280*7*7)         
        out_2 = self.conv.forward_features(inputs_2).view(-1, 1280*7*7)         
        
        # concat out_1 and out_2 into out in axis=0
        out = torch.concatenate((out_1, out_2), dim=1)
        result = self.head(out)

        return result

    # TODO: setup optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.conv.parameters()) + list(self.head.parameters()), lr=self.hparams.lr)
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        # )
        
        # return [optimizer], [self.lr_scheduler]
        return [optimizer], []

    def _calculate_loss(self, batch, mode="train"):
        inputs, targets = batch

        out = self.forward(inputs)
        targets = targets.unsqueeze(1)    # [16] -> [16,1]

        # loss_fn = MeanSquaredError().to('cuda' if torch.cuda.is_available() else 'cpu')
        loss = self.loss_fn(out, targets)

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