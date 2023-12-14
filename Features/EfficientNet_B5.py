import pytorch_lightning as pl
import timm


class EfficientNet_B5(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model = timm.create_model('efficientnet_b5', pretrained=True, num_classes=0, global_pool='')
        self.model = model.eval()

    @staticmethod
    def Name():
        return 'OUR_EfficientNet'

    def forward(self, inputs):

        self.model.reset_classifier(0)
        return self.model(inputs) # output dim: 2048

    def configure_optimizers(self):
        return {}

    def training_step(self, batch, batch_idx):
        return 0