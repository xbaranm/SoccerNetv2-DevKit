import pytorch_lightning as pl
import timm


class ResNET_18(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model = timm.create_model('resnet18.a1_in1k', pretrained=True, num_classes=0, global_pool='')
        self.model = model.eval()

    @staticmethod
    def Name():
        return 'ImageNet_ResNET'

    def forward(self, inputs):

        self.model.reset_classifier(0)
        return self.model(inputs)

    def configure_optimizers(self):
        return {}

    def training_step(self, batch, batch_idx):
        return 0