import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

#Install segmentation-models-pytorch
from segmentation_models_pytorch import Unet


class Network(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=2)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=2)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=2)

    def forward(self, x):
        embedding = self.unet(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        y = y[:, 0].long()
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        self.train_acc(logits, y.long())
        self.log('train_acc', self.train_acc, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        y = y[:, 0].long()
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)
        self.valid_acc(logits, y)
        self.log('val_acc', self.valid_acc, on_epoch=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        y = y[:, 0].long()
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)
        self.test_acc(logits, y)
        self.log('test_acc', self.test_acc, on_epoch=True)
        return loss
