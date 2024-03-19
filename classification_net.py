from typing import Any
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops import sigmoid_focal_loss

from utils.focal_loss import FocalLoss




class classification_net(pl.LightningModule):
    def __init__(self, name='classification_net', input_size=201,  output_size=5, layers=None, loss_function='focal_loss'):
        super(classification_net, self).__init__()

        self.name = name
        self.loss_func_name = str(loss_function)
        self.input_size = input_size
        self.output_size = output_size
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.save_hyperparameters(ignore=['loss_function'])

        if layers is None:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(self.input_size),
                nn.Linear(self.input_size, 172),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.BatchNorm1d(172),
                nn.Linear(172, 128),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.BatchNorm1d(128),
                nn.Linear(128, 96),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.BatchNorm1d(96),
                nn.Linear(96, 64),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.BatchNorm1d(64),
                nn.Linear(64, self.output_size)
                # nn.Softmax(dim=1) #CrossEntropyLoss already has softmax
            )
        else:
            self.layers = layers

        if isinstance(loss_function, str):
            if loss_function == 'focal_loss':
                self.loss_func = FocalLoss(gamma=1.5)
            elif loss_function == 'cross_entropy':
                self.loss_func = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"loss_function of '{loss_function}' is not supported.")
        elif loss_function is None:
            self.loss_func = FocalLoss(gamma=1.5)
        elif isinstance(loss_function, nn.Module):
            self.loss_func = loss_function
        else:
            raise ValueError(f"loss_function has to be of type str or nn.Module.")

        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        if self.training:
            return x
        else:
            return F.softmax(x, dim=1)


    def training_step(self, batch, batch_idx):
        features, label = batch

        output = self(features)

        loss = self.loss_func(output, label)
        mse_loss = self.mse_loss(output, label)
        cross_entropy_loss = self.cross_entropy_loss(output, label)

        self.log('train_step_loss', loss, on_step=True, on_epoch=False)
        self.log('train_epoch_loss', loss, on_step=False, on_epoch=True)
        self.log('train_epoch_mse_loss', mse_loss, on_step=False, on_epoch=True)
        self.log('train_epoch_cross_entropy_loss', cross_entropy_loss, on_step=False, on_epoch=True)
        self.training_step_outputs.append(loss)
        return loss


    def validation_step(self, batch, batch_idx):
        features, label = batch

        output = self(features)

        validation_loss = self.loss_func(output, label)
        mse_validation_loss = self.mse_loss(output, label)
        cross_entropy_validation_loss = self.cross_entropy_loss(output, label)
        acc = (output.argmax(dim=1) == label.argmax(dim=1)).float().mean()

        self.log('validation_loss', validation_loss)
        self.log('validation_mse_loss', mse_validation_loss)
        self.log('validation_cross_entropy_loss', cross_entropy_validation_loss)
        self.log('validation_acc', acc, on_step=False, on_epoch=True)
        self.validation_step_outputs.append(validation_loss)
        return validation_loss


    def on_training_epoch_end(self):
        outputs  = self.training_step_outputs

        epoch_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.training_step_outputs.clear()
        self.logger.experiment.add_scalars('epoch_loss', {'train': epoch_train_loss}, self.current_epoch)


    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        epoch_validation_loss = torch.stack(outputs).mean()
        self.validation_step_outputs.clear()
        self.logger.experiment.add_scalars('epoch_loss', {'validation': epoch_validation_loss}, self.current_epoch)


    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        #optimizer = torch.optim.Adam(self.parameters())
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        optimizer = torch.optim.AdamW(self.parameters())

        return [optimizer]
        # optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=np.sqrt(0.5))
        # return {
        #     "optimizer":optimizer,
        #     "lr_scheduler" : {
        #         "scheduler" : scheduler,
        #         "monitor" : "train_epoch_loss",

        #     }
        # }