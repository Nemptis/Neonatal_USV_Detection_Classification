import numpy as np
from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

from utils.focal_loss import FocalLoss





class fnn(pl.LightningModule):
    def __init__(self, name='fnn', input_size=201,  output_size=5, layers=None, loss_function='focal_loss'):
        super(fnn, self).__init__()

        self.name = name
        self.loss_func_name = str(loss_function)
        self.input_size = input_size
        self.output_size = output_size
        self.save_hyperparameters(ignore=['loss_function'])
        
        if layers is None:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(self.input_size),
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(128),
                nn.Linear(128, 96),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(96),
                nn.Linear(96, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(64)
            )
        else:
            self.layers = layers
            
        self.last_layer = nn.Sequential(
                nn.Linear(65, 32),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.BatchNorm1d(32),
                nn.Linear(32, self.output_size)
            ) 
    
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
        s, t = x
        s = self.layers(s)
        x = torch.cat((s, t), dim=1)
        x = self.last_layer(x)
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
        return validation_loss



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return [optimizer]






class feature_generator():
    def __init__(self, from_key='Sxx_db', length_div = 0.15) -> None:
        self.from_key = from_key
        self.length_div = length_div



    def generate(self, I):
        Sxx = I[self.from_key]
        Sxx -= np.min(Sxx)
        Sxx /= np.max(Sxx)
        rel_duration = (I['end_time'] - I['start_time']) / self.length_div #maximum length = 150 ms
        
        return (torch.tensor(Sxx, dtype=torch.float32), torch.tensor([rel_duration], dtype=torch.float32))




def add_gauss(x, noise=0.05):
    return (x[0] + torch.randn(x[0].size()) * noise, x[1])







class Collate():
    def __init__(self, feature_key='features', resolution=(64,64), crop=(4,4), rand_crop_range=(0, 10), noise=0.05) -> None:
        self.feature_key = feature_key
        self.resolution = resolution
        self.crop = crop
        self.rand_crop_range = rand_crop_range
        self.noise = noise
    
        self.torch_resize = transforms.Resize(resolution, antialias=True)
        
        
    
    def T(self, Sxx, dur):
        a, b = self.crop
        return (self.torch_resize(Sxx[:,a:Sxx.shape[1]-b].unsqueeze(0)).squeeze(0).contiguous().view(-1), dur)



    def T_rand(self, Sxx, dur):
        a, b = torch.randint(*self.rand_crop_range, [2])
        return (self.torch_resize(Sxx[:,a:Sxx.shape[1]-b].unsqueeze(0)).squeeze(0).contiguous().view(-1), dur)



    def collate_fn(self, batch):
        feat_key = self.feature_key
        batch = [(self.T(*d[feat_key]), d['one_hot_category']) for d in batch]
        return torch.utils.data.default_collate(batch)



    def collate_fn_noise(self, batch):
        feat_key = self.feature_key
        batch = [(add_gauss(self.T_rand(*d[feat_key])), d['one_hot_category']) for d in batch]
        return torch.utils.data.default_collate(batch)
