from typing import Any
import numpy as np
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor

from utils.focal_loss import FocalLoss
import torchmetrics.functional as FM
import torchmetrics
import torchaudio

from torchvision.models import resnet50, resnet34, efficientnet_b5, vit_b_16


#from layer_library import Discriminator, GeneratorUnetFromLatent, EConvLayer, E_Normalization_Type, EActivationType


def get_conv2d(conv2d_type, n_ch_in, n_ch_out, kernel_size=3, stride=2, padding=1):
    if conv2d_type is None:
        return Conv2dNormal(n_ch_in, n_ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if conv2d_type == "stack_time":
        return Conv2dStackTime(n_ch_in+1, n_ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if conv2d_type == "stack_time_new":
        return Conv2dStackTime_New(n_ch_in+1, n_ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    elif "stack_class" in conv2d_type:
        return Conv2dStackClass(n_ch_in+int(conv2d_type[-1]), n_ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    return Conv2dNormal(n_ch_in, n_ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)



class Conv2dStackTime(nn.Conv2d):
    def forward(self, x, dict_inputs=None):
        time_feature = dict_inputs['time_feature']
        replicated_time_feature = time_feature*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        #print("time_feature.shape: {}".format(replicated_time_feature.shape))
        #print("x.shape: {}".format(x.shape))
        x = torch.cat([x, replicated_time_feature], dim=1)
        return super().forward(x)


class Conv2dStackTime_New(nn.Conv2d):
    def forward(self, x, dict_inputs=None):
        time_feature = dict_inputs['time_feature']
        replicated_time_feature = time_feature.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        #print("time_feature.shape: {}".format(replicated_time_feature.shape))
        #print("x.shape: {}".format(x.shape))
        x = torch.cat([x, replicated_time_feature], dim=1)
        return super().forward(x)



class Conv2dStackClass(nn.Conv2d):
    def forward(self, x, dict_inputs=None):
        class_vector = torch.clone(dict_inputs['class_vector']).unsqueeze(dim=-1).unsqueeze(dim=-1)
        #print("x.shape: {}".format(x.shape))
        #print("class_vector shape: {}".format(class_vector.shape))
        #replicated_class_vector = class_vector*torch.ones((x.shape[0], len(class_vector), x.shape[2], x.shape[3]), device=x.device)
        replicated_class_vector = class_vector.repeat(1, 1, x.shape[2], x.shape[3])

        x = torch.cat([x, replicated_class_vector], dim=1)
        return super().forward(x)



class Conv2dNormal(nn.Conv2d):
    def forward(self, x, dict_inputs=None):
        return super().forward(x)


class Downsample(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, conv2d_type=None, padding=1, use_dropout=True, **kwargs):
        super().__init__(**kwargs)

        self.use_dropout = use_dropout

        #self.conv2d = nn.Conv2d()
        self.conv2d = get_conv2d(conv2d_type, n_ch_in, n_ch_out, kernel_size=3, stride=2, padding=padding)
        self.normalization = nn.BatchNorm2d(n_ch_out)
        self.activation = nn.ReLU(inplace=True)
        if use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        else:
            self.dropout = nn.Identity()


    def forward(self, x, dict_inputs=None):
        x = self.conv2d(x, dict_inputs)
        x = self.normalization(x)
        x = self.activation(x)
        return self.dropout(x)
        #return x


class Upsample(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, conv2d_type=None, padding=1, **kwargs):
        super().__init__(**kwargs)

        #self.conv2d = nn.Conv2d()
        self.conv2d = get_conv2d(conv2d_type, n_ch_in, n_ch_out, kernel_size=3, stride=1, padding=padding)
        self.normalization = nn.BatchNorm2d(n_ch_out)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x, dict_inputs=None):
        #print("x shape: {}".format(x.shape))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv2d(x, dict_inputs)
        x = self.normalization(x)
        x = self.activation(x)
        return self.dropout(x)


class ResnetBlock(nn.Module):
    def __init__(self, n_ch, conv2d_type=None, padding="same", use_dropout=True, **kwargs):
        super().__init__(**kwargs)

        self.use_dropout = use_dropout

        self.conv2d = get_conv2d(conv2d_type, n_ch, n_ch, kernel_size=3, stride=1, padding="same")
        self.normalization = nn.BatchNorm2d(n_ch)
        self.activation = nn.ReLU(inplace=True)
        if use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        else:
            self.dropout = nn.Identity()

        self.conv2d_01 = get_conv2d(conv2d_type, n_ch, n_ch, kernel_size=3, stride=1, padding="same")
        self.normalization_01 = nn.BatchNorm2d(n_ch)
        self.activation_01 = nn.ReLU(inplace=True)
        if use_dropout:
            self.dropout_01 = nn.Dropout(p=0.2)
        else:
            self.dropout_01 = nn.Identity()


    def forward(self, x, dict_inputs=None):
        #print(x.shape)
        x_out = self.conv2d(x, dict_inputs)
        #print(x_out.shape)
        x_out = self.normalization(x_out)
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)

        x_out = self.conv2d_01(x_out, dict_inputs)
        #print(x_out.shape)
        x_out = self.normalization_01(x_out)
        x_out = self.activation_01(x_out + x)
        return self.dropout_01(x_out)





class DownBlock(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, conv2d_type=None, padding=1, n_resnet_blocks=0, use_dropout=True, **kwargs):
        super().__init__(**kwargs)

        self.downsample = Downsample(n_ch_in=n_ch_in, n_ch_out=n_ch_out, conv2d_type=conv2d_type,
                                     padding=padding, use_dropout=use_dropout)

        self.resnet_blocks = nn.Sequential()
        for i in range(n_resnet_blocks):
            resnet_block = ResnetBlock(n_ch_out, conv2d_type=conv2d_type, padding="same", use_dropout=use_dropout)
            self.resnet_blocks.add_module("resnet_block_" + str(i), resnet_block)


    def forward(self, x, dict_inputs=None):
        x = self.downsample(x, dict_inputs)
        for i, resnet_block in enumerate(self.resnet_blocks):
            #if not self.training:
            #    if i != 0:
            #        if random.random() > 0.8:
            #            continue
            x = resnet_block(x, dict_inputs)
        return x



class UpBlock(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, conv2d_type=None, padding=1, n_resnet_blocks=0, **kwargs):
        super().__init__(**kwargs)

        self.downsample = Upsample(n_ch_in=n_ch_in, n_ch_out=n_ch_out, conv2d_type=conv2d_type,
                                     padding=padding)

        self.resnet_blocks = nn.Sequential()
        for i in range(n_resnet_blocks):
            resnet_block = ResnetBlock(n_ch_out, conv2d_type=conv2d_type, padding="same")
            self.resnet_blocks.add_module("resnet_block_" + str(i), resnet_block)


    def forward(self, x, dict_inputs=None):
        x = self.downsample(x, dict_inputs)
        for i, resnet_block in enumerate(self.resnet_blocks):
            x = resnet_block(x, dict_inputs)
        return x



class ConvNet2d_Modular(nn.Module):
    def __init__(self, num_classes, bias_initialization=None, **kwargs):
        super().__init__(**kwargs)

        self.conv = nn.Conv2d(1, 32, 3)
        self.resnet_block = ResnetBlock(32, conv2d_type="stack_time")
        self.resnet_block_01 = ResnetBlock(32, conv2d_type="stack_time")
        self.layer0 = DownBlock(32, 32, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer1 = DownBlock(32, 32, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer2 = DownBlock(32, 16, n_resnet_blocks=2, conv2d_type="stack_time")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_last = nn.Linear(17, num_classes)

        if bias_initialization is not None:
            print(self.linear_last.bias.data)
            self.linear_last.bias.data = torch.from_numpy(np.log(bias_initialization.cpu().numpy()))
            print(self.linear_last.bias.data)


    def forward(self, x):
        #time_feature = torch.zeros_like(x[:, -1].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1))
        time_feature = x[:, -1].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = torch.reshape(x[:, :200], (x.shape[0], 1, 25, 8))
        dict_inputs = {"time_feature" : time_feature}

        x = self.conv(x)
        x = self.resnet_block(x, dict_inputs)
        x = self.resnet_block_01(x, dict_inputs)

        x = self.layer0(x, dict_inputs)
        x = self.layer1(x, dict_inputs)
        x = self.layer2(x, dict_inputs)

        x = self.pool(x)
        x = torch.cat([x, time_feature], dim=1)
        x = torch.flatten(x, 1)
        #print(x.shape)
        return self.linear_last(x)


class ConvNet2d_EfficentNetB5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.set_layers(num_classes)


    def set_layers(self, num_classes):
        self.model = efficientnet_b5(pretrained=False)
        self.classifier = nn.Sequential(
            #nn.Dropout(p=0.4, inplace=True),
            nn.Linear(2049, num_classes),
        )


    def forward(self, x, time_feature):
        replicated_time_feature = time_feature*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        x = torch.cat([x, replicated_time_feature], dim=1)
        x = self.model.features(x)

        x = self.model.avgpool(x)
        x = torch.cat([x, time_feature], dim=1)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


class ConvNet2d_ViT_B(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.set_layers(num_classes)


    def set_layers(self, num_classes):
        self.model = vit_b_16(pretrained=False, image_size=160, num_classes=5)

    def forward(self, x, time_feature):
        replicated_time_feature = time_feature*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        x = torch.cat([x, replicated_time_feature], dim=1)

        return self.model(x)


class ConvNet2d_ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.set_layers(num_classes)


    def set_layers(self, num_classes):
        self.model = resnet50(pretrained=False)
        self.fc = nn.Linear(2049, num_classes)


    def forward(self, x, time_feature):
        replicated_time_feature = time_feature*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        x = torch.cat([x, replicated_time_feature], dim=1)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        #replicated_time_feature = time_feature*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        #replicated_time_feature = replicated_time_feature[:, 0]
        #x[:, -1, :, :] = replicated_time_feature
        x = self.model.layer1(x)

        #replicated_time_feature = time_feature*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        #replicated_time_feature = replicated_time_feature[:, 0]
        #x[:, -1, :, :] = replicated_time_feature
        x = self.model.layer2(x)

        #replicated_time_feature = time_feature*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        #replicated_time_feature = replicated_time_feature[:, 0]
        #x[:, -1, :, :] = replicated_time_feature
        x = self.model.layer3(x)

        #replicated_time_feature = time_feature*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        #replicated_time_feature = replicated_time_feature[:, 0]
        #x[:, -1, :, :] = replicated_time_feature
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.cat([x, time_feature], dim=1)
        #x[:, -1] = time_feature[:, 0]
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ConvNet2d_ResNet34(ConvNet2d_ResNet50):
    def set_layers(self, num_classes):
        self.model = resnet34(pretrained=False)
        self.fc = nn.Linear(513, num_classes)






class ConvNet2d_Modular_Image(nn.Module):
    def __init__(self, num_classes, bias_initialization=None, **kwargs):
        super().__init__(**kwargs)
        self.set_layers(bias_initialization, num_classes)

    def set_layers(self, bias_initialization, num_classes):
        self.conv = nn.Conv2d(2, 32, 3)
        self.resnet_block = ResnetBlock(32, conv2d_type="stack_time")
        self.resnet_block_01 = ResnetBlock(32, conv2d_type="stack_time")
        self.layer0 = DownBlock(32, 32, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer1 = DownBlock(32, 32, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer2 = DownBlock(32, 16, n_resnet_blocks=2, conv2d_type="stack_time")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_last = nn.Linear(17, num_classes)

        if bias_initialization is not None:
            print(self.linear_last.bias.data)
            self.linear_last.bias.data = torch.from_numpy(np.log(bias_initialization.cpu().numpy()))
            print(self.linear_last.bias.data)


    def forward(self, x, time_feature):
        dict_inputs = {"time_feature" : time_feature}

        x = self.conv(x)
        x = self.resnet_block(x, dict_inputs)
        x = self.resnet_block_01(x, dict_inputs)

        x = self.layer0(x, dict_inputs)
        x = self.layer1(x, dict_inputs)
        x = self.layer2(x, dict_inputs)

        x = self.pool(x)
        x = torch.cat([x, time_feature], dim=1)
        x = torch.flatten(x, 1)
        #print(x.shape)
        return self.linear_last(x)


    def forward_till_pool(self, x, time_feature):
        dict_inputs = {"time_feature" : time_feature}

        x = self.conv(x)
        x = self.resnet_block(x, dict_inputs)
        x = self.resnet_block_01(x, dict_inputs)

        x = self.layer0(x, dict_inputs)
        x = self.layer1(x, dict_inputs)
        x = self.layer2(x, dict_inputs)

        #print("x.shape: {}".format(x.shape))
        #print("time_feature.shape: {}".format(time_feature.shape))

        return torch.cat([x, time_feature*torch.ones_like(x[:, 0, :, :].unsqueeze(dim=1))], dim=1)


class ConvNet2d_Modular_Image_NoRegularization(ConvNet2d_Modular_Image):
    def set_layers(self, bias_initialization, num_classes):
        self.conv = nn.Conv2d(2, 32, 3)
        self.resnet_block = ResnetBlock(32, conv2d_type="stack_time", use_dropout=False)
        self.resnet_block_01 = ResnetBlock(32, conv2d_type="stack_time", use_dropout=False)
        self.layer0 = DownBlock(32, 32, n_resnet_blocks=2, conv2d_type="stack_time", use_dropout=False)
        self.layer1 = DownBlock(32, 32, n_resnet_blocks=2, conv2d_type="stack_time", use_dropout=False)
        self.layer2 = DownBlock(32, 16, n_resnet_blocks=2, conv2d_type="stack_time", use_dropout=False)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_last = nn.Linear(17, num_classes)

        if bias_initialization is not None:
            print(self.linear_last.bias.data)
            self.linear_last.bias.data = torch.from_numpy(np.log(bias_initialization.cpu().numpy()))
            print(self.linear_last.bias.data)



class ConvNet2d_Modular_Image_DoubleChannels(ConvNet2d_Modular_Image):
    def set_layers(self, bias_initialization, num_classes):
        self.conv = nn.Conv2d(2, 2*32, 3)
        self.resnet_block = ResnetBlock(2*32, conv2d_type="stack_time")
        self.resnet_block_01 = ResnetBlock(2*32, conv2d_type="stack_time")
        self.layer0 = DownBlock(2*32, 2*32, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer1 = DownBlock(2*32, 2*32, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer2 = DownBlock(2*32, 2*16, n_resnet_blocks=2, conv2d_type="stack_time")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_last = nn.Linear(33, num_classes)


class ConvNet2d_Modular_Image_DoubleResBlocks(ConvNet2d_Modular_Image):
    def set_layers(self, bias_initialization, num_classes):
        self.conv = nn.Conv2d(2, 32, 3)
        self.resnet_block = ResnetBlock(32, conv2d_type="stack_time")
        self.resnet_block_01 = ResnetBlock(32, conv2d_type="stack_time")
        self.resnet_block_02 = ResnetBlock(32, conv2d_type="stack_time")
        self.resnet_block_03 = ResnetBlock(32, conv2d_type="stack_time")
        self.layer0 = DownBlock(32, 32, n_resnet_blocks=4, conv2d_type="stack_time")
        self.layer1 = DownBlock(32, 32, n_resnet_blocks=4, conv2d_type="stack_time")
        self.layer2 = DownBlock(32, 16, n_resnet_blocks=4, conv2d_type="stack_time")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_last = nn.Linear(17, num_classes)


    def forward(self, x, time_feature):
        dict_inputs = {"time_feature" : time_feature}

        x = self.conv(x)
        x = self.resnet_block(x, dict_inputs)
        x = self.resnet_block_01(x, dict_inputs)
        x = self.resnet_block_02(x, dict_inputs)
        x = self.resnet_block_03(x, dict_inputs)

        x = self.layer0(x, dict_inputs)
        x = self.layer1(x, dict_inputs)
        x = self.layer2(x, dict_inputs)

        x = self.pool(x)
        x = torch.cat([x, time_feature], dim=1)
        x = torch.flatten(x, 1)
        #print(x.shape)
        return self.linear_last(x)


class ConvNet2d_Modular_Image_NoTimeFeature(nn.Module):
    def __init__(self, num_classes, bias_initialization=None, **kwargs):
        super().__init__(**kwargs)

        self.conv = nn.Conv2d(2, 32, 3)
        self.resnet_block = ResnetBlock(32)
        self.resnet_block_01 = ResnetBlock(32)
        self.layer0 = DownBlock(32, 32, n_resnet_blocks=2)
        self.layer1 = DownBlock(32, 32, n_resnet_blocks=2)
        self.layer2 = DownBlock(32, 16, n_resnet_blocks=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_last = nn.Linear(16, num_classes)

        if bias_initialization is not None:
            print(self.linear_last.bias.data)
            self.linear_last.bias.data = torch.from_numpy(np.log(bias_initialization.cpu().numpy()))
            print(self.linear_last.bias.data)


    def forward(self, x):
        x = self.conv(x)
        x = self.resnet_block(x)
        x = self.resnet_block_01(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        return self.linear_last(x)





class ConvNet2d_Modular_Image_Larger(nn.Module):
    def __init__(self, num_classes, bias_initialization=None, **kwargs):
        super().__init__(**kwargs)

        self.conv = nn.Conv2d(2, 32, 3)
        self.resnet_block = ResnetBlock(32, conv2d_type="stack_time")
        self.resnet_block_01 = ResnetBlock(32, conv2d_type="stack_time")
        self.layer0 = DownBlock(32, 32, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer1 = DownBlock(32, 32, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer2 = DownBlock(32, 32, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer3 = DownBlock(32, 16, n_resnet_blocks=2, conv2d_type="stack_time")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_last = nn.Linear(17, num_classes)

        if bias_initialization is not None:
            print(self.linear_last.bias.data)
            self.linear_last.bias.data = torch.from_numpy(np.log(bias_initialization.cpu().numpy()))
            print(self.linear_last.bias.data)


    def forward(self, x, time_feature):
        dict_inputs = {"time_feature" : time_feature}

        x = self.conv(x)
        x = self.resnet_block(x, dict_inputs)
        x = self.resnet_block_01(x, dict_inputs)

        x = self.layer0(x, dict_inputs)
        x = self.layer1(x, dict_inputs)
        x = self.layer2(x, dict_inputs)
        x = self.layer3(x, dict_inputs)

        x = self.pool(x)
        x = torch.cat([x, time_feature], dim=1)
        x = torch.flatten(x, 1)
        #print(x.shape)
        return self.linear_last(x)



    """
    def __init__(self, num_classes, bias_initialization=None, **kwargs):
        super().__init__(**kwargs)

        self.conv = nn.Conv2d(2, 32, 3)
        self.resnet_block = ResnetBlock(32, conv2d_type="stack_time")
        self.resnet_block_01 = ResnetBlock(32, conv2d_type="stack_time")
        self.layer0 = DownBlock(32, 64, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer1 = DownBlock(64, 128, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer2 = DownBlock(128, 256, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer3 = DownBlock(256, 512, n_resnet_blocks=2, conv2d_type="stack_time")
        self.layer4 = DownBlock(512, 128, n_resnet_blocks=2, conv2d_type="stack_time")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_last = nn.Linear(129, num_classes)

        if bias_initialization is not None:
            print(self.linear_last.bias.data)
            self.linear_last.bias.data = torch.from_numpy(np.log(bias_initialization.cpu().numpy()))
            print(self.linear_last.bias.data)


    def forward(self, x, time_feature):
        dict_inputs = {"time_feature" : time_feature}

        x = self.conv(x)
        x = self.resnet_block(x, dict_inputs)
        x = self.resnet_block_01(x, dict_inputs)

        x = self.layer0(x, dict_inputs)
        x = self.layer1(x, dict_inputs)
        x = self.layer2(x, dict_inputs)
        x = self.layer3(x, dict_inputs)
        x = self.layer4(x, dict_inputs)

        x = self.pool(x)
        x = torch.cat([x, time_feature], dim=1)
        x = torch.flatten(x, 1)
        #print(x.shape)
        return self.linear_last(x)
    """


"""

class Discriminator_(nn.Module):
    def __init__(self, num_classes, bias_initialization=None, **kwargs):
        super().__init__(**kwargs)

        self.conv = nn.Conv2d(1, 32, 3)
        self.resnet_block = ResnetBlock(32)
        self.resnet_block_01 = ResnetBlock(32)
        self.layer0 = DownBlock(32, 32, n_resnet_blocks=1)
        self.layer1 = DownBlock(32, 32, n_resnet_blocks=1)
        self.layer2 = DownBlock(32, 16, n_resnet_blocks=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_last = nn.Linear(16, num_classes)

        if bias_initialization is not None:
            print(self.linear_last.bias.data)
            self.linear_last.bias.data = torch.from_numpy(np.log(bias_initialization.cpu().numpy()))
            print(self.linear_last.bias.data)


    def forward(self, x):
        dict_inputs = {}
        x = self.conv(x)
        x = self.resnet_block(x, dict_inputs)
        x = self.resnet_block_01(x, dict_inputs)

        x = self.layer0(x, dict_inputs)
        x = self.layer1(x, dict_inputs)
        x = self.layer2(x, dict_inputs)

        x = self.pool(x)
        #x = torch.cat([x, time_feature], dim=1)
        x = torch.flatten(x, 1)
        #print(x.shape)
        return self.linear_last(x)



class LinearBlock(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, **kwargs):
        super().__init__(**kwargs)
        self.fc = nn.Linear(n_ch_in, n_ch_out, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        return self.relu(x)


class GANGeneratorDecoder(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.fc_0 = LinearBlock(512+num_classes, 4096)
        self.fc_1 = LinearBlock(4096, 4096)

        self.layer0 = UpBlock(256, 128, padding=0, n_resnet_blocks=2, conv2d_type="stack_class::" + str(num_classes))   #8x8
        self.layer1 = UpBlock(128, 64, padding=0, n_resnet_blocks=2, conv2d_type="stack_class::" + str(num_classes))    #16x16
        self.layer2 = UpBlock(64, 32, padding=0, n_resnet_blocks=2, conv2d_type="stack_class::" + str(num_classes))    #32x32
        self.last = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.center_crop = torchvision.transforms.CenterCrop((25, 8))


    def forward(self, class_vector):
        noise = torch.randn((class_vector.shape[0], 512), device=class_vector.device)
        noise = torch.cat([class_vector, noise], dim=1)
        latent = self.fc_1(self.fc_0(noise))
        latent = torch.reshape(latent, (latent.shape[0], 256, 4, 4))

        dict_inputs = {"class_vector" : class_vector}

        x = self.layer0(latent, dict_inputs=dict_inputs)
        x = self.layer1(x, dict_inputs=dict_inputs)
        x = self.layer2(x, dict_inputs=dict_inputs)
        x = self.center_crop(x)
        x = self.last(x)
        return self.tanh(x)




class GANClassConditional(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.discriminator = Discriminator(num_classes=1)
        self.generator = GANGeneratorDecoder(num_classes)

        self.optim_g, self.optim_d = self.configure_optimizers()


    def forward(self, targets):
        return self.generator(targets)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)


    def training_step(self, x, target, optimizer_idx):
        if optimizer_idx == 0:
            generated = self.generator(target)
            disc_out_fake = self.discriminator(generated)
            disc_loss = self.adversarial_loss(disc_out_fake, torch.ones_like(disc_out_fake))

            return disc_loss


        elif optimizer_idx == 1:
            generated = self.generator(target)
            disc_out_fake = self.discriminator(generated)
            disc_loss_fake = self.adversarial_loss(disc_out_fake, torch.zeros_like(disc_out_fake))

            disc_out_real = self.discriminator(x)
            disc_loss_real = self.adversarial_loss(disc_out_real, 0.9*torch.ones_like(disc_out_real))

            disc_loss = 0.5*(disc_loss_fake + disc_loss_real)
            return disc_loss


    def fit(self, dataloader, num_epochs):
        for epoch in tqdm(range(num_epochs), ascii=True):
            for x, target in tqdm(dataloader, ascii=True, disable=True):
                x = x.to("cuda")
                x = torch.reshape(x[:, :200], (x.shape[0], 1, 25, 8))

                target = target.to("cuda")
                loss = self.training_step(x, target, 0)
                #print(loss)
                self.optim_g.zero_grad()
                loss.backward()
                self.optim_g.step()

                loss = self.training_step(x, target, 1)
                #print(loss)
                self.optim_d.zero_grad()
                loss.backward()
                self.optim_d.step()

        torch.save(self.state_dict(), "trained_models/gans/class_conditional_gan.pt")


    def configure_optimizers(self):
        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        #return [opt_generator, opt_discriminator], []
        return opt_generator, opt_discriminator

class GANFromFeatureExtractor(pl.LightningModule):
    def __init__(self, seg_loss_factor=1.0, seg_net=None, **kwargs):
        super().__init__(**kwargs)
        self.seg_loss_factor = seg_loss_factor
        self.seg_net = seg_net
        self.virtual_set_models()


    def forward(self, mean_activation_vec):
        #print("forward gan patch gen...")
        #print("activation vec shape: {}".format(mean_activation_vec.shape))
        return self.identity_layer_for_output_hook(self.generator(mean_activation_vec))
        #print("output shape: {}".format(output.shape))
        #return output

    def forward_tissue_patch(self, tissue_patch):
        with torch.no_grad():
            mean_activation_vec = self.get_latent_vector(tissue_patch)

        return self.identity_layer_for_output_hook(self.generator(mean_activation_vec))

    def virtual_set_models(self):
        #from xai_core.pt_layer_library_2022_06_03 import ResnetEncoderUnetDecoder, EConvLayer, E_Normalization_Type, Discriminator
        self.generator = GeneratorUnetFromLatent(kernel_size_full_res_conv=None, kernel_size_encoder=4,
                    kernel_size_decoder=4, n_channels_input=1,
                    n_channels_out_encoder=[32, 64, 128, 256, 512], n_channels_out_decoder=[256, 128, 64, 32, 32], n_ch_out=2,
                    use_head_block=True,
                    n_resnet_blocks_encoder=0,
                    bias_initialization=False,
                    n_channels_from_seg_net=17,
                    default_noise_size=200,
                    e_conv_layer=EConvLayer.conv2d,
                    e_normalization_type_encoder=[E_Normalization_Type.no_norm, E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm],
                    e_normalization_type_decoder=E_Normalization_Type.batch_norm)


        self.discriminator = Discriminator(kernel_size=4,
                            e_conv_layer=EConvLayer.bilinear_conv_first,
                            n_channels_out=[32, 64, 128, 256, 256], n_channels_in=[2, 32, 64, 128, 256],
                            n_resnet_blocks=0, e_normalization_type=[E_Normalization_Type.no_norm, E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm,
                                E_Normalization_Type.batch_norm, E_Normalization_Type.batch_norm], use_spectral_norm=True,
                                e_activation_type=EActivationType.leaky_relu, n_ch_last_layer=256,
                            )

        self.identity_layer_for_output_hook = nn.Identity()
        self.identity_layer_for_mask_hook = nn.Identity()
        self.identity_layer_for_discriminator_output_hook = nn.Identity()


    def configure_optimizers(self):
        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        #return [opt_generator], []
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return [opt_generator, opt_discriminator], []


    def get_latent_vector(self, tissue_patches, with_grad=False):
        if with_grad:
            activation_layer2 = self.seg_net.forward_till_pool(tissue_patches)
            return torch.mean(activation_layer2, dim=[2, 3])
        else:
            with torch.no_grad():
                activation_layer2 = self.seg_net.forward_till_pool(tissue_patches)
                return torch.mean(activation_layer2, dim=[2, 3])


    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)


    def on_train_epoch_end(self):
        self.eval()
        if (self.current_epoch % 10 == 0):
            self.seg_net.eval()
            mean_activation_vec = self.get_latent_vector(self.tissue_patch_cached_for_plot.to(memory_format=torch.channels_last))
            with torch.no_grad():
                generated_patch = self.generator(mean_activation_vec)[:, :, :, 25:175]
            print("generated_patch_cached_for_plot.shape: {}".format(self.generated_patch_cached_for_plot.shape))
            print("tissue_patch_cached_for_plot.shape: {}".format(self.tissue_patch_cached_for_plot.shape))
            torchvision.utils.save_image(generated_patch[:, 0].unsqueeze(dim=1), "logs/gans/images/" + str(self.current_epoch) + "_gen_0.jpg")
            torchvision.utils.save_image(generated_patch[:, 1].unsqueeze(dim=1), "logs/gans/images/" + str(self.current_epoch) + "_gen_1.jpg")

            torchvision.utils.save_image(self.tissue_patch_cached_for_plot[:, 0].unsqueeze(dim=1), "logs/gans/images/" + str(self.current_epoch) + "_real_0.jpg")
            torchvision.utils.save_image(self.tissue_patch_cached_for_plot[:, 1].unsqueeze(dim=1), "logs/gans/images/" + str(self.current_epoch) + "_real_1.jpg")
        self.train()

    def training_step(self, b_data, i, optimizer_idx=0):
        self.seg_net.eval()
        tissue_patch = b_data[0]
        self.tissue_patch_cached_for_plot = tissue_patch
        tissue_patch = tissue_patch.to(memory_format=torch.channels_last)

        if optimizer_idx == 0:# or optimizer_idx == 1:
            mean_activation_vec = self.get_latent_vector(b_data[0].to(memory_format=torch.channels_last))
            #print("1 mean activation vec shape: {}".format(mean_activation_vec.shape))

            #print("tissue_patch.shape: {}".format(tissue_patch.shape))
            # crop from 201 time steps to 150
            generated_patch = self.generator(mean_activation_vec)[:, :, :, 25:175]
            self.generated_patch_cached_for_plot = generated_patch
            generated_patch_inc_time = torch.cat([generated_patch, tissue_patch[:, -1, :200].unsqueeze(dim=1)], dim=1)
            mean_activation_vec_pred = self.get_latent_vector(generated_patch_inc_time, with_grad=True)

            L1_loss = F.l1_loss(mean_activation_vec_pred, mean_activation_vec)
            #self.log('L1_loss', L1_loss)

            #return L1_loss*10.0

            disc_out_fake = self.discriminator(generated_patch)

            h = disc_out_fake.shape[2]
            w = disc_out_fake.shape[3]
            #size = 64
            real = torch.ones((tissue_patch.shape[0], 1, h, w)).to(tissue_patch.device)
            fake = torch.zeros((tissue_patch.shape[0], 1, h, w)).to(tissue_patch.device)
            #print("disc out fake shape: {}".format(disc_out_fake.shape))
            disc_loss = self.adversarial_loss(disc_out_fake, real)

            self.log('L1_loss', L1_loss)
            self.log('gen disc loss', disc_loss)

            return disc_loss + self.seg_loss_factor*L1_loss#10.0*L1_loss + disc_loss

        # train discriminator
        if optimizer_idx == 1:
            if i % 5 != 0:
                return None
            mean_activation_vec = self.get_latent_vector(b_data[0].to(memory_format=torch.channels_last))
            #print("2 mean activation vec shape: {}".format(mean_activation_vec_pred.shape))

            generated_patch = self.generator(mean_activation_vec)[:, :, :, 25:175]
            disc_out_fake = self.discriminator(generated_patch)

            h = disc_out_fake.shape[2]
            w = disc_out_fake.shape[3]
            #size = 64
            real = 0.9*torch.ones((tissue_patch.shape[0], 1, h, w)).to(tissue_patch.device)
            fake = torch.zeros((tissue_patch.shape[0], 1, h, w)).to(tissue_patch.device)

            disc_loss_fake = self.adversarial_loss(disc_out_fake, fake)

            disc_out_real = self.discriminator(tissue_patch[:, :2])
            disc_loss_real = self.adversarial_loss(disc_out_real, real)

            disc_loss = 0.5*(disc_loss_fake + disc_loss_real)

            self.log('disc loss real', disc_loss_real)
            self.log('disc loss fake', disc_loss_fake)

            return disc_loss

"""





class classification_net_cnn(pl.LightningModule):
    def __init__(self, input_size=201,  output_size=5,
                 class_weights=None, bias_initialization=None,
                 focal_loss_gamma=None, **kwargs):
        super().__init__(**kwargs)

        self.name = "cnn"
        self.input_size = input_size
        self.output_size = output_size
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.save_hyperparameters()
        """
        if focal_loss_gamma != None:
            self.loss_func = FocalLoss(gamma=focal_loss_gamma)
        else:
            self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
        """
        self.class_4_bias = 3.0

        #self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
        #self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.identity_layer_for_input_hook = nn.Identity()
        self.identity_layer_for_output_hook = nn.Identity()

        self.set_model()
        self.set_loss()

    def set_loss(self):
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.1, weight=None)

    def set_model(self):
        self.layers = ConvNet2d_Modular(self.output_size, bias_initialization=None)#ConvNet2d(self.output_size)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.identity_layer_for_input_hook(x)
        x = self.layers(x)
        #raise RuntimeError
        #x[:, 3] += self.class_4_bias
        #x = self.identity_layer_for_output_hook(x)
        x = F.softmax(x)
        return x


    def _forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)



    def training_step(self, batch, batch_idx):
        features, label = batch

        #features[:, :1] += torch.randn_like(features[:, :1])*0.01

        output = self._forward(features)

        #loss = self.cross_entropy_loss(output, label)
        #print(self.loss_func)
        loss = self.loss_func(output, label)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())#, lr=0.0005)
        #self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        return optimizer

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_acc",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

        #return [optimizer], [torch.optim.lr_scheduler.ReduceLROnPlateau]



class classification_net_cnn_image(classification_net_cnn):
    def set_model(self):
        self.layers = ConvNet2d_Modular_Image(self.output_size, bias_initialization=None)#ConvNet2d(self.output_size)


    def forward_till_pool(self, x):
        time_feature = x[:, 2, 0, 0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = x[:, :2]
        return self.layers.forward_till_pool(x, time_feature)

    def forward(self, x):
        #print("x shape: {}".format(x.shape))
        time_feature = x[:, 2, 0, 0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = x[:, :2]
        x = self.identity_layer_for_input_hook(x)
        x = self.layers(x, time_feature)
        #raise RuntimeError
        #x[:, 3] += self.class_4_bias
        #x = self.identity_layer_for_output_hook(x)
        x = F.softmax(x)
        return x

    def forward_logits(self, x):
        #print("x shape: {}".format(x.shape))
        time_feature = x[:, 2, 0, 0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = x[:, :2]
        x = self.identity_layer_for_input_hook(x)
        x = self.layers(x, time_feature)

        return x

    def _forward(self, x):
        time_feature = x[:, 2, 0, 0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = x[:, :2]
        return self.layers(x, time_feature)


    def training_step(self, batch, batch_idx):
        spectograms, label = batch
        #print("label shape: {}".format(label.shape))
        output = self._forward(spectograms)
        loss = self.loss_func(output, label)

        #print("label values: {}".format(label))
        #print("output shape: {}".format(output.shape))

        #self.train_acc.update(F.softmax(output, dim=1), label)

        return loss



class classification_net_cnn_image_lightning(classification_net_cnn_image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)
        #self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        #y = y[:, 1:]
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        y = torch.argmax(y, dim=1)

        self.valid_acc.update(y_hat, y)


    def validation_epoch_end(self, out):
        self.log('val_acc', self.valid_acc.compute())
        self.valid_acc.reset()

    #def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        #self.log('train_acc', self.train_acc.compute())
        #self.train_acc.reset()
        #return super().training_epoch_end(outputs)


class classification_net_cnn_image_lightning_ViT_B(classification_net_cnn_image_lightning):
    def set_model(self):
        self.layers = ConvNet2d_ViT_B(self.output_size)#ConvNet2d(self.output_size)


class classification_net_cnn_image_lightning_ResNet50(classification_net_cnn_image_lightning):
    def set_model(self):
        self.layers = ConvNet2d_ResNet50(self.output_size)#ConvNet2d(self.output_size)

    """
    def on_train_epoch_end(self) -> None:
        if self.my_current_epoch % 20 == 0:
            self.lr_scheduler.step()
        self.my_current_epoch += 1
        return super().on_train_epoch_end()
    """


class classification_net_cnn_image_lightning_ResNet34(classification_net_cnn_image_lightning):
    def set_model(self):
        self.layers = ConvNet2d_ResNet34(self.output_size)#ConvNet2d(self.output_size)


class classification_net_cnn_image_lightning_EfficentNetB5(classification_net_cnn_image_lightning):
    def set_model(self):
        self.layers = ConvNet2d_EfficentNetB5(self.output_size)#ConvNet2d(self.output_size)



class classification_net_cnn_image_lightning_double_channels(classification_net_cnn_image_lightning):
    def set_model(self):
        self.layers = ConvNet2d_Modular_Image_DoubleChannels(self.output_size, bias_initialization=None)#ConvNet2d(self.output_size)


class classification_net_cnn_image_lightning_double_resnet_blocks(classification_net_cnn_image_lightning):
    def set_model(self):
        self.layers = ConvNet2d_Modular_Image_DoubleResBlocks(self.output_size, bias_initialization=None)#ConvNet2d(self.output_size)


class classification_net_cnn_image_lightning_no_regularization(classification_net_cnn_image_lightning):
    def set_model(self):
        self.layers = ConvNet2d_Modular_Image_NoRegularization(self.output_size, bias_initialization=None)#ConvNet2d(self.output_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())#, lr=0.0005)
        return optimizer

    def set_loss(self):
        self.loss_func = nn.CrossEntropyLoss()




class classification_net_cnn_image_lightning_semi_supervised(classification_net_cnn_image_lightning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pseudo_label_threshold = 0.8

        self.frequency_masking_transforms = [torchaudio.transforms.FrequencyMasking(freq_mask_param=5) for i in range(5)]
        self.time_masking_transforms = [torchaudio.transforms.TimeMasking(time_mask_param=5) for i in range(5)]


    def apply_strong_augmentation(self, spectrogram):
        for transform in self.frequency_masking_transforms:
            spectrogram[:, :2] = transform(spectrogram[:, :2])
        for transform in self.time_masking_transforms:
            spectrogram[:, :2] = transform(spectrogram[:, :2])

        return spectrogram


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, 1:]
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        y = torch.argmax(y, dim=1)

        self.valid_acc.update(y_hat, y)



    def training_step(self, batch, batch_idx):
        #print("training step...")
        spectrograms, labels = batch
        output = self._forward(spectrograms)

        spectrograms_pseudo_labels = []
        pseudo_labels = []

        labeled_labels = []
        labeled_output = []
        #with torch.no_grad():

        #print(labels)
        #print("labels shape: {}".format(labels.shape))
        #print("labels[:, :1] shape: {}".format(labels[:, 1:].shape))
        #raise RuntimeError

        #with torch.no_grad():
        #    labeled_labels = labels[:, 1:]
        #labeled_output = output


        with torch.no_grad():
            softmax_outputs = F.softmax(output, dim=1)
        for i, label in enumerate(labels):
            if label[0] != 1:
                labeled_labels.append(label)
                labeled_output.append(output[i])

            elif torch.max(softmax_outputs[i]) > self.pseudo_label_threshold:
                with torch.no_grad():
                    pseudo_label = F.one_hot(torch.argmax(softmax_outputs[i]), len(softmax_outputs[i]))
                pseudo_labels.append(pseudo_label)
                spectrograms_pseudo_labels.append(spectrograms[i])

        if len(labeled_output) > 0:
            #print("has labeled data")
            labeled_output = torch.stack(labeled_output, dim=0)
            labeled_labels = torch.stack(labeled_labels, dim=0)

        if len(spectrograms_pseudo_labels) > 0:
            #print("has pseudo labeled data")
            spectrograms_pseudo_labels = torch.stack(spectrograms_pseudo_labels, dim=0)
            pseudo_labels = torch.stack(pseudo_labels, dim=0).float()

            with torch.no_grad():
                spectrograms_pseudo_labels = self.apply_strong_augmentation(spectrograms_pseudo_labels)
                #spectrograms_pseudo_labels[:, :2] += torch.randn_like(spectrograms_pseudo_labels[:, :2])*0.1

        loss_labeled = 0
        loss_pseudo_labeled = 0

        #print("len labeled output: {}".format(len(labeled_output)))
        #print("len spectrograms_pseudo_labels: {}".format(len(spectrograms_pseudo_labels)))

        if len(labeled_output) > 0:
            #print("compute loss labels...")
            #print("labeled_labels shape: {}".format(labeled_labels.shape))
            #print("labels shape: {}".format(labels.shape))
            # labeled_labels has 6 classes (includes -1 for not labeled), therefore skip the -1
            loss_labeled = self.loss_func(labeled_output, labeled_labels[:, 1:])*len(labeled_output)

        if len(spectrograms_pseudo_labels) > 0:
            #print("get pred for pseudo labels...")
            #print(len(spectrograms_pseudo_labels))
            #print("spectrograms_pseudo_labels.shape: {}".format(spectrograms_pseudo_labels.shape))
            outputs_pseudo_labeled = self._forward(spectrograms_pseudo_labels)
            #print("run loss function pseudo labels...")
            #print("pseudo_labels.shape : {}".format(pseudo_labels.shape))
            # pseudo labels already does not include the not labeled class, only the same amount of classes the model predicts
            loss_pseudo_labeled = self.loss_func(outputs_pseudo_labeled, pseudo_labels)*len(outputs_pseudo_labeled)

        #print("add loss...")
        if (len(labeled_labels) + len(pseudo_labels)) > 0:
            loss = (loss_labeled + loss_pseudo_labeled)/(len(labeled_labels) + len(pseudo_labels))
            #print("return loss")
            return loss
        else:
            #print("return none")
            return None




class classification_net_cnn_image_no_time_feature(classification_net_cnn):
    def set_model(self):
        self.layers = ConvNet2d_Modular_Image_NoTimeFeature(self.output_size, bias_initialization=None)#ConvNet2d(self.output_size)


    def forward(self, x):
        #print("x shape: {}".format(x.shape))
        time_feature = x[:, 2, 0, 0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = x[:, :2]
        x = self.identity_layer_for_input_hook(x)
        x = self.layers(x)
        #raise RuntimeError
        #x[:, 3] += self.class_4_bias
        #x = self.identity_layer_for_output_hook(x)
        x = F.softmax(x)
        return x

    def _forward(self, x):
        return self.layers(x)


    def training_step(self, batch, batch_idx):
        x, label = batch

        time_feature = x[:, 2, 0, 0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = x[:, :2]

        output = self._forward(x)

        #loss = self.cross_entropy_loss(output, label)
        #print(self.loss_func)
        loss = self.loss_func(output, label)
        return loss






class classification_net_cnn_image_larger(classification_net_cnn_image):
    def set_model(self):
        self.layers = ConvNet2d_Modular_Image_Larger(self.output_size, bias_initialization=None)#ConvNet2d(self.output_size)


class SimpleBinaryClassifier(nn.Module):
    def __init__(self, n_features, weight=torch.tensor([1.0, 1.0]), **kwargs):
        super().__init__(**kwargs)

        self.layers = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(32, 2),
            )

        self.loss = nn.CrossEntropyLoss(weight=weight)


    def _forward(self, x):
        return self.layers(x)

    def forward(self, x):
        return F.softmax(self.layers(x))



    def training_step(self, x, targets):
        pred = self._forward(x)
        #print("pred.shape {}".format(pred.shape))
        #print("targets.shape {}".format(targets.shape))

        return self.loss(pred, targets)
        #return torchvision.ops.sigmoid_focal_loss(pred, targets, reduction="mean", alpha=0.75)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())

        return [optimizer]


