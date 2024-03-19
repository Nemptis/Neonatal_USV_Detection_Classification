import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from enum import Enum
from torch.utils.checkpoint import checkpoint
#from fairscale.nn import checkpoint_wrapper
import torchvision
#from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils import spectral_norm
import torch.distributed as dist

use_gradient_checkpointing = False
use_activation_functions = True
padding_mode = "reflect"
do_reflect_padding_manually = False
relu_inplace = True

class E_Normalization_Type(Enum):
    no_norm = 1
    batch_norm = 2
    group_norm = 3
    instance_norm = 4
    adaptive_instance_norm = 5
    instance_norm_running = 6
    pixel_norm = 7
    batch_norm_replace_activations = 8


class EDownsamplingType(Enum):
    convolution = 1
    bilinear = 2
    average_pooling = 3
    bilinear_conv_first = 4

class EActivationType(Enum):
    relu = 1
    leaky_relu = 2
    tanh = 3
    relu_replace_activation = 4

class EConvLayer(Enum):
    conv2d = 1
    conv2dsegmap = 2
    conv2dsegmap_noise = 3
    conv2dnoise = 4
    conv2dreplace_activations = 5
    bilinear = 6
    bilinear_conv_first = 7
    conv2dadd_input = 8



def add_normalization_layer(sequential_layers, e_normalization_type, n_ch, name_append=""):
    if e_normalization_type == E_Normalization_Type.batch_norm:
        sequential_layers.add_module("batch_norm" + name_append, nn.BatchNorm2d(n_ch, eps=0.001))
    elif e_normalization_type == E_Normalization_Type.group_norm:
        sequential_layers.add_module("group_norm" + name_append, nn.GroupNorm(32, n_ch))
    elif e_normalization_type == E_Normalization_Type.instance_norm:
        sequential_layers.add_module("instance_norm" + name_append, nn.InstanceNorm2d(n_ch, affine=True))
    elif e_normalization_type == E_Normalization_Type.instance_norm_running:
        sequential_layers.add_module("instance_norm" + name_append, nn.InstanceNorm2d(n_ch, affine=True, track_running_stats=True))
    elif e_normalization_type == E_Normalization_Type.adaptive_instance_norm:
        sequential_layers.add_module("adaptive_instance_norm" + name_append, AdaptiveInstanceNorm(n_ch, dimension_style_vec=512))
    elif e_normalization_type == E_Normalization_Type.pixel_norm:
        sequential_layers.add_module("pixel_norm" + name_append, PixelNorm())
    #elif e_normalization_type == E_Normalization_Type.spade:
    #    sequential_layers.add_module("spade_norm" + name_append, SPADE(n_ch))

    return sequential_layers


def get_activation_layer(e_activation_type):
    if e_activation_type == EActivationType.relu:
        return nn.ReLU(inplace=relu_inplace)
    elif e_activation_type == EActivationType.leaky_relu:
        return nn.LeakyReLU(0.2, inplace=relu_inplace)
    elif e_activation_type == EActivationType.relu_replace_activation:
        return ReLUReplaceActivations(inplace=relu_inplace)


def get_normalization_layer(e_normalization_type, n_ch, name_append=""):
    if e_normalization_type == E_Normalization_Type.batch_norm:
        return nn.BatchNorm2d(n_ch, eps=0.001)
    elif e_normalization_type == E_Normalization_Type.group_norm:
        return nn.GroupNorm(32, n_ch)
    elif e_normalization_type == E_Normalization_Type.instance_norm:
        return nn.InstanceNorm2d(n_ch, affine=True)
    elif e_normalization_type == E_Normalization_Type.adaptive_instance_norm:
        return AdaptiveInstanceNorm(n_ch, dimension_style_vec=512)
    elif e_normalization_type == E_Normalization_Type.no_norm:
        return nn.Identity()
    elif e_normalization_type == E_Normalization_Type.instance_norm_running:
        return nn.InstanceNorm2d(n_ch, affine=True, track_running_stats=True)
    elif e_normalization_type == E_Normalization_Type.pixel_norm:
        return PixelNorm()
    elif e_normalization_type == E_Normalization_Type.batch_norm_replace_activations:
        return BatchnormReplaceActivations(n_ch, eps=0.001)
    else:
        print(e_normalization_type)
        raise RuntimeError("specified normalization type not understood")



def get_conv_layer(e_conv_layer, in_ch, out_ch, kernel_size, stride, padding, padding_mode, bias=False, conv_n_groups=1):
    if e_conv_layer == EConvLayer.conv2d:
        #conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode, groups=conv_n_groups)
        conv = Conv2dNormal(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding, padding_mode='reflect', groups=conv_n_groups)
    elif e_conv_layer == EConvLayer.conv2dsegmap:
        conv = Conv2dSegMap(in_ch+3, out_ch, kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode, groups=conv_n_groups)
    elif e_conv_layer == EConvLayer.conv2dsegmap_noise:
        conv = Conv2dSegMapNoise(in_channels=in_ch+3, out_channels=out_ch, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode, groups=conv_n_groups)
    elif e_conv_layer == EConvLayer.conv2dnoise:
        conv = Conv2dNoise(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode, groups=conv_n_groups)
    elif e_conv_layer == EConvLayer.conv2dreplace_activations:
        conv = Conv2dReplaceActivations(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode, groups=conv_n_groups)
    elif e_conv_layer == EConvLayer.bilinear:
        conv = Conv2dBilinearDownsample(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, bias=bias, padding=padding, padding_mode=padding_mode, groups=conv_n_groups)
    elif e_conv_layer == EConvLayer.bilinear_conv_first:
        if stride > 1:
            conv = Conv2dBilinearDownsample_ConvFirst(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, bias=bias, padding=padding, padding_mode=padding_mode, groups=conv_n_groups)
        else:
            conv = Conv2dNormal(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode, groups=conv_n_groups)
    elif e_conv_layer == EConvLayer.conv2dadd_input:
        conv = Conv2dAddInput(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode, groups=conv_n_groups)


    elif isinstance(e_conv_layer, list):
        if e_conv_layer[0] == EConvLayer.conv2dsegmap:
            conv = Conv2dSegMap(in_ch+e_conv_layer[1], out_ch, kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode, groups=conv_n_groups)



    return conv


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal(m.weight, 0, 0.02)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal(m.weight, 0, 0.02)




class conv_same_2_resnet_normal_block(nn.Module):
    def __init__(self, n_ch, kernel_size, e_normalization_type=E_Normalization_Type.group_norm, use_spectral_norm=False, e_activation_type=EActivationType.relu,
        e_conv_layer=EConvLayer.conv2d, conv_n_groups=1, **kwargs):
        super().__init__()

        self.resnet_block_1 = nn.Sequential()

        conv = get_conv_layer(e_conv_layer, n_ch, n_ch, kernel_size, stride=1, bias=False, padding="same", padding_mode=padding_mode, conv_n_groups=conv_n_groups)

        if use_spectral_norm:
            self.conv1 = spectral_norm(conv)
        else:
            self.conv1 = conv

        self.normalization_1 = get_normalization_layer(e_normalization_type, n_ch)

        if e_activation_type == EActivationType.relu:
            self.resnet_block_1.add_module("relu_1", nn.ReLU(inplace=relu_inplace))
        elif e_activation_type == EActivationType.leaky_relu:
            self.resnet_block_1.add_module("relu_1", nn.LeakyReLU(0.2, inplace=relu_inplace))

        self.resnet_block_2 = nn.Sequential()

        conv = get_conv_layer(e_conv_layer, n_ch, n_ch, kernel_size, stride=1, bias=False, padding="same", padding_mode=padding_mode, conv_n_groups=conv_n_groups)

        if use_spectral_norm:
            self.conv2 = spectral_norm(conv)#padding = to_pad)
        else:
            self.conv2 = conv

        self.normalization_2 = get_normalization_layer(e_normalization_type, n_ch)

        if e_activation_type == EActivationType.relu:
            self.relu_out = nn.ReLU(inplace=relu_inplace)
        elif e_activation_type == EActivationType.leaky_relu:
            self.relu_out = nn.LeakyReLU(0.2, inplace=relu_inplace)


    def forward_resnet_blocks(self, x, dict_inputs=None):
        if dict_inputs is not None:
            out = self.conv1(x, dict_inputs=dict_inputs)
            if isinstance(self.normalization_1, AdaptiveInstanceNorm):
                out = self.normalization_1(out, dict_inputs=dict_inputs)
            else:
                out = self.normalization_1(out)
        else:
            out = self.conv1(x)
            out = self.normalization_1(out)
        out = self.resnet_block_1(out)
        if dict_inputs is not None:
            out = self.conv2(out, dict_inputs=dict_inputs)
            if isinstance(self.normalization_2, AdaptiveInstanceNorm):
                out = self.normalization_2(out, dict_inputs=dict_inputs)
            else:
                out = self.normalization_2(out)
        else:
            out = self.conv2(out)
            out = self.normalization_2(out)
        return self.resnet_block_2(out)


    def forward(self, x, dict_inputs=None):
        return self.relu_out(self.forward_resnet_blocks(x, dict_inputs) + x)



class conv_same_2_resnet_normal_block_cat_skip(nn.Module):
    def __init__(self, n_ch, kernel_size, e_normalization_type=E_Normalization_Type.group_norm, use_spectral_norm=False, e_activation_type=EActivationType.relu,
        e_conv_layer=EConvLayer.conv2d, conv_n_groups=1, n_ch_skip=None):
        super().__init__()

        self.resnet_block_1 = nn.Sequential()

        conv = get_conv_layer(e_conv_layer, n_ch+n_ch_skip, n_ch, kernel_size, stride=1, bias=False, padding="same", padding_mode=padding_mode, conv_n_groups=conv_n_groups)

        if use_spectral_norm:
            self.conv1 = spectral_norm(conv)
        else:
            self.conv1 = conv

        self.normalization_1 = get_normalization_layer(e_normalization_type, n_ch)

        if e_activation_type == EActivationType.relu:
            self.resnet_block_1.add_module("relu_1", nn.ReLU(inplace=relu_inplace))
        elif e_activation_type == EActivationType.leaky_relu:
            self.resnet_block_1.add_module("relu_1", nn.LeakyReLU(0.2, inplace=relu_inplace))

        self.resnet_block_2 = nn.Sequential()

        conv = get_conv_layer(e_conv_layer, n_ch+n_ch_skip, n_ch, kernel_size, stride=1, bias=False, padding="same", padding_mode=padding_mode, conv_n_groups=conv_n_groups)

        if use_spectral_norm:
            self.conv2 = spectral_norm(conv)#padding = to_pad)
        else:
            self.conv2 = conv

        self.normalization_2 = get_normalization_layer(e_normalization_type, n_ch)

        if e_activation_type == EActivationType.relu:
            self.relu_out = nn.ReLU(inplace=relu_inplace)
        elif e_activation_type == EActivationType.leaky_relu:
            self.relu_out = nn.LeakyReLU(0.2, inplace=relu_inplace)


    def forward_resnet_blocks(self, x, skip, dict_inputs=None):
        #x, skip = x
        x = torch.cat([x, skip], dim=1)
        if dict_inputs is not None:
            out = self.conv1(x, dict_inputs=dict_inputs)
            if isinstance(self.normalization_1, AdaptiveInstanceNorm):
                out = self.normalization_1(out, dict_inputs=dict_inputs)
            else:
                out = self.normalization_1(out)
        else:
            out = self.conv1(x)
            out = self.normalization_1(out)
        out = self.resnet_block_1(out)
        out = torch.cat([out, skip], dim=1)
        if dict_inputs is not None:
            out = self.conv2(out, dict_inputs=dict_inputs)
            if isinstance(self.normalization_2, AdaptiveInstanceNorm):
                out = self.normalization_2(out, dict_inputs=dict_inputs)
            else:
                out = self.normalization_2(out)
        else:
            out = self.conv2(out)
            out = self.normalization_2(out)
        return self.resnet_block_2(out)


    def forward(self, x, dict_inputs=None):
        x, skip = x
        return self.relu_out(self.forward_resnet_blocks(x, skip, dict_inputs) + x)



class conv_same_2_resnet_normal_block_no_skip(conv_same_2_resnet_normal_block):
    def forward(self, x, dict_inputs=None):
        return self.relu_out(self.forward_resnet_blocks(x, dict_inputs))



class conv_same_block(nn.Module):
    def __init__(self, in_ch, out_chs, kernel_size, e_normalization_type=E_Normalization_Type.group_norm, e_conv_layer=EConvLayer.conv2d, use_spectral_norm=False,
                    e_activation_type=EActivationType.relu, conv_n_groups=1):
        super().__init__()

        self.result = nn.Sequential()

        for i, n_ch in enumerate(out_chs):
            if i == 0:
                ch_in = in_ch
            else:
                ch_in = out_chs[i-1]

            #conv1 = nn.Conv2d(in_channels=ch_in, out_channels=n_ch, kernel_size=kernel_size, stride=1, bias=False, padding="same")

            self.conv = get_conv_layer(e_conv_layer, ch_in, n_ch, kernel_size, stride=1, bias=False, padding="same", padding_mode="zeros", conv_n_groups=conv_n_groups)
            """
            if e_conv_layer == EConvLayer.conv2d:
                self.conv = nn.Conv2d(in_channels=ch_in, out_channels=n_ch, kernel_size=kernel_size, stride=1, bias=False, padding="same")
            elif e_conv_layer == EConvLayer.conv2dsegmap:
                self.conv = Conv2dSegMap(in_channels=ch_in+3, out_channels=n_ch, kernel_size=kernel_size, stride=1, bias=False, padding="same")
            """
            if use_spectral_norm:
                self.conv = spectral_norm(self.conv)

            #self.conv = nn.Conv2d(in_channels=ch_in, out_channels=n_ch, kernel_size=kernel_size, stride=1, bias=False, padding="same")
            #self.result.add_module("conv", conv1)

            #self.result = add_normalization_layer(self.result, e_normalization_type, n_ch)
            self.normalization = get_normalization_layer(e_normalization_type, n_ch)

            """
            if e_normalization_type == E_Normalization_Type.batch_norm:
                self.result.add_module("batch_norm", nn.BatchNorm2d(n_ch, eps = 0.001))
            elif e_normalization_type == E_Normalization_Type.group_norm:
                self.result.add_module("group_norm", nn.GroupNorm(32, n_ch))
            elif e_normalization_type == E_Normalization_Type.instance_norm:
                self.result.add_module("instance_norm", nn.InstanceNorm2d(n_ch, affine=True))
            else:
                self.result.add_module("batch_norm", nn.BatchNorm2d(n_ch))
            """
            self.activation_function = get_activation_layer(e_activation_type)

            """
            if e_activation_type == EActivationType.leaky_relu:
                self.result.add_module("relu_1", nn.LeakyReLU(inplace=relu_inplace, negative_slope=0.2))
            else:
                self.result.add_module("relu_1", nn.ReLU(inplace=relu_inplace))
            """

    def forward(self, x, dict_inputs=None):
        #print("forward conv downsample")
        if dict_inputs is not None:
            x = self.conv(x, dict_inputs=dict_inputs)
            #print(self.normalization)
            if isinstance(self.normalization, AdaptiveInstanceNorm):
                x = self.normalization(x, dict_inputs=dict_inputs)
            else:
                x = self.normalization(x)
        else:
            x = self.conv(x)
            x = self.normalization(x)

        if isinstance(self.activation_function, ReLUReplaceActivations):
            return self.activation_function(x, dict_inputs=dict_inputs)
        return self.activation_function(x)





class up_block(conv_same_block):
    def __init__(self, in_ch, out_chs, kernel_size, e_normalization_type=E_Normalization_Type.group_norm, e_conv_layer=EConvLayer.conv2d,
                 use_spectral_norm=False, n_resnet_blocks=0, e_activation_type=EActivationType.relu, conv_n_groups=1,
                 resnet_block_class=conv_same_2_resnet_normal_block, **kwargs):
        super().__init__(in_ch, out_chs, kernel_size, e_normalization_type, e_conv_layer=e_conv_layer, use_spectral_norm=use_spectral_norm,
                    e_activation_type=e_activation_type, conv_n_groups=conv_n_groups)
        self.identity_cat_for_hook = torch.nn.Identity()

        self.n_resnet_blocks = n_resnet_blocks

        self.resnet_blocks = nn.Sequential()

        for i in range(n_resnet_blocks):
            resnet_block = resnet_block_class(out_chs[0], kernel_size, e_normalization_type, use_spectral_norm=use_spectral_norm,
                                e_conv_layer=e_conv_layer, e_activation_type=e_activation_type, conv_n_groups=conv_n_groups)
            self.resnet_blocks.add_module("resnet_block " + str(i), resnet_block)


    def forward(self, x, skip=None, dict_inputs=None):
        #print("dict inputs: {}".format(dict_inputs))
        #print("up block x input shape: {}".format(x.shape))
        if skip is not None:
            #print("skip not None")
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            x = self.identity_cat_for_hook(torch.cat([x, skip], dim=1))

            x = super().forward(x, dict_inputs=dict_inputs)
        else:
            #print("skip None")
            #print("up block x shape before: {}".format(x.shape))
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            #print("up block x shape after: {}".format(x.shape))
            x = super().forward(x, dict_inputs=dict_inputs)
            #print("up block x shape super: {}".format(x.shape))

        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, dict_inputs=dict_inputs)

        #print("up block x output shape: {}".format(x.shape))
        return x


class up_block_pass_skip(conv_same_block):
    def __init__(self, in_ch, out_chs, kernel_size, n_channels_in_skip, e_normalization_type=E_Normalization_Type.group_norm, e_conv_layer=EConvLayer.conv2d,
                 use_spectral_norm=False, n_resnet_blocks=0, e_activation_type=EActivationType.relu, conv_n_groups=1, resnet_block_class=conv_same_2_resnet_normal_block):
        super().__init__(in_ch, out_chs, kernel_size, e_normalization_type, e_conv_layer=e_conv_layer, use_spectral_norm=use_spectral_norm,
                    e_activation_type=e_activation_type, conv_n_groups=conv_n_groups)
        self.identity_cat_for_hook = torch.nn.Identity()

        self.n_resnet_blocks = n_resnet_blocks

        self.resnet_blocks = nn.Sequential()

        for i in range(n_resnet_blocks):
            resnet_block = resnet_block_class(out_chs[0], kernel_size, e_normalization_type, use_spectral_norm=use_spectral_norm,
                                e_conv_layer=e_conv_layer, e_activation_type=e_activation_type, conv_n_groups=conv_n_groups,
                                n_ch_skip=n_channels_in_skip)
            self.resnet_blocks.add_module("resnet_block " + str(i), resnet_block)


    def forward(self, x, skip=None, dict_inputs=None):
        #print("dict inputs: {}".format(dict_inputs))
        #print("up block x input shape: {}".format(x.shape))
        if skip is not None:
            #print("skip not None")
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            x = self.identity_cat_for_hook(torch.cat([x, skip], dim=1))

            x = super().forward(x, dict_inputs=dict_inputs)
        else:
            #print("skip None")
            #print("up block x shape before: {}".format(x.shape))
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            #print("up block x shape after: {}".format(x.shape))
            x = super().forward(x, dict_inputs=dict_inputs)
            #print("up block x shape super: {}".format(x.shape))

        for resnet_block in self.resnet_blocks:
            x = resnet_block([x, skip], dict_inputs=dict_inputs)

        #print("up block x output shape: {}".format(x.shape))
        return x






class decoder(nn.Module):
    def __init__(self, kernel_size, n_channels_in, n_channels_in_skip, n_channels_out,
                 e_normalization_type=E_Normalization_Type.group_norm,
                 e_conv_layer=EConvLayer.conv2d,
                 use_spectral_norm=False, n_resnet_blocks=0,
                 resnet_block_class=conv_same_2_resnet_normal_block, e_activation_type=EActivationType.relu,
                 conv_n_groups=1, upsample_block_class=up_block):

        super().__init__()

        print("decoder n_channels_in: {}".format(n_channels_in))
        print("decoder n_channels_out: {}".format(n_channels_out))

        self.up_blocks = nn.Sequential()

        for i in range(len(n_channels_in)):
            n_ch_in = n_channels_in[i]
            n_ch_out = n_channels_out[i]
            if isinstance(n_resnet_blocks, list):
                n_res_blocks = n_resnet_blocks[i]
            else:
                n_res_blocks = n_resnet_blocks

            if isinstance(conv_n_groups, list):
                conv_n_groups_ = conv_n_groups[i]
            else:
                conv_n_groups_ = conv_n_groups

            if isinstance(e_normalization_type, list):
                e_normalization_type_ = e_normalization_type[i]
            else:
                e_normalization_type_ = e_normalization_type

            if isinstance(e_conv_layer, list):
                e_conv_layer_ = e_conv_layer[i]
            else:
                e_conv_layer_ = e_conv_layer

            if isinstance(e_activation_type, list):
                e_activation_type_ = e_activation_type[i]
            else:
                e_activation_type_ = e_activation_type

            if i < len(n_channels_in_skip):
                n_ch_skip = n_channels_in_skip[i]

                upsample_block = upsample_block_class(in_ch=n_ch_in, out_chs=[n_ch_out], kernel_size=kernel_size,
                                        e_normalization_type=e_normalization_type_, e_conv_layer=e_conv_layer_,
                                        use_spectral_norm=use_spectral_norm, n_resnet_blocks=n_res_blocks,
                                        e_activation_type=e_activation_type_, conv_n_groups=conv_n_groups_,
                                        resnet_block_class=resnet_block_class, n_channels_in_skip=n_ch_skip)
            else:
                upsample_block = up_block(in_ch=n_ch_in, out_chs=[n_ch_out], kernel_size=kernel_size,
                                        e_normalization_type=e_normalization_type_, e_conv_layer=e_conv_layer_,
                                        use_spectral_norm=use_spectral_norm, n_resnet_blocks=n_res_blocks,
                                        e_activation_type=e_activation_type_, conv_n_groups=conv_n_groups_,
                                        resnet_block_class=conv_same_2_resnet_normal_block)



            self.up_blocks.add_module("up block " + str(i), upsample_block)


    def forward(self, skips, dict_inputs=None):
        all_blocks_out = []
        x = skips[-1]

        for i, upsample_block in enumerate(self.up_blocks):
            if i > len(skips)-2:
                #x = checkpoint(upsample_block, x, None, dict_inputs)
                x = upsample_block(x, dict_inputs=dict_inputs)
            else:
                #x = checkpoint(upsample_block, x, skips[-(i+2)], dict_inputs)
                x = upsample_block(x, skips[-(i+2)], dict_inputs=dict_inputs)
            all_blocks_out.append(x)

        return all_blocks_out

    """
    def forward(self, skips, dict_inputs=None):
        all_blocks_out = []

        #for skip in skips:
        #    print("skip.shape: {}".format(skip.shape))

        x = skips[-1]

        for i, upsample_block in enumerate(self.up_blocks):
            #if i == 0:
            #    x = upsample_block(x)
            #else:
            if i != len(self.up_blocks)-1:
                x = upsample_block(x, skips[-(i+2)], dict_inputs=dict_inputs)
                #print("x.shape out decoder: {}".format(x.shape))
            else:
                x = upsample_block(x, dict_inputs=dict_inputs)
            all_blocks_out.append(x)

        return all_blocks_out
    """



class ResnetEncoderUnetDecoder(nn.Module):
    def __init__(self, kernel_size_full_res_conv, kernel_size_encoder, kernel_size_decoder, n_channels_input,
                    n_channels_out_encoder, n_channels_out_decoder,
                    n_resnet_blocks_encoder, e_normalization_type_encoder=E_Normalization_Type.group_norm, e_normalization_type_decoder=E_Normalization_Type.group_norm,
                    n_ch_out=1, bias_initialization=False, n_channel_full_res=None, n_resnet_blocks_full_res=0, use_head_block=True,
                    e_conv_layer=EConvLayer.conv2d, e_conv_layer_encoder=None, e_conv_layer_decoder=None, e_conv_layer_head=None, apply_tanh=True, use_spectral_norm=False, n_resnet_blocks_decoder=0, decoder_class=decoder,
                    n_channels_in_decoder=None, e_activation_type=EActivationType.relu, e_activation_type_decoder=EActivationType.relu, conv_n_groups_encoder=1, conv_n_groups_decoder=1,
                    resnet_block_class=conv_same_2_resnet_normal_block, upsample_block_class=up_block):

        super().__init__()

        if e_conv_layer_encoder == None:
            e_conv_layer_encoder = e_conv_layer
        if e_conv_layer_decoder == None:
            e_conv_layer_decoder = e_conv_layer

        n_channels_in_skip = []
        n_channels_in_encoder = [n_channels_input]
        for i in range(len(n_channels_out_encoder)-1):
            n_ch = n_channels_out_encoder[i]
            n_channels_in_encoder.append(n_ch)
            n_channels_in_skip.insert(0, n_ch)

        if n_channels_in_decoder == None:
            n_channels_in_decoder = [n_channels_out_encoder[-1] + n_channels_out_encoder[-2]]
            for i in range(len(n_channels_out_encoder)-2):
                #if i == 0:
                #    n_channels_in_decoder.append(n_channels_out_encoder[-1] + n_channels_out_encoder[-2])
                #else:
                n_channels = n_channels_out_decoder[i]+n_channels_out_encoder[-(i+3)]
                n_channels_in_decoder.append(n_channels)
            for j in range(len(n_channels_out_decoder) -i-2):
                #print(j)
                n_channels_in_decoder.append(n_channels_out_decoder[j+i+1])

        print("n channels in encoder: {}".format(n_channels_in_encoder))
        print("n channels in decoder: {}".format(n_channels_in_decoder))
        print("n_channels_out_encoder: {}".format(n_channels_out_encoder))
        print("n channels out decoder: {}".format(n_channels_out_decoder))

        """
        if n_channel_full_res is not None:
            n_channels_in_encoder = [n_channel_full_res]
        else:
            n_channels_in_encoder = [n_channels_input]
        for i in range(len(n_channels_out_encoder)-1):
            n_ch = n_channels_out_encoder[i]
            n_channels_in_encoder.append(n_ch)

        n_channels_in_decoder = []
        for i in range(len(n_channels_out_encoder)-1):
            #if i == 0:
            #    n_channels_in_decoder.append(n_channels_out_encoder[-1])
            #else:
            n_channels = n_channels_out_decoder[i]+n_channels_out_encoder[-(i+1)]
            n_channels_in_decoder.append(n_channels)
        n_channels_in_decoder.append(n_channels_out_decoder[-2])
            #n_channels_in_dec = n_channels_out_decoder[i] + n_channels_out_encoder[-(i+2)]
            #n_channels_in_decoder.append(n_channels_in_dec)
        #n_channels_in_decoder.append(n_channels_out_decoder[-2]+n_channel_full_res)
        #n_channels_in_decoder.insert(0, n_channels_out_encoder[-1])

        #print("n_channel in encoder: {}".format(n_channels_in_encoder))
        #print("n channel out encoder: {}".format(n_channels_out_encoder))
        #print("n channel in decoder: {}".format(n_channels_in_decoder))
        #print("n channel out decoder: {}".format(n_channels_out_decoder))
        """


        self.encoder = encoder(kernel_size=kernel_size_encoder, n_channels_in=n_channels_in_encoder, n_channels_out=n_channels_out_encoder,
                            n_resnet_blocks=n_resnet_blocks_encoder, e_normalization_type=e_normalization_type_encoder, kernel_size_full_res_conv=kernel_size_full_res_conv,
                            n_resnet_blocks_full_res=n_resnet_blocks_full_res, n_channels_input=n_channels_input, n_channel_full_res=n_channel_full_res,
                            e_conv_layer=e_conv_layer_encoder, use_spectral_norm=use_spectral_norm, e_activation_type=e_activation_type,
                            conv_n_groups=conv_n_groups_encoder, resnet_block_class=resnet_block_class)

        self.decoder = decoder_class(kernel_size=kernel_size_decoder, n_channels_in=n_channels_in_decoder, n_channels_out=n_channels_out_decoder,
                            e_normalization_type=e_normalization_type_decoder, e_conv_layer=e_conv_layer_decoder, use_spectral_norm=use_spectral_norm,
                            n_resnet_blocks=n_resnet_blocks_decoder, e_activation_type=e_activation_type_decoder,
                            conv_n_groups=conv_n_groups_decoder, resnet_block_class=resnet_block_class, n_channels_in_skip=n_channels_in_skip,
                            upsample_block_class=upsample_block_class)

        if use_head_block:
            if e_conv_layer_head is None:
                e_conv_layer_head = e_conv_layer
            self.head_block = head_block(n_ch_in=n_channels_out_decoder[-1], n_ch_out=n_ch_out, kernel_size=4, bias_initialization=bias_initialization,
                                    e_conv_layer=e_conv_layer_head, use_spectral_norm=use_spectral_norm)

        self.apply_tanh = apply_tanh

        #self.apply(init_weights)
        self.tanh = torch.nn.Tanh()



    def forward(self, x, dict_inputs=None):
        #print("forward resnet encoder unet decoder")
        #print("dict inputs: {}".format(dict_inputs))
        #raise RuntimeError
        #x = self.full_res_blocks(x)
        x = self.decoder(self.encoder(x, output_all_blocks=True, dict_inputs=dict_inputs), dict_inputs=dict_inputs)[-1]
        if hasattr(self, "head_block"):
            x = self.head_block(x, dict_inputs=dict_inputs)
            if self.apply_tanh:
                return self.tanh(x)#.type(torch.cuda.FloatTensor)
            return x
        else:
            if self.apply_tanh:
                return self.tanh(x)#.type(torch.cuda.FloatTensor)
            else:
                return x
        #return {"pred" : self.head_block(x).type(torch.cuda.FloatTensor)}




class StainModel(ResnetEncoderUnetDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)






class GeneratorResnetEncoderUnetDecoder(ResnetEncoderUnetDecoder):
    def forward(self, x, skips_segmentation_net, dict_inputs=None):
        #print("dict inputs: {}".format(dict_inputs))
        #raise RuntimeError
        #x = self.full_res_blocks(x)
        x = self.decoder(self.encoder(x, output_all_blocks=True, dict_inputs=dict_inputs), skips_segmentation_net, dict_inputs=dict_inputs)[-1]
        if hasattr(self, "head_block"):
            x = self.head_block(x, dict_inputs=dict_inputs)
            if self.apply_tanh:
                return self.tanh(x).type(torch.cuda.FloatTensor)
            return x
        else:
            if self.apply_tanh:
                return self.tanh(x).type(torch.cuda.FloatTensor)
            else:
                return x


class EncoderSegmentationHead(nn.Module):
    def __init__(self, kernel_size, n_channels_in, n_channels_out,
                 n_resnet_blocks, kernel_size_full_res_conv, n_channel_full_res,
                 n_resnet_blocks_full_res, n_channels_input,
                 n_output_channel_head,
                 e_normalization_type=E_Normalization_Type.group_norm,
                 e_downsampling_type=EDownsamplingType.convolution):
        super().__init__()

        self.encoder = encoder(kernel_size=kernel_size, n_channels_in=n_channels_in, n_channels_out=n_channels_out,
                 n_resnet_blocks=n_resnet_blocks, kernel_size_full_res_conv=kernel_size_full_res_conv, n_channel_full_res=n_channel_full_res,
                 n_resnet_blocks_full_res=n_resnet_blocks_full_res, n_channels_input=n_channels_input,
                 e_normalization_type=e_normalization_type,
                 e_downsampling_type=e_downsampling_type)

        self.head_block = head_block(n_ch_in=n_channels_out[-1], n_ch_out=n_output_channel_head, kernel_size=1, bias_initialization=False)


    def forward(self, x):
        x = self.encoder(x, output_all_blocks=False)
        x = self.head_block(x)
        return x


class encoder(nn.Module):
    def __init__(self, kernel_size, n_channels_in, n_channels_out,
                 n_resnet_blocks,
                 use_minibatch_mean_std=False,
                 e_normalization_type=E_Normalization_Type.batch_norm,
                 e_downsampling_type=EDownsamplingType.convolution,
                 kernel_size_full_res_conv=0,
                 n_channel_full_res=None,
                 n_resnet_blocks_full_res=0,
                 n_channels_input=None,
                 use_spectral_norm=False,
                 e_activation_type=EActivationType.relu,
                 e_conv_layer=EConvLayer.conv2d,
                 resnet_block_class=conv_same_2_resnet_normal_block,
                 conv_n_groups=1):

        super().__init__()

        self.down_blocks = nn.Sequential()

        for i in range(len(n_channels_in)):
            n_ch_in = n_channels_in[i]
            n_ch_out = n_channels_out[i]
            if isinstance(n_resnet_blocks, list):
                n_res_blocks = n_resnet_blocks[i]
            else:
                n_res_blocks = n_resnet_blocks

            if isinstance(kernel_size, list):
                kernel_size_to_use = kernel_size[i]
            else:
                kernel_size_to_use = kernel_size
            if isinstance(e_normalization_type, list):
                e_normalization_type_to_use = e_normalization_type[i]
            else:
                e_normalization_type_to_use = e_normalization_type

            if isinstance(use_minibatch_mean_std, list):
                minibatch_mean_std = use_minibatch_mean_std[i]
            else:
                minibatch_mean_std = use_minibatch_mean_std

            if isinstance(conv_n_groups, list):
                conv_n_groups_ = conv_n_groups[i]
            else:
                conv_n_groups_ = conv_n_groups

            if isinstance(e_conv_layer, list):
                e_conv_layer_ = e_conv_layer[i]
            else:
                e_conv_layer_ = e_conv_layer

            if isinstance(e_downsampling_type, list):
                e_downsampling_type_ = e_downsampling_type[i]
            else:
                e_downsampling_type_ = e_downsampling_type






            downsample_block = down_block(in_ch=n_ch_in, out_ch=n_ch_out, kernel_size=kernel_size_to_use,
                                    e_normalization_type=e_normalization_type_to_use, n_resnet_blocks=n_res_blocks,
                                    e_downsampling_type=e_downsampling_type_, use_spectral_norm=use_spectral_norm,
                                    e_activation_type=e_activation_type, e_conv_layer=e_conv_layer_, use_minibatch_mean_std=minibatch_mean_std,
                                    conv_n_groups=conv_n_groups_, resnet_block_class=resnet_block_class)

            self.down_blocks.add_module("down block " + str(i), downsample_block)

        if n_resnet_blocks_full_res > 0:
            self.full_res_blocks = SingleConv_PlusResnetBlocks(n_ch_in=n_channels_input, n_ch_out=n_channel_full_res,
                                        kernel_size_single_conv=kernel_size_full_res_conv, kernel_size_resnet_blocks=kernel_size,
                                        n_resnet_blocks=n_resnet_blocks_full_res, e_normalization_type=e_normalization_type,
                                        e_conv_layer=e_conv_layer_)


    def forward(self, x, output_all_blocks=False, dict_inputs=None):
        all_blocks_out = []

        if hasattr(self, "full_res_blocks"):
            x = self.full_res_blocks(x, dict_inputs=dict_inputs)
            all_blocks_out.append(x)

        for downsample_block in self.down_blocks:
            #x = deepspeed.checkpointing.checkpoint(downsample_block, x, dict_inputs)
            #x = checkpoint(downsample_block, x, dict_inputs)
            x = downsample_block(x, dict_inputs=dict_inputs)
            #print("down x shape: {}".format(x.shape))
            if output_all_blocks:
                all_blocks_out.append(x)
        if output_all_blocks:
            return all_blocks_out
        else:
            return x













class decoder_no_skips(nn.Module):
    def __init__(self, kernel_size, n_channels_in, n_channels_out,
                 e_normalization_type=E_Normalization_Type.group_norm,
                 e_conv_layer=EConvLayer.conv2d,
                 use_spectral_norm=False, n_resnet_blocks=0,
                 e_activation_type=EActivationType.relu):

        super().__init__()

        self.up_blocks = nn.Sequential()

        for i in range(len(n_channels_in)):
            n_ch_in = n_channels_in[i]
            n_ch_out = n_channels_out[i]
            if isinstance(n_resnet_blocks, list):
                n_res_blocks = n_resnet_blocks[i]
            else:
                n_res_blocks = n_resnet_blocks

            upsample_block = up_block(in_ch=n_ch_in, out_chs=[n_ch_out], kernel_size=kernel_size,
                                    e_normalization_type=e_normalization_type, e_conv_layer=e_conv_layer,
                                    use_spectral_norm=use_spectral_norm, n_resnet_blocks=n_res_blocks,
                                    e_activation_type=e_activation_type)

            self.up_blocks.add_module("up block " + str(i), upsample_block)


    def forward(self, x):
        all_blocks_out = []

        for i, upsample_block in enumerate(self.up_blocks):
            #if i != len(self.up_blocks)-1:
            x = upsample_block(x)
            #else:
            #    x = upsample_block(x)
            all_blocks_out.append(x)

        return x




class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, e_normalization_type=E_Normalization_Type.group_norm, n_resnet_blocks=0, apply_activation_function=True,
            e_downsampling_type=EDownsamplingType.convolution, use_spectral_norm=False, e_activation_type=EActivationType.relu, e_conv_layer=EConvLayer.conv2d,
            use_minibatch_mean_std=False, resnet_block_class=None, conv_n_groups=1):
        super().__init__()

        if resnet_block_class == None:
            resnet_block_class = conv_same_2_resnet_normal_block

        self.n_resnet_blocks = n_resnet_blocks

        self.resnet_blocks = nn.Sequential()

        if use_minibatch_mean_std:
            in_ch += 1
            self.minibatch_mean_std_layer = MinibatchMeanStd()
        self.downsample = downsample(in_ch, out_ch, kernel_size, e_normalization_type, apply_activation_function, e_downsampling_type, use_spectral_norm=use_spectral_norm,
                            e_activation_type=e_activation_type, e_conv_layer=e_conv_layer, conv_n_groups=conv_n_groups)

        for i in range(n_resnet_blocks):
            resnet_block = resnet_block_class(out_ch, kernel_size, e_normalization_type, use_spectral_norm=use_spectral_norm, e_activation_type=e_activation_type,
                                e_conv_layer=e_conv_layer, conv_n_groups=conv_n_groups)
            self.resnet_blocks.add_module("resnet_block " + str(i), resnet_block)

    def forward(self, x, dict_inputs=None):
        if hasattr(self, "minibatch_mean_std_layer"):
            x = self.minibatch_mean_std_layer(x)
        x = self.downsample(x, dict_inputs=dict_inputs)

        for i in range(self.n_resnet_blocks):
            x = self.resnet_blocks[i](x, dict_inputs=dict_inputs)

        return x



class downsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, e_normalization_type=E_Normalization_Type.group_norm, apply_activation_function=True,
            e_downsampling_type=EDownsamplingType.convolution, use_spectral_norm=False, e_activation_type=EActivationType.relu,
            e_conv_layer=EConvLayer.conv2d, conv_n_groups=1):
        super().__init__()

        #if e_conv_layer == EConvLayer.conv2d:
        #    raise RuntimeError

        #to_pad = int((kernel_size - 1) / 2)
        to_pad = int((kernel_size - 1) / 2)

        #self.result = nn.Sequential()
        self.e_downsampling_type = e_downsampling_type

        if e_downsampling_type == EDownsamplingType.convolution:
            conv = get_conv_layer(e_conv_layer, in_ch, out_ch, kernel_size, stride=2, bias=False, padding=to_pad, padding_mode=padding_mode, conv_n_groups=conv_n_groups)
            """
            if e_conv_layer == EConvLayer.conv2d:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, bias=False, padding=to_pad, padding_mode=padding_mode)
            elif e_conv_layer == EConvLayer.conv2dsegmap:
                conv = Conv2dSegMap(in_ch+3, out_ch, kernel_size, stride=2, bias=False, padding=to_pad, padding_mode=padding_mode)
            elif e_conv_layer == EConvLayer.conv2dsegmap_noise:
                conv = Conv2dSegMapNoise(in_ch+3, out_ch, kernel_size, stride=2, bias=False, padding=to_pad, padding_mode=padding_mode)
            """

            if use_spectral_norm:
                self.conv = spectral_norm(conv)
            else:
                self.conv = conv
            #self.result.add_module("conv2d", conv)
        elif e_downsampling_type == EDownsamplingType.bilinear or e_downsampling_type == EDownsamplingType.average_pooling or e_downsampling_type == EDownsamplingType.bilinear_conv_first:
            conv = get_conv_layer(e_conv_layer, in_ch, out_ch, kernel_size, stride=1, bias=False, padding=to_pad, padding_mode=padding_mode, conv_n_groups=conv_n_groups)
            """
            if e_conv_layer == EConvLayer.conv2d:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, bias=False, padding=to_pad, padding_mode=padding_mode)
            elif e_conv_layer == EConvLayer.conv2dsegmap:
                conv = Conv2dSegMap(in_ch+3, out_ch, kernel_size, stride=1, bias=False, padding=to_pad, padding_mode=padding_mode)
            """
            if use_spectral_norm:
                self.conv = spectral_norm(conv)
            else:
                #conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, bias=False, padding=to_pad, padding_mode=padding_mode)
                self.conv = conv

            #self.result.add_module("conv2d", conv)

        self.normalization = get_normalization_layer(e_normalization_type, out_ch)
        #self.result = add_normalization_layer(self.result, e_normalization_type, out_ch)
        """
        if e_normalization_type == E_Normalization_Type.batch_norm:
            self.result.add_module("batch_norm", nn.BatchNorm2d(out_ch, eps=0.001))
        elif e_normalization_type == E_Normalization_Type.group_norm:
            self.result.add_module("group_norm", nn.GroupNorm(32, out_ch))
        elif e_normalization_type == E_Normalization_Type.instance_norm:
            self.result.add_module("instance_norm", nn.InstanceNorm2d(out_ch, affine=True))
        elif e_normalization_type == E_Normalization_Type.spade:
            self.result.add_module("spade_norm", SPADE(out_ch))
        """
        if apply_activation_function:
            self.activation_function = get_activation_layer(e_activation_type)
        else:
            self.activation_function = nn.Identity()
        """
            if e_activation_type == EActivationType.relu:
                self.result.add_module("relu", nn.ReLU(inplace=relu_inplace))
            elif e_activation_type == EActivationType.leaky_relu:
                self.result.add_module("leaky_relu", nn.LeakyReLU(0.2, inplace=relu_inplace))
        """

    def forward(self, x, dict_inputs=None):
        if self.e_downsampling_type == EDownsamplingType.bilinear:
            x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear')
        elif self.e_downsampling_type == EDownsamplingType.average_pooling:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        if dict_inputs is not None:
            #print("dict inputs is not None")
            x = self.conv(x, dict_inputs=dict_inputs)
            if self.e_downsampling_type == EDownsamplingType.bilinear_conv_first:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear')
            if isinstance(self.normalization, AdaptiveInstanceNorm):
                x = self.normalization(x, dict_inputs=dict_inputs)
            else:
                #print("x shape: {}".format(x.shape))
                x = self.normalization(x)
        else:
            #print("len x: {}".format(len(x)))
            x = self.conv(x)
            x = self.normalization(x)

        if isinstance(self.activation_function, ReLUReplaceActivations):
            return self.activation_function(x, dict_inputs=dict_inputs)
        return self.activation_function(x)



class conv_downsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, e_normalization_type=E_Normalization_Type.group_norm, apply_activation_function=True,
        e_conv_layer=EConvLayer.conv2d):
        super().__init__()

        to_pad = int((kernel_size - 1) / 2)

        self.result = nn.Sequential()

        self.conv = get_conv_layer(e_conv_layer, in_ch, out_ch, kernel_size, stride=2, bias=False, padding=to_pad, padding_mode="zeros")
        """
        if e_conv_layer == EConvLayer.conv2d:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, bias=False, padding=to_pad)
        elif e_conv_layer == EConvLayer.conv2dsegmap:
            self.conv = Conv2dSegMap(in_ch+3, out_ch, kernel_size, stride=2, bias=False, padding=to_pad)
        """
        #self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, bias=False, padding=to_pad)
        #self.result.add_module("conv2d", conv)

        #self.result = add_normalization_layer(self.result, e_normalization_type, out_ch)
        self.normalization = get_normalization_layer(e_normalization_type, out_ch)

        """
        if e_normalization_type == E_Normalization_Type.batch_norm:
            self.result.add_module("batch_norm", nn.BatchNorm2d(out_ch, eps=0.001))
        elif e_normalization_type == E_Normalization_Type.group_norm:
            self.result.add_module("group_norm", nn.GroupNorm(32, out_ch))
        elif e_normalization_type == E_Normalization_Type.spade:
            self.result.add_module("spade_norm", SPADE(out_ch))
        """
        if apply_activation_function:
            self.result.add_module("relu", nn.ReLU(inplace=relu_inplace))


    def forward(self, x, dict_inputs=None):
        if dict_inputs is not None:
            x = self.conv(x, dict_inputs=dict_inputs)
            if isinstance(self.normalization, AdaptiveInstanceNorm):
                x = self.normalization(x, dict_inputs=dict_inputs)
            else:
                x = self.normalization(x)
        else:
            x = self.conv(x)
            x = self.normalization(x)
        return self.result(x)


#class MinibatchDiscrimination(nn.Module):






class conv2d_transpose_up_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, e_normalization_type=E_Normalization_Type.group_norm):
        super().__init__()

        self.result = nn.Sequential()
        conv2dtranspose = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2, bias=False)
        self.result.add_module("conv_transpose", conv2dtranspose)

        #self.result = add_normalization_layer(self.result, e_normalization_type, out_ch)
        self.normalization = get_normalization_layer(e_normalization_type, out_ch)

        """

        if e_normalization_type == E_Normalization_Type.batch_norm:
            self.result.add_module("batch_norm", nn.BatchNorm2d(out_ch, eps = 0.001))
        elif e_normalization_type == E_Normalization_Type.group_norm:
            self.result.add_module("group_norm", nn.GroupNorm(32, out_ch))
        else:
            self.result.add_module("batch_norm", nn.BatchNorm2d(out_ch))
        """
        self.result.add_module("relu_1", nn.ReLU(inplace = True))


    def forward(self, x, skip=None, dict_inputs=None):
        #print("conv transpose x shape: {}".format(x.shape))
        if skip is not None:
            x = self.normalization(x, dict_inputs=dict_inputs)
            x = self.result(x)
            #print("x upscaled shape: {}".format(x.shape))
            #print("skip shape: {}".format(skip.shape))
            #x = torch.cat([x, skip], dim=1)
            x = concatenate(x, skip)

            return x
        else:
            x = self.normalization(x, dict_inputs=dict_inputs)
            #print("WARNING: No skip")
            x = self.result(x)
            return x

"""
class conv_same_2_block(nn.Module):
    def __init__(self, n_ch, kernel_size, e_normalization_type=E_Normalization_Type.group_norm, use_spectral_norm=False, e_activation_type=EActivationType.relu,
        e_conv_layer=EConvLayer.conv2d):

        self.block = nn.Sequential()

        conv = get_conv_layer(e_conv_layer, n_ch, n_ch, kernel_size, stride=1, bias=False, padding="same", padding_mode=padding_mode)
        conv = spectral_norm(conv)
        self.block.add_module("conv_1", conv)
        normalization_ = get_normalization_layer(e_normalization_type, n_ch
        self.block.add_module("normalization_1", normalization_)
        if e_activation_type == EActivationType.relu:
            self.block.add_module("relu_1", nn.ReLU(inplace=True))
        elif e_activation_type == EActivationType.leaky_relu:
            self.block.add_module("relu_1", nn.LeakyReLU(0.2, inplace=True))


    def forward(self, x, dict_inputs=None):
        return self.block(x)
"""




class head_block(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, kernel_size, bias_initialization=False, e_conv_layer=EConvLayer.conv2d, use_spectral_norm=False, conv_n_groups=1):
        super().__init__()

        if e_conv_layer == EConvLayer.conv2dsegmap_noise:
            e_conv_layer = EConvLayer.conv2dsegmap
        self.conv = get_conv_layer(e_conv_layer, n_ch_in, n_ch_out, kernel_size, stride=1, bias=True, padding="same", padding_mode=padding_mode, conv_n_groups=conv_n_groups)

        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)

        if bias_initialization:
            new_bias = torch.ones(3)*(1.2)
            self.conv.bias.data[0] = new_bias

    def forward(self, x, dict_inputs=None):
        if dict_inputs is not None:
            return self.conv(x, dict_inputs=dict_inputs)
        return self.conv(x)#-2.0




class MinibatchStd(nn.Module):
    def forward(self, x):
        process_group = torch.distributed.group.WORLD
        world_size = torch.distributed.get_world_size(process_group)
        need_sync = world_size > 1


        #new_feature_map = mean_batch_std*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).to(x.device)

        if need_sync:
            if process_group._get_backend_name() == 'nccl':
                # world_size * (2C + 1)
                combined_size = x.shape[0]*world_size*x.shape[1]*x.shape[2]*x.shape[3]
                combined_x = torch.empty((x.shape[0]*world_size, x.shape[1], x.shape[2], x.shape[3]),
                                            dtype=x.dtype,
                                            device=x.device)
                dist._all_gather_base(combined_x, x, process_group, async_op=False)
                #print("combined flat shape: {}".format(combined_x.shape))
                mean_batch_std = torch.mean(torch.std(combined_x, dim=0, unbiased=False))
                #combined = torch.reshape(combined_flat, (world_size, combined_size))
                # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
                #mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
            else:
                raise RuntimeError
                # world_size * (2C + 1)
                combined_list = [
                    torch.empty_like(combined) for _ in range(world_size)
                ]
                dist.all_gather(combined_list, combined, process_group, async_op=False)
                combined = torch.stack(combined_list, dim=0)
                # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
                #mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)


        else:
            mean_batch_std = torch.mean(torch.std(x, dim=0, unbiased=False))
        new_feature_map = mean_batch_std*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).to(x.device)
        return torch.cat([x, new_feature_map], dim=1)



class MinibatchMeanStd(nn.Module):
    def forward(self, x):
        process_group = torch.distributed.group.WORLD
        world_size = torch.distributed.get_world_size(process_group)
        need_sync = world_size > 1

        mean_activation_vec = torch.mean(x, dim=[2, 3])
        #new_feature_map = mean_batch_std*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).to(x.device)

        if need_sync:
            if process_group._get_backend_name() == 'nccl':
                # world_size * (2C + 1)
                combined_x = torch.empty((x.shape[0]*world_size, x.shape[1]),
                                            dtype=x.dtype,
                                            device=x.device)
                dist._all_gather_base(combined_x, mean_activation_vec, process_group, async_op=False)
                #print("combined flat shape: {}".format(combined_x.shape))
                mean_batch_std = torch.mean(torch.std(combined_x, dim=0, unbiased=False))
                #combined = torch.reshape(combined_flat, (world_size, combined_size))
                # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
                #mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
            else:
                raise RuntimeError
                # world_size * (2C + 1)
                combined_list = [
                    torch.empty_like(combined) for _ in range(world_size)
                ]
                dist.all_gather(combined_list, combined, process_group, async_op=False)
                combined = torch.stack(combined_list, dim=0)
                # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
                #mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)


        else:
            mean_batch_std = torch.mean(torch.std(x, dim=0, unbiased=False))
        new_feature_map = mean_batch_std*torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).to(x.device)
        return torch.cat([x, new_feature_map], dim=1)




class SingleConv_PlusResnetBlocks(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, kernel_size_single_conv, kernel_size_resnet_blocks, n_resnet_blocks=0,
                 e_normalization_type=E_Normalization_Type.group_norm, e_conv_layer=EConvLayer.conv2d):
        super().__init__()

        self.n_resnet_blocks = n_resnet_blocks

        self.conv = nn.Conv2d(n_ch_in, n_ch_out, kernel_size_single_conv, stride=1, bias=False, padding="same")
        #nn.init.normal_(self.conv.weight, 0, 0.02)

        self.conv_same_blocks = nn.Sequential()

        for k in range(self.n_resnet_blocks):
            self.conv_same_blocks.add_module("2_normal_resnet_block_first_" + "_" + str(k), conv_same_2_resnet_normal_block(n_ch_out, kernel_size_resnet_blocks, e_normalization_type=e_normalization_type))


    def forward(self, x, dict_inputs=None):
        if dict_inputs is not None:
            x = self.conv(x, dict_inputs=dict_inputs)
        else:
            x = self.conv(x)

        for k in range(self.n_resnet_blocks):
            y = self.conv_same_blocks[k](x, dict_inputs=dict_inputs)
            x = x.add(y)
            x = torch.nn.functional.relu_(x)

        return x







def concatenate(x, skip):
    x = torchvision.transforms.functional.center_crop(x, [skip.shape[2], skip.shape[3]])
    return torch.cat([x, skip], dim=1)




class AdaptiveInstanceNorm(nn.InstanceNorm2d):
    def __init__(self, num_channels, dimension_style_vec=512, **kwargs):
        super().__init__(num_channels, **kwargs)

        self.W_mean = nn.Linear(in_features=dimension_style_vec, out_features=num_channels)
        self.W_std = nn.Linear(in_features=dimension_style_vec, out_features=num_channels)


    def forward(self, x, dict_inputs):
        x = super().forward(x)

        style_vector = dict_inputs["style_vector"]
        new_std = self.W_std(style_vector)
        new_mean = self.W_mean(style_vector)

        x_renorm = x*new_std.unsqueeze(-1).unsqueeze(-1) + new_mean.unsqueeze(-1).unsqueeze(-1)

        return x_renorm


class BatchnormReplaceActivations(nn.BatchNorm2d):
    def forward(self, x, dict_inputs=None):
        print("forward batchnorm replace activations")
        #print("dict inputs: {}".format(dict_inputs))
        if dict_inputs is not None:
            #print("dict inputs is not none")
            if "activation_replace" in dict_inputs.keys():
                #print("has activation replace in keys")
                #raise RuntimeError
                #print("activation replace shape: {}".format(dict_inputs["activation_replace"].shape))
                #print("original activation shape: {}".format(x.shape))
                return dict_inputs["activation_replace"]
        #print("dict inputs are none!!!!")
        return super().forward(x)



class ReLUReplaceActivations(nn.ReLU):
    def forward(self, x, dict_inputs=None):
        #print("forward batchnorm replace activations")
        if dict_inputs is not None:
            if "activation_replace" in dict_inputs.keys():
                #print("relu replace activations")
                return super().forward(dict_inputs["activation_replace"])
        return super().forward(x)

#relu_replace_activation


class Conv2dNormal(nn.Conv2d):
    def forward(self, x, dict_inputs=None):
        return super().forward(x)



class Conv2dSegMap(nn.Conv2d):
    #def __init__(self, **kwargs):
    #    #kwargs["in_channels"] = kwargs["in_channels"]+3
    #    super().__init__(**kwargs)

    def forward(self, x, dict_inputs=None):
        #print("dict inputs: {}".format(dict_inputs))
        if dict_inputs is not None:
            if "segmap" in dict_inputs.keys():
                segmap = F.interpolate(dict_inputs["segmap"], size=x.size()[2:], mode='nearest')
                x = torch.cat([x, segmap], dim=1)
                return super().forward(x)
        return super().forward(x)


class Conv2dSegMapNoise(Conv2dSegMap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = torch.nn.Parameter(torch.randn(kwargs["out_channels"]))


    def forward(self, x, dict_inputs=None):
        x = super().forward(x, dict_inputs=dict_inputs)

        b, c, h, w = x.shape
        noise = torch.randn_like(x)

        return x+noise*self.W.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)


class Conv2dAddInput(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = torch.nn.Parameter(torch.zeros(kwargs["out_channels"]))
        self.W_input = torch.nn.Parameter(torch.ones(kwargs["out_channels"]))

    def forward(self, x, dict_inputs=None):
        #print("dict inputs: {}".format(dict_inputs))
        if dict_inputs is not None:
            if "input" in dict_inputs.keys():
                x = super().forward(x)
                return x*self.W.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1) + dict_inputs["input"]*self.W_input.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        raise RuntimeError
        return super().forward(x)



class Conv2dNoise(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = torch.nn.Parameter(0.05*torch.randn(kwargs["in_channels"]))


    def forward(self, x, dict_inputs=None):
        noise = torch.randn_like(x)
        #print("x input shape: {}".format(x.shape))
        #print("mean abs weight: {}".format(torch.mean(torch.abs(self.W))))
        x = x+noise*self.W.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        #print("x output shape: {}".format(x.shape))

        x_super = super().forward(x)
        #print("x_super output shape: {}".format(x_super.shape))

        return x_super


class Conv2dBilinearDownsample(nn.Conv2d):
    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5)
        return super().forward(x)


class Conv2dBilinearDownsample_ConvFirst(nn.Conv2d):
    def forward(self, x):
        return F.interpolate(super().forward(x), scale_factor=0.5, mode="bilinear")



class Conv2dReplaceActivations(nn.Conv2d):
    def forward(self, x, dict_inputs=None):
        #print("dict inputs: {}".format(dict_inputs))
        if dict_inputs is not None:
            if "activation_replace" in dict_inputs.keys():
                return dict_inputs["activation_replace"]
        return super().forward(x)



"""
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, num_channels, dimension_style_vec, **kwargs):
        super().__init__(**kwargs)

        self.W_mean = nn.Linear(in_features=dimension_style_vec, out_features=num_channels)
        self.W_std = nn.Linear(in_features=dimension_style_vec, out_features=num_channels)


    def forward(self, x, dict_inputs=None):
        mu = torch.mean(x, dim=[0, 2, 3])
        std = torch.std(x, dim=[0, 2, 3])

        x_normalized = (x - mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))/(std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+0.001)

        style_vector = dict_inputs["style_vector"]
        new_std = self.W_std(style_vector)
        new_mean = self.W_mean(style_vector)

        x_renorm = x_normalized*new_std.unsqueeze(-1).unsqueeze(-1) + new_mean.unsqueeze(-1).unsqueeze(-1)

        return x_renorm
"""








#---------------------------- build models ------------------------





class Discriminator(encoder):
    def __init__(self, n_final_output_channel=1, last_conv_kernel_size=4, use_minibatch_std=False, n_ch_last_layer=512, **kwargs):
        super().__init__(**kwargs)

        #n_ch_last_layer = 512
        if use_minibatch_std:
            self.minibatch_std_layer = MinibatchStd()
            n_ch_last_layer += 1

        self.last_conv = spectral_norm(nn.Conv2d(in_channels=n_ch_last_layer, out_channels=n_final_output_channel, kernel_size=last_conv_kernel_size, stride=1, bias=True, padding="same", padding_mode="reflect"))

        #self.last_conv = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=4, stride=1, bias=True, padding="same", padding_mode="reflect")

        #self.last_conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, bias=True, padding="same")

        self.apply(init_weights)

    def forward(self, x, dict_inputs=None):
        #return self.last_conv(x)
        if dict_inputs is not None:
            x = super().forward(x, dict_inputs=dict_inputs)
        else:
            x = super().forward(x)#[-1]
        if hasattr(self, "minibatch_std_layer"):
            x = self.minibatch_std_layer(x)
        x = self.last_conv(x)
        return x


class DiscriminatorSingleValue(encoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.linear_layer = torch.nn.Linear(in_features=512, out_features=1)
        #self.apply(init_weights)


    def forward(self, x, dict_inputs=None):
        x = super().forward(x, dict_inputs)

        x = torch.mean(x, dim=[2, 3])
        return self.linear_layer(x)





class Generator(ResnetEncoderUnetDecoder):
    def forward(self, x):
        x = super().forward(x)["pred"]
        x = F.tanh(x)
        return {"pred" : x.type(torch.cuda.FloatTensor)}


class GeneratorFromSegementationNet(nn.Module):
    def __init__(self, kernel_size, n_channels_in, n_channels_out,
                 e_normalization_type=E_Normalization_Type.group_norm,
                 e_conv_layer=EConvLayer.conv2d,
                 use_spectral_norm=False, n_resnet_blocks=0):

        super().__init__()

        print("decoder n_channels_in: {}".format(n_channels_in))
        print("decoder n_channels_out: {}".format(n_channels_out))

        self.up_blocks = nn.Sequential()

        for i in range(len(n_channels_in)):
            n_ch_in = n_channels_in[i]
            n_ch_out = n_channels_out[i]
            if isinstance(n_resnet_blocks, list):
                n_res_blocks = n_resnet_blocks[i]
            else:
                n_res_blocks = n_resnet_blocks

            upsample_block = up_block(in_ch=n_ch_in, out_chs=[n_ch_out], kernel_size=kernel_size,
                                    e_normalization_type=e_normalization_type, e_conv_layer=e_conv_layer,
                                    use_spectral_norm=use_spectral_norm, n_resnet_blocks=n_res_blocks)

            self.up_blocks.add_module("up block " + str(i), upsample_block)

        self.head_block = head_block(n_ch_in=n_channels_out[-1], n_ch_out=3, kernel_size=4)
        self.tanh = torch.nn.Tanh()


    def forward(self, skips, dict_inputs=None):
        all_blocks_out = []
        x = skips[0]

        for i, upsample_block in enumerate(self.up_blocks):
            if i == 0:
                x = upsample_block(x, dict_inputs=dict_inputs)
            elif i == 1 or i == 2:
                x = upsample_block(x, skips[i], dict_inputs=dict_inputs)
            else:
                #x = checkpoint(upsample_block, x, None, dict_inputs)
                x = upsample_block(x, dict_inputs=dict_inputs)

        x = self.head_block(x)

        return self.tanh(x)




class DecoderGeneratorFromSegmentationNet(nn.Module):
    def __init__(self, kernel_size, n_channels_in, n_channels_out,
                 e_normalization_type=E_Normalization_Type.group_norm,
                 e_conv_layer=EConvLayer.conv2d,
                 use_spectral_norm=False, n_resnet_blocks=0,
                 resnet_block_class=None, e_activation_type=EActivationType.relu,
                 conv_n_groups=1):

        super().__init__()

        print("decoder n_channels_in: {}".format(n_channels_in))
        print("decoder n_channels_out: {}".format(n_channels_out))

        self.up_blocks = nn.Sequential()

        for i in range(len(n_channels_in)):
            n_ch_in = n_channels_in[i]
            n_ch_out = n_channels_out[i]
            if isinstance(n_resnet_blocks, list):
                n_res_blocks = n_resnet_blocks[i]
            else:
                n_res_blocks = n_resnet_blocks

            if isinstance(conv_n_groups, list):
                conv_n_groups_ = conv_n_groups[i]
            else:
                conv_n_groups_ = conv_n_groups

            if isinstance(e_normalization_type, list):
                e_normalization_type_ = e_normalization_type[i]
            else:
                e_normalization_type_ = e_normalization_type

            if isinstance(e_conv_layer, list):
                e_conv_layer_ = e_conv_layer[i]
            else:
                e_conv_layer_ = e_conv_layer

            if isinstance(e_activation_type, list):
                e_activation_type_ = e_activation_type[i]
            else:
                e_activation_type_ = e_activation_type

            upsample_block = up_block(in_ch=n_ch_in, out_chs=[n_ch_out], kernel_size=kernel_size,
                                    e_normalization_type=e_normalization_type_, e_conv_layer=e_conv_layer_,
                                    use_spectral_norm=use_spectral_norm, n_resnet_blocks=n_res_blocks,
                                    e_activation_type=e_activation_type_, conv_n_groups=conv_n_groups_)

            self.up_blocks.add_module("up block " + str(i), upsample_block)


    def forward(self, skips, skips_segmentation_net, dict_inputs=None):
        all_blocks_out = []
        x = skips[-1]
        #skips_segmentation_net = dict_inputs["skips_segmentation_net"]

        for i, upsample_block in enumerate(self.up_blocks):
            if i > len(skips)-2:
                #x = checkpoint(upsample_block, x, None, dict_inputs)
                x = upsample_block(x)
            else:
                if i == 2:
                    wanted_size = skips_segmentation_net[0].shape[-1]
                    x = upsample_block(x[:, :, :wanted_size, :wanted_size], torch.cat([skips[-(i+2)][:, :, :wanted_size, :wanted_size], skips_segmentation_net[0]], dim=1))
                else:
                    x = upsample_block(x, skips[-(i+2)])
            all_blocks_out.append(x)

        return all_blocks_out









class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder(kernel_size=3, n_channels_in=[3, 32, 64, 128, 256, 512, 512], n_channels_out=[32, 64, 128, 256, 512, 512, 512],
                n_resnet_blocks=0, e_normalization_type=E_Normalization_Type.batch_norm, e_activation_type=EActivationType.leaky_relu)
        self.decoder = decoder_no_skips(kernel_size=3, n_channels_in=[1024, 1024, 512, 256, 128, 64, 32], n_channels_out=[1024, 512, 256, 128, 64, 32, 16],
                            e_normalization_type=E_Normalization_Type.batch_norm, e_activation_type=EActivationType.leaky_relu)

        self.head_block = head_block(n_ch_in=16, n_ch_out=3, kernel_size=3)

        self.linear_for_mean = nn.Linear(in_features=8192, out_features=256)
        self.linear_for_std = nn.Linear(in_features=8192, out_features=256)
        self.linear_for_latent = nn.Linear(in_features=256, out_features=16384)

        self.apply(init_weights)

    def forward(self, x):
        x = self.encoder(x)
        b, c, h, w = x.shape
        x = torch.reshape(x, shape=(b, c*h*w))

        mean = self.linear_for_mean(x)
        std = self.linear_for_std(x)

        latent = torch.randn(1).to(x.device)*std + mean

        x = self.linear_for_latent(latent)
        x = torch.reshape(x, shape=(b, 1024, 4, 4))

        return torch.nn.functional.tanh(self.head_block(self.decoder(x)))






class GeneratorFromNoiseVector(nn.Module):
    def __init__(self, bias_initialization=True, **kwargs):
        super().__init__(**kwargs)

        self.decoder = decoder_no_skips(kernel_size=4, n_channels_in=[512, 512, 512, 256, 128, 64, 32], n_channels_out=[512, 512, 256, 128, 64, 32, 16],
                            e_normalization_type=E_Normalization_Type.batch_norm, e_activation_type=EActivationType.relu,
                            e_conv_layer=EConvLayer.conv2dnoise,
                            n_resnet_blocks=[1, 1, 1, 0, 0, 0, 0])

        self.head_block = head_block(n_ch_in=16, n_ch_out=3, kernel_size=3, bias_initialization=bias_initialization)

        self.linear_for_latent = nn.Linear(in_features=256, out_features=8192)

        self.apply(init_weights)

    def forward(self, latent):
        #b = x.shape[0]
        #latent = torch.randn((b, 256)).to(x.device)

        x = self.linear_for_latent(latent)
        x = torch.reshape(x, shape=(latent.shape[0], 512, 4, 4))

        return torch.nn.functional.tanh(self.head_block(self.decoder(x)))




class GeneratorUnetFromLatent(ResnetEncoderUnetDecoder):
    def __init__(self, n_channels_from_seg_net=256, default_noise_size=512, **kwargs):
        n_channels_in_encoder = [kwargs['n_channels_input']]
        for i in range(len(kwargs['n_channels_out_encoder'])-1):
            n_ch = kwargs['n_channels_out_encoder'][i]
            n_channels_in_encoder.append(n_ch)

        n_channels_in_decoder = [kwargs['n_channels_out_encoder'][-1] + kwargs['n_channels_out_encoder'][-2]]
        for i in range(len(kwargs['n_channels_out_encoder'])-2):
            #if i == 0:
            #    n_channels_in_decoder.append(n_channels_out_encoder[-1] + n_channels_out_encoder[-2])
            #else:
            n_channels = kwargs['n_channels_out_decoder'][i]+kwargs['n_channels_out_encoder'][-(i+3)]
            n_channels_in_decoder.append(n_channels)
        for j in range(len(kwargs['n_channels_out_decoder']) -i-2):
            #print(j)
            n_channels_in_decoder.append(kwargs['n_channels_out_decoder'][j+i+1])

        n_channels_in_decoder[0] += n_channels_from_seg_net

        kwargs['n_channels_in_decoder'] = n_channels_in_decoder

        self.default_noise_size = default_noise_size
        super().__init__(**kwargs)
        #self.linear_for_latent = nn.Linear(in_features=256, out_features=8192)
        self.identity_hook_for_latent = nn.Identity()


    def transform_latent_from_sec_to_spatial_size(self, latent, x):
        latent_transformed = self.identity_hook_for_latent(torch.ones((latent.shape[0], latent.shape[1], x[-1].shape[-1], x[-1].shape[-1]), device=latent.device)*latent.unsqueeze(dim=-1).unsqueeze(dim=-1))
        #raise RuntimeError
        return latent_transformed


    def forward(self, latent, noise=None, dict_inputs=None):
        if noise == None:
            noise = torch.randn((latent.shape[0], 1, self.default_noise_size, self.default_noise_size), device=latent.device)

        x = self.encoder(noise, output_all_blocks=True, dict_inputs=dict_inputs)
        #latent_transformed = self.identity_hook_for_latent(torch.ones((latent.shape[0], latent.shape[1], x[-1].shape[-1], x[-1].shape[-1]), device=latent.device)*latent.unsqueeze(dim=-1).unsqueeze(dim=-1))
        latent_transformed = self.transform_latent_from_sec_to_spatial_size(latent, x)

        #print("x[-1].shape before: {}".format(x[-1].shape))
        #print("latent_transformed: {}".format(latent_transformed.shape))
        x[-1] = torch.cat([x[-1], latent_transformed], dim=1)
        #print("x[-1].shape: {}".format(x[-1].shape))
        x = self.decoder(x, dict_inputs=dict_inputs)[-1]

        if hasattr(self, "head_block"):
            x = self.head_block(x, dict_inputs=dict_inputs)
            if self.apply_tanh:
                return self.tanh(x)#.type(torch.cuda.FloatTensor)
            return x
        else:
            if self.apply_tanh:
                return self.tanh(x)#.type(torch.cuda.FloatTensor)
            else:
                return x
        #return {"pred" : self.head_block(x).type(torch.cuda.FloatTensor)}


class GeneratorUnetFromLatent_Spatial(GeneratorUnetFromLatent):
    def transform_latent_from_sec_to_spatial_size(self, latent, x):
        #latent_transformed = self.identity_hook_for_latent(torch.ones((latent.shape[0], latent.shape[1], x[-1].shape[-1], x[-1].shape[-1]), device=latent.device)*latent.unsqueeze(dim=-1).unsqueeze(dim=-1))
        latent_transformed = self.identity_hook_for_latent(latent)
        #print("latent transformed shape: {}".format(latent_transformed.shape))
        return latent_transformed



"""
class GeneratorUnetFromLatent_IncMask(ResnetEncoderUnetDecoder):
    def __init__(self, n_channels_from_seg_net=256, default_noise_size=512, **kwargs):
        n_channels_in_encoder = [kwargs['n_channels_input']]
        for i in range(len(kwargs['n_channels_out_encoder'])-1):
            n_ch = kwargs['n_channels_out_encoder'][i]
            n_channels_in_encoder.append(n_ch)

        n_channels_in_decoder = [kwargs['n_channels_out_encoder'][-1] + kwargs['n_channels_out_encoder'][-2]]
        for i in range(len(kwargs['n_channels_out_encoder'])-2):
            #if i == 0:
            #    n_channels_in_decoder.append(n_channels_out_encoder[-1] + n_channels_out_encoder[-2])
            #else:
            n_channels = kwargs['n_channels_out_decoder'][i]+kwargs['n_channels_out_encoder'][-(i+3)]
            n_channels_in_decoder.append(n_channels)
        for j in range(len(kwargs['n_channels_out_decoder']) -i-2):
            #print(j)
            n_channels_in_decoder.append(kwargs['n_channels_out_decoder'][j+i+1])

        n_channels_in_decoder[0] += n_channels_from_seg_net

        kwargs['n_channels_in_decoder'] = n_channels_in_decoder

        self.default_noise_size = default_noise_size
        super().__init__(**kwargs)
        #self.linear_for_latent = nn.Linear(in_features=256, out_features=8192)


    def forward(self, latent, noise=None, dict_inputs=None):
        # latent (b, c) where b is batch, c is #diagnoses * seg_encoder channels (256)

        if noise == None:
            noise = torch.randn((latent.shape[0], 1, self.default_noise_size, self.default_noise_size), device=latent.device)

        x = self.encoder(noise, output_all_blocks=True, dict_inputs=dict_inputs)
        latent_transformed = torch.ones((latent.shape[0], latent.shape[1], x[-1].shape[-1], x[-1].shape[-1]), device=latent.device)*latent.unsqueeze(dim=-1).unsqueeze(dim=-1)

        #print("x[-1].shape: {}".format(x[-1].shape))
        #print("latent_transformed: {}".format(latent_transformed.shape))
        x[-1] = torch.cat([x[-1], latent_transformed], dim=1)
        #print("x[-1].shape: {}".format(x[-1].shape))
        x = self.decoder(x, dict_inputs=dict_inputs)[-1]

        if hasattr(self, "head_block"):
            x = self.head_block(x, dict_inputs=dict_inputs)
            if self.apply_tanh:
                return self.tanh(x)#.type(torch.cuda.FloatTensor)
            return x
        else:
            if self.apply_tanh:
                return self.tanh(x)#.type(torch.cuda.FloatTensor)
            else:
                return x
        #return {"pred" : self.head_block(x).type(torch.cuda.FloatTensor)}
"""







#----------------------------------------------------------------------------------------


class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_layers=8, dimension=512, **kwargs):
        super().__init__(**kwargs)

        self.sequential = nn.Sequential()
        for i in range(n_layers):
            linear = nn.Linear(dimension, dimension)
            activation = nn.LeakyReLU(inplace=True, negative_slope=0.2)

            self.sequential.add_module("linear_" + str(i), linear)
            self.sequential.add_module("activation_" + str(i), activation)

    def forward(self, x):
        return self.sequential(x)



class StyleStackedGenerator(ResnetEncoderUnetDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mlp_style_vec_gen = MultiLayerPerceptron(n_layers=8, dimension=512)

    def forward(self, x, dict_inputs={}):
        noise_latent_vec = torch.randn((x.shape[0], 512)).to(x.device)#.to(memory_format=torch.channels_last)
        noise_latent_vec = torch.nn.functional.normalize(noise_latent_vec)
        style_vec = self.mlp_style_vec_gen(noise_latent_vec)

        dict_inputs["style_vector"] = style_vec

        return super().forward(x, dict_inputs=dict_inputs)



#--------------------------------------------------------------------------------------------------------------------------------------

class DenseLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        dense = spectral_norm(nn.Linear(in_features, out_features))
        normalization = nn.BatchNorm1d(out_features)
        activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.layers = nn.Sequential()
        self.layers.add_module("dense", dense)
        self.layers.add_module("normalization", normalization)
        self.layers.add_module("activation", activation)

    def forward(self, x):
        return self.layers(x)


class DiscriminatorDense(nn.Module):
    def __init__(self, n_ch_input=256):
        super().__init__()

        self.layers = nn.Sequential()
        for i in range(8):
            self.layers.add_module("dense_block", DenseLayer(n_ch_input, n_ch_input))
        self.layers.add_module("last_dense", spectral_norm(nn.Linear(n_ch_input, 1)))

    def forward(self, x):
        return self.layers(x)


class GeneratorDense(nn.Module):
    def __init__(self, n_ch_input=256):
        super().__init__()

        self.layers = nn.Sequential()
        for i in range(8):
            self.layers.add_module("dense_block", DenseLayer(n_ch_input, n_ch_input))
        self.layers.add_module("last_dense", spectral_norm(nn.Linear(n_ch_input, n_ch_input)))

    def forward(self, shape_like):
        x = torch.randn_like(shape_like)
        return self.layers(x)







#--------------------- SPADE from nvlabs ------------------------------------------------

# from https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, norm_nc, e_param_free_norm_type=E_Normalization_Type.batch_norm, ks=3, label_nc=3):
        super().__init__()

        #assert config_text.startswith('spade')
        #parsed = re.search('spade(\D+)(\d)x\d', config_text)
        #param_free_norm_type = str(parsed.group(1))
        #ks = 3#int(parsed.group(2))


        if e_param_free_norm_type == E_Normalization_Type.instance_norm:
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        #elif param_free_norm_type == 'syncbatch':
        #    self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif e_param_free_norm_type == E_Normalization_Type.batch_norm:
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % e_param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out



#--------------------------- Custom SPADE Generator ---------------------------------------


class SPADE_ResnetDecoder(nn.Module):
    def __init__(self, num_output_channels, n_resnet_blocks, use_spectral_norm=True):
        super().__init__()

        num_input_channels = [1024]
        for i in range(len(num_output_channels)-1):
            n_ch = num_output_channels[i]
            num_input_channels.append(n_ch)

        self.up_blocks = nn.Sequential()
        for i in range(len(num_output_channels)):
            n_ch_out = num_output_channels[i]
            n_ch_in = num_input_channels[i]

            res_up_block = SPADE_up_block(in_ch=n_ch_in, out_chs=n_ch_out, kernel_size=3, e_normalization_type=E_Normalization_Type.batch_norm,
                                n_resnet_blocks=n_resnet_blocks, use_spectral_norm=use_spectral_norm)
            self.up_blocks.add_module("up_block_" + str(i), res_up_block)

        self.head_block = self.last_conv = spectral_norm(nn.Conv2d(in_channels=num_output_channels[-1], out_channels=3, kernel_size=3, stride=1, bias=True, padding="same", padding_mode="reflect"))
        self.tanh = nn.Tanh()

        self.apply(init_weights)

    def forward(self, segmap):
        x = torch.randn((segmap.shape[0], 1024, 4, 4)).to(segmap.device)
        for res_up_block in self.up_blocks:
            x = res_up_block(x, segmap)
        x = self.head_block(x)
        x = self.tanh(x)

        return x



class SPADE_conv_same_block(nn.Module):
    def __init__(self, in_ch, out_chs, kernel_size, e_normalization_type=E_Normalization_Type.batch_norm, use_spectral_norm=True):
        super().__init__()

        self.result = nn.Sequential()

        if use_spectral_norm:
            self.conv = spectral_norm(nn.Conv2d(in_channels=in_ch, out_channels=out_chs, kernel_size=kernel_size, stride=1, bias=False, padding="same", padding_mode="reflect"))
        else:
            self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_chs, kernel_size=kernel_size, stride=1, bias=False, padding="same", padding_mode="reflect")
        self.spade = SPADE(out_chs, e_normalization_type)
        self.relu = nn.ReLU(inplace=relu_inplace)


    def forward(self, x, segmap):
        x = self.conv(x)
        x = self.spade(x, segmap)

        return self.relu(x)



class SPADE_up_block(SPADE_conv_same_block):
    def __init__(self, in_ch, out_chs, kernel_size, e_normalization_type=E_Normalization_Type.batch_norm, n_resnet_blocks=0, use_spectral_norm=True):
        super().__init__(in_ch, out_chs, kernel_size, e_normalization_type, use_spectral_norm)

        self.resnet_blocks = nn.Sequential()

        for i in range(n_resnet_blocks):
            self.resnet_blocks.add_module("resnet_block_" + str(i), SPADE_ResnetBlock(n_ch=out_chs, kernel_size=kernel_size, e_normalization_type=e_normalization_type, use_spectral_norm=use_spectral_norm))


    def forward(self, x, segmap):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = super().forward(x, segmap)

        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, segmap)

        return x


class SPADE_ResnetBlock(nn.Module):
    def __init__(self, n_ch, kernel_size=3, e_normalization_type=E_Normalization_Type.batch_norm, use_spectral_norm=True):
        super().__init__()

        self.spade_01 = SPADE(n_ch, e_normalization_type, kernel_size)
        self.relu_conv_01 = nn.Sequential()
        if use_spectral_norm:
            conv = spectral_norm(nn.Conv2d(n_ch, n_ch, kernel_size, padding="same", padding_mode="reflect"))
        else:
            conv = nn.Conv2d(n_ch, n_ch, kernel_size, padding="same", padding_mode="reflect")
        relu = nn.ReLU(inplace=relu_inplace)
        self.relu_conv_01.add_module("conv", conv)
        self.relu_conv_01.add_module("relu", relu)

        self.spade_02 = SPADE(n_ch, e_normalization_type, kernel_size)
        self.relu_conv_02 = nn.Sequential()
        if use_spectral_norm:
            conv = spectral_norm(nn.Conv2d(n_ch, n_ch, kernel_size, padding="same", padding_mode="reflect"))
        else:
            conv = nn.Conv2d(n_ch, n_ch, kernel_size, padding="same", padding_mode="reflect")
        relu = nn.ReLU(inplace=relu_inplace)
        self.relu_conv_02.add_module("conv", conv)
        self.relu_conv_02.add_module("relu", relu)


    def forward(self, x, segmap):
        x = self.spade_01(x, segmap)
        x = self.relu_conv_01(x)
        x = self.spade_02(x, segmap)
        return self.relu_conv_02(x)








#------------------------------ gan generator no encoder ----------------------------------------------------------




class encoder_growable(encoder):
    def __init__(self, n_layers_to_use=None, **kwargs):
        super().__init__(**kwargs)

        self.n_layers_to_use = n_layers_to_use
        if self.n_layers_to_use == None:
            self.n_layers_to_use = len(self.down_blocks)

        self.input_head_block = head_block(n_ch_in=kwargs["n_channels_input"], n_ch_out=kwargs["n_channels_in"][-self.n_layers_to_use], kernel_size=1, bias_initialization=False,
                                e_conv_layer=kwargs["e_conv_layer"], use_spectral_norm=kwargs["use_spectral_norm"])


    def forward(self, x, dict_inputs=None):
        x = self.input_head_block(x)

        for i in range(self.n_layers_to_use):
            x = self.down_blocks[i-self.n_layers_to_use](x, dict_inputs=dict_inputs)

        return x




class encoder_growing(encoder):
    def __init__(self, n_layers_to_use=None, **kwargs):
        super().__init__(**kwargs)

        self.n_layers_to_use = n_layers_to_use
        if self.n_layers_to_use == None:
            self.n_layers_to_use = len(self.down_blocks)

        self.input_head_block = head_block(n_ch_in=kwargs["n_channels_input"], n_ch_out=kwargs["n_channels_in"][-self.n_layers_to_use], kernel_size=1, bias_initialization=False,
                                e_conv_layer=kwargs["e_conv_layer"], use_spectral_norm=kwargs["use_spectral_norm"])

        self.input_head_block_old = head_block(n_ch_in=kwargs["n_channels_input"], n_ch_out=kwargs["n_channels_in"][-self.n_layers_to_use+1], kernel_size=1, bias_initialization=False,
                                e_conv_layer=kwargs["e_conv_layer"], use_spectral_norm=kwargs["use_spectral_norm"])

        self.alpha = 0.0


    def forward(self, x, dict_inputs=None):
        x = self.input_head_block(x)

        x_half = F.interpolate(x, size=(int(x.shape[-1]/2), int(x.shape[-1]/2)), mode="nearest")
        x_half = self.input_head_block_old(x_half, dict_inputs=dict_inputs)

        for i in range(self.n_layers_to_use):
            x = self.down_blocks[i-self.n_layers_to_use](x, dict_inputs=dict_inputs)

            if i == 0:
                x = x*self.alpha + (1.0 - self.alpha)*x_half


        return x





class decoder_without_encoder_usage(decoder):
    def __init__(self, n_layers_to_use=None, **kwargs):
        super().__init__(**kwargs)

        self.n_layers_to_use = n_layers_to_use
        if self.n_layers_to_use == None:
            self.n_layers_to_use = len(self.up_blocks)

        self.head_block = head_block(n_ch_in=kwargs["n_channels_out"][self.n_layers_to_use-1], n_ch_out=3, kernel_size=1, bias_initialization=False,
                                e_conv_layer=EConvLayer.conv2d, use_spectral_norm=kwargs["use_spectral_norm"])
        new_bias = torch.ones(3)*1.2
        self.head_block.conv.bias.data[:] = new_bias


    def forward(self, x, dict_inputs=None):
        for i in range(self.n_layers_to_use):
            upsample_block = self.up_blocks[i]
            x = upsample_block(x, dict_inputs=dict_inputs)
            #print("x shape: {}".format(x.shape))

        return F.tanh(self.head_block(x))


class decoder_growing(decoder):
    def __init__(self, n_layers_to_use=None, **kwargs):
        super().__init__(**kwargs)

        self.n_layers_to_use = n_layers_to_use
        if self.n_layers_to_use == None:
            self.n_layers_to_use = len(self.up_blocks)

        self.alpha = 0.0

        self.head_block = head_block(n_ch_in=kwargs["n_channels_out"][self.n_layers_to_use-1], n_ch_out=3, kernel_size=1, bias_initialization=False,
                                e_conv_layer=kwargs["e_conv_layer"], use_spectral_norm=kwargs["use_spectral_norm"])
        new_bias = torch.ones(3)*1.2
        self.head_block.conv.bias.data[:] = new_bias

        self.head_block_old = head_block(n_ch_in=kwargs["n_channels_out"][self.n_layers_to_use-2], n_ch_out=3, kernel_size=1, bias_initialization=False,
                                e_conv_layer=kwargs["e_conv_layer"], use_spectral_norm=kwargs["use_spectral_norm"])

    def forward(self, x, dict_inputs=None):
        for i in range(self.n_layers_to_use-1):
            upsample_block = self.up_blocks[i]
            x = upsample_block(x, dict_inputs=dict_inputs)

        x_out_old = self.head_block_old(x, dict_inputs=dict_inputs)

        x = self.up_blocks[i+1](x, dict_inputs=dict_inputs)
        x = F.tanh(x)

        x_out_old = F.interpolate(x_out_old, size=(x.shape[-1], x.shape[-2]), mode="nearest")
        x_out_old = F.tanh(x_out_old)

        return x*self.alpha + x_out_old*(1.0-self.alpha)


class GeneratorNoEncoder(decoder_without_encoder_usage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply(init_weights)

    def forward(self, batch_size_shape_like):
        x = torch.zeros((batch_size_shape_like.shape[0], 512, 4, 4)).to(batch_size_shape_like.device)

        return super().forward(x)




class GeneratorNoEncoderGrowing(decoder_growing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply(init_weights)

    def forward(self, batch_size_shape_like):
        x = torch.zeros((batch_size_shape_like.shape[0], 512, 4, 4)).to(batch_size_shape_like.device)

        return super().forward(x)




class Discriminator_Growable(encoder_growable):
    def __init__(self, n_final_output_channel=1, use_minibatch_std=False, **kwargs):
        super().__init__(**kwargs)

        n_ch_last_layer = 512
        if use_minibatch_std:
            self.minibatch_std_layer = MinibatchStd()
            n_ch_last_layer = 513

        self.last_conv = spectral_norm(nn.Conv2d(in_channels=n_ch_last_layer, out_channels=n_final_output_channel, kernel_size=4, stride=1, bias=True, padding="same", padding_mode="reflect"))

        #self.last_conv = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=4, stride=1, bias=True, padding="same", padding_mode="reflect")

        #self.last_conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, bias=True, padding="same")

        self.apply(init_weights)

    def forward(self, x, dict_inputs=None):
        #return self.last_conv(x)
        if dict_inputs is not None:
            x = super().forward(x, dict_inputs=dict_inputs)
        else:
            x = super().forward(x)#[-1]
        #x = self.minibatch_std_layer(x)
        x = self.last_conv(x)
        return x



class Discriminator_Growing(encoder_growing):
    def __init__(self, n_final_output_channel=1, use_minibatch_std=False, **kwargs):
        super().__init__(**kwargs)

        n_ch_last_layer = 512
        if use_minibatch_std:
            self.minibatch_std_layer = MinibatchStd()
            n_ch_last_layer = 513

        self.last_conv = spectral_norm(nn.Conv2d(in_channels=n_ch_last_layer, out_channels=n_final_output_channel, kernel_size=4, stride=1, bias=True, padding="same", padding_mode="reflect"))

        #self.last_conv = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=4, stride=1, bias=True, padding="same", padding_mode="reflect")

        #self.last_conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, bias=True, padding="same")

        self.apply(init_weights)

    def forward(self, x, dict_inputs=None):
        #return self.last_conv(x)
        if dict_inputs is not None:
            x = super().forward(x, dict_inputs=dict_inputs)
        else:
            x = super().forward(x)#[-1]
        #x = self.minibatch_std_layer(x)
        x = self.last_conv(x)
        return x



class PixelNorm(nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x)















#----------------------------- 3d -------------------------------------------------------


def init_weights_conv3d(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.normal(m.weight, 0, 0.02)
    if isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.normal(m.weight, 0, 0.02)

class ResnetEncoderUnetDecoder_3d(nn.Module):
    def __init__(self, kernel_size_full_res_conv, kernel_size, n_channels_input, n_channel_full_res,
                    n_channels_out_encoder, n_channels_out_decoder, n_resnet_blocks_full_res,
                    n_resnet_blocks_encoder, e_normalization_type=E_Normalization_Type.group_norm,
                    n_ch_out=1, bias_initialization=False):

        super().__init__()

        n_channels_in_encoder = [n_channel_full_res]
        for i in range(len(n_channels_out_encoder)-1):
            n_ch = n_channels_out_encoder[i]
            n_channels_in_encoder.append(n_ch)

        n_channels_in_decoder = []
        for i in range(len(n_channels_out_encoder)):
            if i == 0:
                n_channels_in_decoder.append(n_channels_out_encoder[-1])
            else:
                n_channels = n_channels_out_decoder[i-1]+n_channels_out_encoder[-(i+1)]
                n_channels_in_decoder.append(n_channels)


        self.encoder = encoder_3d(kernel_size=kernel_size, n_channels_in=n_channels_in_encoder, n_channels_out=n_channels_out_encoder,
                            n_resnet_blocks=n_resnet_blocks_encoder, e_normalization_type=e_normalization_type, kernel_size_full_res_conv=kernel_size_full_res_conv,
                            n_resnet_blocks_full_res=n_resnet_blocks_full_res, n_channels_input=n_channels_input, n_channel_full_res=n_channel_full_res)

        self.decoder = decoder_3d(kernel_size=kernel_size, n_channels_in=n_channels_in_decoder, n_channels_out=n_channels_out_decoder,
                            e_normalization_type=e_normalization_type)

        self.head_block = head_block_3d(n_ch_in=n_channels_out_decoder[-1]+n_channels_in_encoder[0], n_ch_out=n_ch_out, kernel_size=kernel_size, bias_initialization=bias_initialization)

        self.apply(init_weights_conv3d)

    def forward(self, x):
        #x = self.full_res_blocks(x)
        x = self.decoder(self.encoder(x))[-1]

        return {"pred" : self.head_block(x).type(torch.cuda.FloatTensor)}


class encoder_3d(nn.Module):
    def __init__(self, kernel_size, n_channels_in, n_channels_out,
                 n_resnet_blocks, kernel_size_full_res_conv, n_channel_full_res,
                 n_resnet_blocks_full_res, n_channels_input,
                 e_normalization_type=E_Normalization_Type.group_norm):

        super().__init__()

        self.down_blocks = nn.Sequential()

        for i in range(len(n_channels_in)):
            n_ch_in = n_channels_in[i]
            n_ch_out = n_channels_out[i]
            n_res_blocks = n_resnet_blocks[i]

            downsample_block = down_block_3d(in_ch=n_ch_in, out_ch=n_ch_out, kernel_size=kernel_size,
                                    e_normalization_type=e_normalization_type, n_resnet_blocks=n_res_blocks)

            self.down_blocks.add_module("down block " + str(i), downsample_block)

        self.full_res_blocks = SingleConv_PlusResnetBlocks_3d(n_ch_in=n_channels_input, n_ch_out=n_channel_full_res,
                                    kernel_size_single_conv=kernel_size_full_res_conv, kernel_size_resnet_blocks=kernel_size,
                                    n_resnet_blocks=n_resnet_blocks_full_res, e_normalization_type=e_normalization_type)


    def forward(self, x):
        all_blocks_out = []

        x = self.full_res_blocks(x)
        #x = deepspeed.checkpointing.checkpoint(self.full_res_blocks, x)
        all_blocks_out.append(x)

        for downsample_block in self.down_blocks:
            x = downsample_block(x)
            #x = deepspeed.checkpointing.checkpoint(downsample_block, x)
            #x = torch.utils.checkpoint.checkpoint(downsample_block, x)
            all_blocks_out.append(x)

        return all_blocks_out



class decoder_3d(nn.Module):
    def __init__(self, kernel_size, n_channels_in, n_channels_out,
                 e_normalization_type=E_Normalization_Type.group_norm):

        super().__init__()

        self.up_blocks = nn.Sequential()

        for i in range(len(n_channels_in)):
            n_ch_in = n_channels_in[i]
            n_ch_out = n_channels_out[i]

            upsample_block = conv_transpose_up_block_3d(in_ch=n_ch_in, out_ch=n_ch_out, kernel_size=kernel_size,
                                    e_normalization_type=e_normalization_type)

            self.up_blocks.add_module("up block " + str(i), upsample_block)


    def forward(self, skips):
        all_blocks_out = []

        x = skips[-1]

        for i, upsample_block in enumerate(self.up_blocks):
            #if i != len(self.up_blocks)-1:
            x = upsample_block(x, skips[-(i+2)])
            #else:
            #    x = upsample_block(x)
            all_blocks_out.append(x)

        return all_blocks_out





class down_block_3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, e_normalization_type=E_Normalization_Type.group_norm, n_resnet_blocks=0, apply_activation_function=True):
        super().__init__()

        self.n_resnet_blocks = n_resnet_blocks

        self.resnet_blocks = nn.Sequential()

        self.downsample = conv_downsample_3d(in_ch, out_ch, kernel_size, e_normalization_type, apply_activation_function)

        for i in range(n_resnet_blocks):
            resnet_block = conv_same_2_resnet_normal_block_3d(out_ch, kernel_size, e_normalization_type)
            self.resnet_blocks.add_module("resnet_block " + str(i), resnet_block)

    def forward(self, x):
        x = self.downsample(x)

        for i in range(self.n_resnet_blocks):
            y = self.resnet_blocks[i](x)
            x = x.add(y)
            x = torch.nn.functional.relu(x)

        return x



class conv_downsample_3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, e_normalization_type=E_Normalization_Type.group_norm, apply_activation_function=True):
        super().__init__()

        to_pad = int((kernel_size - 1) / 2)

        self.result = nn.Sequential()

        conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=(2, 2, 2), bias=False, padding=to_pad)
        self.result.add_module("conv3d", conv)

        if e_normalization_type == E_Normalization_Type.batch_norm:
            self.result.add_module("batch_norm", nn.BatchNorm3d(out_ch, eps=0.001))
        elif e_normalization_type == E_Normalization_Type.group_norm:
            self.result.add_module("group_norm", nn.GroupNorm(32, out_ch))

        if apply_activation_function:
            if use_activation_functions:
                self.result.add_module("relu", nn.ReLU(inplace=relu_inplace))


    def forward(self, x):
        return self.result(x)



class conv_same_block_3d(nn.Module):
    def __init__(self, in_ch, out_chs, kernel_size, e_normalization_type=E_Normalization_Type.group_norm):
        super().__init__()

        self.result = nn.Sequential()

        for i, n_ch in enumerate(out_chs):
            if i == 0:
                ch_in = in_ch
            else:
                ch_in = out_chs[i-1]

            conv1 = nn.Conv3d(in_channels=ch_in, out_channels=n_ch, kernel_size=kernel_size, stride=1, bias=False, padding="same")
            self.result.add_module("conv", conv1)

            if e_normalization_type == E_Normalization_Type.batch_norm:
                self.result.add_module("batch_norm", nn.BatchNorm3d(n_ch, eps = 0.001))
            elif e_normalization_type == E_Normalization_Type.group_norm:
                self.result.add_module("group_norm", nn.GroupNorm(32, n_ch))
            else:
                self.result.add_module("batch_norm", nn.BatchNorm3d(n_ch))
            if use_activation_functions:
                self.result.add_module("relu_1", nn.ReLU(inplace = True))

    def forward(self, x):
        return self.result(x)





class up_block_3d(conv_same_block_3d):
    def __init__(self, in_ch, out_chs, kernel_size, e_normalization_type=E_Normalization_Type.group_norm):
        super().__init__(in_ch, out_chs, kernel_size, e_normalization_type)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear')
            x = torch.cat([x, skip], dim=1)

            return super().forward(x)

        else:
            #print("up block x shape before: {}".format(x.shape))
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
            #print("up block x shape after: {}".format(x.shape))
            return super().forward(x)



class conv_transpose_up_block_3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, e_normalization_type=E_Normalization_Type.group_norm):
        super().__init__()

        self.result = nn.Sequential()
        conv3dtranspose = nn.ConvTranspose3d(in_ch, out_ch, kernel_size, stride=(2, 2, 2), bias=False)
        self.result.add_module("conv_transpose", conv3dtranspose)
        if e_normalization_type == E_Normalization_Type.batch_norm:
            self.result.add_module("batch_norm", nn.BatchNorm3d(out_ch, eps = 0.001))
        elif e_normalization_type == E_Normalization_Type.group_norm:
            self.result.add_module("group_norm", nn.GroupNorm(32, out_ch))
        else:
            self.result.add_module("batch_norm", nn.BatchNorm3d(out_ch))
        if use_activation_functions:
            self.result.add_module("relu_1", nn.ReLU(inplace = True))


    def forward(self, x, skip=None):
        #print("conv transpose x shape: {}".format(x.shape))
        if skip is not None:
            x = self.result(x)
            #print("x upscaled shape: {}".format(x.shape))
            #print("skip shape: {}".format(skip.shape))
            #x = torch.cat([x, skip], dim=1)
            x = concatenate_3d(x, skip)

            return x
        else:
            #print("WARNING: No skip")
            x = self.result(x)
            return x


class conv_same_2_resnet_normal_block_3d(nn.Module):
    def __init__(self, n_ch, kernel_size, e_normalization_type=E_Normalization_Type.group_norm):
        super().__init__()

        self.result = nn.Sequential()

        conv1 = nn.Conv3d(in_channels=n_ch, out_channels=n_ch, kernel_size=kernel_size, stride=1, bias=False, padding="same")
        self.result.add_module("conv", conv1)

        if e_normalization_type == E_Normalization_Type.batch_norm:
            self.result.add_module("batch_norm", nn.BatchNorm3d(n_ch, eps = 0.001))
        elif e_normalization_type == E_Normalization_Type.group_norm:
            self.result.add_module("group_norm", nn.GroupNorm(32, n_ch))
        else:
            self.result.add_module("batch_norm", nn.BatchNorm3d(n_ch))
        if use_activation_functions:
            self.result.add_module("relu_1", nn.ReLU(inplace = True))

        conv2 = nn.Conv3d(n_ch, n_ch, kernel_size, stride = 1, bias=False, padding = "same")#padding = to_pad)
        self.result.add_module("conv2d_1, ", conv2)

        if e_normalization_type == E_Normalization_Type.batch_norm:
            self.result.add_module("batch_norm", nn.BatchNorm3d(n_ch, eps = 0.001))
        elif e_normalization_type == E_Normalization_Type.group_norm:
            self.result.add_module("group_norm", nn.GroupNorm(32, n_ch))
        else:
            self.result.add_module("batch_norm", nn.BatchNorm3d(n_ch))

    def forward(self, x):
        #print("input x: {}".format(x))
        #print("x shape: {}".format(x.shape))
        #print("x[0]".format(x[0]))
        #return self.result(x)

        x = deepspeed.checkpointing.checkpoint(self.result, x)
        #x = self.result(x)
        return x
        #return self.result(x)



class head_block_3d(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, kernel_size, bias_initialization=False):
        super().__init__()

        self.result = nn.Sequential()
        conv1 = nn.Conv3d(in_channels=n_ch_in, out_channels=n_ch_out, kernel_size=kernel_size, stride=1, bias=True, padding="same")
        self.result.add_module("conv head", conv1)

        if bias_initialization:
            print("bias: {}".format(self.result[0].bias))
            new_bias = np.log([1/400.0])
            new_bias = new_bias.astype(np.single)
            new_bias = torch.from_numpy(new_bias)

            self.result[0].bias.data[:] = new_bias

            print("bias: {}".format(self.result[0].bias))

    def forward(self, x):
        #print("x: {}".format(x))
        return self.result(x)




class SingleConv_PlusResnetBlocks_3d(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, kernel_size_single_conv, kernel_size_resnet_blocks, n_resnet_blocks=0,
                 e_normalization_type=E_Normalization_Type.group_norm):
        super().__init__()

        self.n_resnet_blocks = n_resnet_blocks

        self.conv = nn.Conv3d(n_ch_in, n_ch_out, kernel_size_single_conv, stride=1, bias=False, padding="same")
        #nn.init.normal_(self.conv.weight, 0, 0.02)

        self.conv_same_blocks = nn.Sequential()

        for k in range(self.n_resnet_blocks):
            self.conv_same_blocks.add_module("2_normal_resnet_block_first_" + "_" + str(k), conv_same_2_resnet_normal_block_3d(n_ch_out, kernel_size_resnet_blocks, e_normalization_type=e_normalization_type))


    def forward(self, x):
        x = self.conv(x)

        for k in range(self.n_resnet_blocks):
            y = self.conv_same_blocks[k](x)
            x = x.add(y)
            x = torch.nn.functional.relu_(x)

        return x







def concatenate_3d(x, skip):
    #print("x shape: {}".format(x.shape))
    #print("skip shape: {}".format(skip.shape))
    lower_d = int(np.floor((x.shape[2]-skip.shape[2])/2.0))
    higher_d = x.shape[2] - int(np.ceil((x.shape[2]-skip.shape[2])/2.0))
    lower_h = int(np.floor((x.shape[3]-skip.shape[3])/2.0))
    higher_h = x.shape[3] - int(np.ceil((x.shape[3]-skip.shape[3])/2.0))
    lower_w = int(np.floor((x.shape[4]-skip.shape[4])/2.0))
    higher_w = x.shape[4] - int(np.ceil((x.shape[4]-skip.shape[4])/2.0))

    x = x[:, :, lower_d:higher_d, lower_h:higher_h, lower_w:higher_w]
    #print("x shape after: {}".format(x.shape))
    #x = torchvision.transforms.functional.center_crop(x, [skip.shape[2], skip.shape[3], skip.shape[4]])
    return torch.cat([x, skip], dim=1)
























# ------------------------------- Deprecated -------------------------------------------------------------

