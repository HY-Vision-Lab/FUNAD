import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import random

from model_utils import (
        normalization,
        Downsample,
        zero_module,
        AttentionBlock
)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(28 * 28, 768)
    
    def forward(self, x):
        size = x.shape[0]
        idx = torch.linspace(0, 28*28-1, steps=28*28, dtype=torch.long)
        idx = torch.repeat_interleave(idx.unsqueeze(0), repeats=size, dim=0)
        embed_x = self.embedding(idx.to('cuda'))
        x[:, :, 768:] = x[:, :, 768:] + embed_x

        return x

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]     

class feature_adaptor(nn.Module):
    def __init__(self):
        super(feature_adaptor, self).__init__()

        self.adaptor = nn.Sequential(
            nn.Linear(len_feature, len_feature), # feature adoptor
            nn.LeakyReLU(.2),
        )
    
    def forward(self, x):
        adapted_features = self.adaptor(x)
        return adapted_features

class Conv1x1(nn.Module):
    def __init__(self, in_channels=768, out_channels=1536):
        super(Conv1x1, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Reshape from (batch_size, patch_size, channels) to (batch_size, channels, height, width)
        #batch_size, patch_size, channels = x.shape
        #x = torch.tensor(x)
        #x = x.reshape(batch_size, int(np.sqrt(patch_size)), int(np.sqrt(patch_size)), channels).permute(0,3,1,2)
        
        # Apply 1x1 convolution
        x = self.conv1x1(x)
        
        # Reshape back to (batch_size, patch_size, out_channels)
        #x = x.view(batch_size, -1, patch_size).permute(0,2,1)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.ConvTranspose2d(in_channels=channels,
                                           out_channels=self.out_channels,
                                           kernel_size=4,
                                           stride=2, padding=1)
    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv:
            x = self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x

class ResBlock(nn.Module):

    def __init__(
        self,
        channels,
        out_channels=None,
        use_conv=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, True)
            self.x_upd = Upsample(channels, True)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                 channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d( channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        return self.skip_connection(x) + h
    

class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels, # 
        out_channels,
        model_channels,
        num_res_blocks,
        channel_mult,
        attention_mult,
        num_heads = 4,
        num_heads_upsample=-1,
        num_head_channels = 64,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.channel_mult = channel_mult

        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, ch, 3, padding=1)]
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        out_channels=int(mult * model_channels),
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_mult:
                    layers.append(
                                AttentionBlock(
                                    ch,
                                    num_heads=num_heads,
                                    num_head_channels=num_head_channels,
                            )
                    )

                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                        ResBlock(
                            ch,
                            out_channels=out_ch,
                            down=True,
                        )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch


        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        out_channels=int(model_channels * mult),
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_mult:
                    layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads_upsample,##
                                num_head_channels=num_head_channels,
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            out_channels=out_ch,
                            up=True,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )
    def forward(self, x):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # [B, 256, 64, 64]
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h)
        return self.out(h) # [B, 256, 64, 64]
    


class localnet(nn.Module):
    def __init__(self, len_feature, feat_select=False):
        super(localnet, self).__init__()
        
        # org
        self.adaptor = nn.Sequential(
            nn.Linear(len_feature, len_feature), # feature adoptor
            nn.LeakyReLU(.2),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(len_feature, 1024),
            nn.LeakyReLU(.2),
            nn.Linear(1024, 128),
            nn.LeakyReLU(.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # chgd
        self.feat_select = feat_select
        if feat_select:
            self.conv1x1_layer = Conv1x1(in_channels=512, out_channels=1536).cuda()

    def forward(self, x, synthetic_feat=None):
        adapted_features = self.adaptor(x) 
        local_score = self.discriminator(adapted_features)
        return adapted_features, local_score.squeeze()

class globalnet(nn.Module):
    def __init__(self, len_feature):
        super(globalnet, self).__init__()


        self.adaptor = nn.Sequential(
            nn.Linear(len_feature, len_feature), # feature adoptor
            nn.LeakyReLU(.2),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(len_feature, 1024),
            nn.LeakyReLU(.2),
            nn.Linear(1024, 128),
            nn.LeakyReLU(.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        adapted_features = self.adaptor(x)
        global_score = self.discriminator(adapted_features)
        return global_score.squeeze()