from torch import nn
import numpy as np
from abc import abstractmethod
import sys
import os
import torch # 现在可以进行绝对导入
from __init__ import time_embedding
from __init__ import Downsample
from __init__ import Upsample
from torch.nn import functional as F

import math

def group_norm(channels):
    return nn.GroupNorm(4, channels)


# 包含 time_embedding 的 block
class TimeBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        函数不能为空，但可以添加注释
        """


class TimeSequential(nn.Sequential, TimeBlock):
    def forward(self, x, fd=None, emb=None):  # 给fd和emb设置默认值
        for layer in self:
            # 判断该 layer 中是否是 TimeBlock 类型
            if isinstance(layer, TimeBlock):
                if fd is not None and emb is not None:
                    x = layer(x, fd, emb)  # 如果需要 fd 和 emb，则传递它们
                else:
                    x = layer(x, emb)  # 如果不需要 fd 和 emb，只传递 x
            else:
                x = layer(x)
        return x

class FFParser(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x

class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)
    
class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = torch.sigmoid(y)
        return x * y
    
    
class SPPELAN(nn.Module):
    # spp-elan
    def __init__(
        self, c1
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        # Feature extraction layers
        self.cv1= nn.Sequential(   
            group_norm(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1//2, kernel_size=1),

        )

        self.cv2 = SP(3)
        self.cv3 = SP(5)
        self.cv4 = SP(7)
        #SE layer
        self.se = SELayer(2 * c1)
        # Feature fusion layers
        self.cv_fuse= nn.Sequential(   
            group_norm(2 * c1),
            nn.SiLU(),
            nn.Conv2d(2 * c1, c1, kernel_size=1),
        )

    def forward(self, x):
        # Initial 1x1 convolution to adjust channels
        x1 = self.cv1(x)
        
        # Apply pooling operations
        x2 = self.cv2(x1)
        x3 = self.cv3(x1)
        x4 = self.cv4(x1)
        
        # Concatenate features
        x_concat = torch.cat([x1, x2, x3, x4], dim=1)
        
        # Apply SE layer to adjust feature weights
        x_se = self.se(x_concat)
        
        # Final convolution to adjust output channels
        return self.cv_fuse(x_se)



class OutNoiseBlock(TimeBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),

        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(     
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(   
            group_norm(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x,fd, t):
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.SiLU(),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=4, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            group_norm(num_feat),
            nn.SiLU(),
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            group_norm(num_feat // compress_ratio),
            nn.SiLU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        # print(qkv.shape)
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)

        # print(q.shape)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        # print(print(h.shape))
        h = h.reshape(B, -1, H, W)
        # print("s",print(h.shape))
        h = self.proj(h)
        return h + x
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList([
            nn.Sequential(

                group_norm(in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),

            ) for rate in atrous_rates
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),

            group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),

        )
        self.output = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        res = []
        for block in self.aspp_blocks:
            res.append(block(x))
        res.append(F.interpolate(self.global_avg_pool(x), size=x.shape[2:], mode='bilinear', align_corners=False))
        return self.output(torch.cat(res, dim=1))

class CombinedAttention(nn.Module):
    def __init__(self, in_channels,num_heads):
        super(CombinedAttention, self).__init__()
        self.ca = CAB(in_channels)
        self.sa = AttentionBlock(in_channels,num_heads)
        self.residual = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),

        )

    def forward(self, x):
        ca_out = self.ca(x) * x
        sa_out = self.sa(x)
        out = ca_out + sa_out
        return self.residual(out + x)+x

class Fusion(nn.Sequential):

    def __init__(self, input):
        super(Fusion, self).__init__()
        self.convA = nn.Sequential(
            group_norm(input*2),
            nn.SiLU(),
            nn.Conv2d(input*2, input, kernel_size=3, stride=1, padding=1),
        )
        self.convB = nn.Sequential(
            group_norm(input),
            nn.SiLU(),
            nn.Conv2d(input, input, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x, concat_with):
        return self.convB(self.convA(torch.cat([x, concat_with], dim=1)))
    
def sepConv3d(in_planes, out_planes, kernel_size, stride, pad,bias=False):

    if bias:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias))
    else:
        return nn.Sequential(group_norm(in_planes),
                             nn.SiLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias))
                         
class ResBlock(TimeBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            group_norm(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x,fd, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        h = h.view(t.shape[0], -1, *h.shape[-3:])    
        B, FS, C, H, W = h.shape
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None].unsqueeze(1) 
        h = h.view(B*FS, C, H, W) # BxFS C H W
        h = self.conv2(h)
        return h + self.shortcut(x)
    
class SMFA(nn.Module):
    def __init__(self, inp):
        super(SMFA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, 3, padding=1, stride=1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # self.batch=batch
  
        self.excite = nn.Sequential(
            nn.Conv2d(inp, inp//2, kernel_size=3, padding=1,stride=1 ,groups=inp//2),
            group_norm(inp//2),
            nn.SiLU(),
            nn.Conv2d(inp//2, inp, kernel_size=3, padding=1,stride=1,groups=inp//2),
            nn.Sigmoid()
        )

        self.classify = nn.Sequential(sepConv3d(inp, inp, 3, (1,1,1), 1),

                                       sepConv3d(inp, 1, 3, (1,1,1),1))
        self.conv1x11 = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)

        self.conv3x3=nn.Sequential(
            nn.Conv2d(inp,inp,3,1,1,groups=inp),
            nn.Conv2d(inp,inp//2,1,1)
        )
        self.conv1x1 =nn.Sequential(
            nn.Conv2d(inp, inp//2, kernel_size=1, stride=1, padding=0)
        ) 
        self.conv1x12=nn.Sequential(group_norm(inp),
                                    nn.SiLU(),
                                    nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0))

        self.outconv=nn.Sequential(group_norm(inp),
                                    nn.SiLU(),
                                    nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0))

    def sge(self, x):
        # [N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w # .repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather) # [N, 1, C, 1]

        return ge

    def forward(self, x,fd):
        s_x = x.view(-1, fd.shape[1], *x.shape[-3:])    
        B, FS, C, H, W = s_x.shape
        new_x=s_x.permute(0, 2, 1, 3, 4)

        cost=self.classify(new_x).squeeze(1)
        regression=F.softmax(cost,1)*fd.unsqueeze(-1).unsqueeze(-1)

        loc = self.dwconv_hw(x)
        costloc=regression.unsqueeze(2)*loc.view(B,FS, C, H, W)
        att = self.sge(x)


        out = att * loc+ costloc.view(B*FS, C, H, W) # BxFS C H W
        output1=self.conv1x11(out)+loc

        x2=self.conv3x3(x)
        x3=self.conv1x1(x)
        output2=self.conv1x12(torch.cat([x2, x3], dim=1))

        output=self.outconv(output1+output2)
        return output

class DPW(nn.Module):

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3],
                 channel_split=[1, 3],
                ):
        super(DPW, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_0 = int(self.split_ratio[0] * embed_dims) 
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims) 
        self.embed_dims = embed_dims

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=3,
            padding=(1 + 2 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
            
        )

        self.DW_conv1 = nn.Sequential(group_norm(self.embed_dims_0),
                                      nn.SiLU(),
                                      nn.Conv2d(
                                        in_channels=self.embed_dims_0,
                                        out_channels=self.embed_dims_0,
                                        kernel_size=3,
                                        padding=1,
                                        groups=self.embed_dims_0,
                                        stride=1)
                                    )
        self.DW_conv2 = nn.Sequential(  
                                        group_norm(self.embed_dims_1),
                                        nn.SiLU(),
                                        nn.Conv2d(
                                        in_channels=self.embed_dims_1,
                                        out_channels=self.embed_dims_1,
                                        kernel_size=5,
                                        padding=(1 + 4 * dw_dilation[1]) // 2,
                                        groups=self.embed_dims_1,
                                        stride=1, dilation=dw_dilation[1])
                                    )

        self.PW_conv = nn.Sequential(  
                                        group_norm(embed_dims),
                                        nn.SiLU(),
                                        nn.Conv2d(
                                                    in_channels=embed_dims,
                                                    out_channels=embed_dims,
                                                    kernel_size=1)
                                    )

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, :self.embed_dims_0, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims_0:, ...])
        x = torch.cat(
            [x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x

class FRF(nn.Module):

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3]
                ):
        super(FRF, self).__init__()

        self.embed_dims = embed_dims

        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = DPW(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )

        self.proj_2 = nn.Sequential(
            group_norm(embed_dims*2),
            nn.SiLU(),
            nn.Conv2d(
            in_channels=embed_dims*2, out_channels=embed_dims, kernel_size=1)

        )

    def forward(self, x):
        shortcut = x.clone()
        g = self.gate(x)
        v = self.value(x)
        x = self.proj_2(torch.cat([g, v], dim=1))
        x = x + shortcut
        return x

    
class FeatureResBlock(TimeBlock):
    def __init__(self, in_channels, out_channels,time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(),
            SMFA(in_channels)

        )
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            group_norm(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            FRF(out_channels)


        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x,fd, t):

        x = self.conv1[0](x)  
        x = self.conv1[1](x) 
        
  
        h = self.conv1[2](x, fd)  

        h = h.view(t.shape[0], -1, *h.shape[-3:])    
        B, FS, C, H, W = h.shape

        h += self.time_emb(t)[:, :, None, None].unsqueeze(1) 
        h = h.view(B*FS, C, H, W) # BxFS C H W
        h = self.conv2(h)
        return h + self.shortcut(x)
    
class NoisePred(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 model_channels,
                 num_res_blocks,
                 dropout,
                 time_embed_dim_mult,
                 down_sample_mult
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.down_sample_mult = down_sample_mult

        # time embedding
        time_embed_dim = model_channels * time_embed_dim_mult
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 下采样和上采样的通道数
        down_channels = [model_channels * i for i in down_sample_mult]
        up_channels = down_channels[::-1]

        # 每个块中 ResBlock 的数量
        downBlock_chanNum = [num_res_blocks + 1] * (len(down_sample_mult) - 1)
        downBlock_chanNum.append(num_res_blocks)
        upBlock_chanNum = downBlock_chanNum[::-1]
        self.downBlock_chanNum_cumsum = np.cumsum(downBlock_chanNum)
        self.upBlock_chanNum_cumsum = np.cumsum(upBlock_chanNum)[:-1]

        self.inBlock = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        self.downBlock = nn.ModuleList()
        down_init_channel = model_channels
        for level, channel in enumerate(down_channels):
            for i in range(num_res_blocks):
                layer1 = FeatureResBlock(in_channels=down_init_channel,
                                  out_channels=channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout,
                                  )
                if i!=num_res_blocks-1:
                    down_init_channel = channel
                else:
                    down_init_channel = channel*2
                self.downBlock.append(TimeSequential(layer1))

            if level != len(down_sample_mult) - 1:
                down_layer = Downsample(channels=channel)
                self.downBlock.append(TimeSequential(down_layer))
        self.spp=SPPELAN(down_channels[-1])

        self.middleBlock = nn.ModuleList()
        for i in range(num_res_blocks):
            layer2 = FeatureResBlock(in_channels=down_channels[-1]*2,
                              out_channels=down_channels[-1]*2,
                              time_channels=time_embed_dim,
                              dropout=dropout,
                              )
            self.middleBlock.append(TimeSequential(layer2))
            if i==num_res_blocks-1:
                down_channelsBlock=nn.Conv2d(down_channels[-1]*2, down_channels[-1], kernel_size=1, stride=1, padding=0)
                self.middleBlock.append(TimeSequential(down_channelsBlock))
        
        self.in_noiseBlock = nn.Conv2d(1, down_channels[0], kernel_size=3, padding=1)

        self.noise_downBlock = nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.fusion2 = nn.Sequential(
            Fusion(down_channels[3]),
            Fusion(down_channels[2]),
            Fusion(down_channels[1]),
            Fusion(down_channels[0]),
        )
        self.midfusion=Fusion(down_channels[3])
        self.ffparser=nn.ModuleList()
        down_init_channel = model_channels
        ds = 1
        for level, channel in enumerate(down_channels):
            for _ in range(num_res_blocks):
                layer1 =[ResBlock(in_channels=down_init_channel,
                                  out_channels=channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout)] 
                down_init_channel = channel
                if ds in [8,16]:
                    layer1.append(CombinedAttention(down_init_channel, num_heads=4))
                self.noise_downBlock.append(TimeSequential(*layer1))
            self.ffparser.append(FFParser(channel, 224 // (2 **level), 224 // (2 **(level+1))+1))
            self.fusion.append(Fusion(channel))

            if level != len(down_sample_mult) - 1:
                down_layer = Downsample(channels=channel)
                self.noise_downBlock.append(TimeSequential(down_layer))
                ds *= 2

        # middle block
        self.noise_middle_block = TimeSequential(
            ResBlock(down_init_channel, down_init_channel, time_embed_dim, dropout),
            CombinedAttention(down_init_channel, num_heads=4),
            ResBlock(down_init_channel, down_init_channel, time_embed_dim, dropout)
        )

        self.noise_upBlock = nn.ModuleList()
        up_init_channel = down_channels[-1]
        for level, channel in enumerate(up_channels):
            if level == len(up_channels) - 1:
                out_channel = model_channels
            else:
                out_channel = channel // 2
            for _ in range(num_res_blocks):
                layer3 = [ResBlock(in_channels=up_init_channel,
                                  out_channels=out_channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout)]
                up_init_channel = out_channel
                if ds in [8,16]:
                    layer3.append(CombinedAttention(up_init_channel, num_heads=4))
                    ds *= 2
                self.noise_upBlock.append(TimeSequential(*layer3))
            if level > 0:
                up_layer = Upsample(channels=out_channel)
                self.noise_upBlock.append(TimeSequential(up_layer))
        self.out_nosie=TimeSequential(OutNoiseBlock(1,32,time_channels=time_embed_dim,dropout=dropout))
        # out block
        self.outBlock = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )
        self.learnable_weight = nn.Parameter(torch.ones((1,1,224,224)))
        
    def forward(self, x,fd, y, timesteps):

        B, FS, C, H, W = x.shape
        h = x.view(B*FS, C, H, W) # BxFS C H W

        embedding = time_embedding(timesteps, self.model_channels)
        time_emb = self.time_embed(embedding)

        n_res = []
        n_ffparser=[]
        n = self.in_noiseBlock(y)

        num_down = 1
        for down_block in self.noise_downBlock:
            n = down_block(n,fd, time_emb)
            if num_down in self.downBlock_chanNum_cumsum:
                n_res.append(n)
            if num_down in [2,5,8,11]:
                n_ffparser.append(self.ffparser[num_down//3](n))
            num_down += 1

        res = []
        h = self.inBlock(h)


        num_down = 1
        for down_block in self.downBlock:
            h = down_block(h,fd, time_emb)
            if num_down in self.downBlock_chanNum_cumsum:

                w_h = h.view(B, FS, *h.shape[-3:]) 
                res.append(torch.max(w_h, dim=1)[0]) 
                stack_pool = h.view(B, FS, *h.shape[-3:])    
                sum_pool_max  = torch.max(stack_pool, dim=1)[0].unsqueeze(1).expand_as(stack_pool).contiguous().view(B*FS, *h.shape[-3:])
                h = torch.cat([h, sum_pool_max], dim=1)

            # add
            elif(num_down in [1,4,7,10]):
                th = h.view(B, FS, *h.shape[-3:])   
                h=self.fusion[num_down//3](h,n_ffparser[num_down//3].unsqueeze(1).expand_as(th).contiguous().view(B*FS, *h.shape[-3:]))
            num_down += 1


        # middle stage
        for middle_block in self.middleBlock:
            h = middle_block(h,fd, time_emb)

        w_h = h.view(B, FS, *h.shape[-3:]) # B FS C H W

        h = torch.max(w_h, dim=1)[0]
        h=self.spp(h)
        
        n = self.noise_middle_block(n,fd, time_emb)
        n_f = self.midfusion(h, n)
        n = self.fusion2[0](n_f,n_res.pop())+n_f


        num_up = 1
        i=1
        for up_block in self.noise_upBlock:

            if num_up in self.upBlock_chanNum_cumsum:
                n = up_block(n,fd, time_emb)
                n_crop = n[:, :, :n_res[-1].shape[2], :n_res[-1].shape[3]]
                n=self.fusion2[i](n_crop,n_res.pop())
                i+=1
            else:
                n = up_block(n,fd, time_emb)
            num_up += 1
        n1=self.out_nosie(y,fd,time_emb)
        n=n+n1

        out = self.outBlock(n)*self.learnable_weight

        return out
if __name__ == '__main__':

    net=NoisePred(3,1,32,2,0.1,4,[1, 2, 4, 8]) #不变 只需该batchsize
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(net(torch.randn(2,5,3,224,224),torch.randn(2,5),torch.randn(2,1,224,224),torch.randint(0, 100, (2,)).long()))