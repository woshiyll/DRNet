import torch
import torch.nn as nn
import pvt_v2
import torchvision.models as models
from torch.nn import functional as F
from layers import *
import numbers
from NDM_model import Conv2dBlock
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np


class Transformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.encoder = getattr(pvt_v2, backbone)()
        if pretrained:
            checkpoint = torch.load('pvt_v2_b3.pth', map_location='cpu')
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = self.encoder.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            self.encoder.load_state_dict(checkpoint_model, strict=False)

def Encoder():
    model = Transformer('pvt_v2_b3', pretrained=True)
    return model

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class DEBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DEBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res



def to_3d(x):
    # return rearrange(x, 'b c h w -> b (h w) c')
    b, c, h, w = x.shape
    return x.view(b, c, -1).transpose(1, 2).contiguous()

def to_4d(x, h, w):
    # return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
    b, t, c = x.shape
    return x.transpose(1, 2).contiguous().view(b, c, h, w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):

        input_R = F.interpolate(input_R, [input_S.shape[2], input_S.shape[3]])

        input_R = self.conv1(input_R)  #input_R : 512-->3
        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        input_R = input_R + self.ffn(self.norm2(input_R))

        return input_R


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
class BatchImageNormalize:
    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()

        tensor.sub_(self.mean.to(tensor.device)).div_(self.std.to(tensor.device))
        return tensor

def W(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels)
    )
class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out
class MultiEmbedding(nn.Module):
    def __init__(self, in_channels, num_head=2, ratio=1):
        super(MultiEmbedding, self).__init__()
        self.in_channels = in_channels

        self.num_head = num_head
        self.out_channel = int(num_head * in_channels * ratio)
        self.query_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.key_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.value_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.W = W(int(in_channels * ratio), in_channels)

        self.fuse = nn.Sequential(ConvBlock(in_channels * 2, in_channels),
                                  nn.Conv2d(in_channels, in_channels, kernel_size=1))

    def forward(self, key, query):

        batch, channels, height, width = query.size()
        q_out = self.query_conv(query).contiguous().view(batch, self.num_head, -1, height, width)
        k_out = self.key_conv(key).contiguous().view(batch, self.num_head, -1, height, width)
        v_out = self.value_conv(key).contiguous().view(batch, self.num_head, -1, height, width)

        att = (q_out * k_out).sum(dim=2) / np.sqrt(self.out_channel)

        if self.num_head == 1:
            softmax = att.unsqueeze(dim=2)
        else:
            softmax = F.softmax(att, dim=1).unsqueeze(dim=2)

        weighted_value = v_out * softmax
        weighted_value = weighted_value.sum(dim=1)
        out = self.W(weighted_value)
        return self.fuse(torch.cat([key, out], dim=1))


class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        model = Encoder()
        self.rgb_net = model.encoder
        channels512 = 512
        channels3 = 3


        self.down_level1_block = DEBlock(default_conv, 64, 3)
        self.down_level2_block = DEBlock(default_conv, 128, 3)
        self.down_level3_block = DEBlock(default_conv, 320, 3)

        self.down_level4_block = TransformerBlock(channels512, channels3, num_heads=3)

        self.up_level3_block = DEBlock(default_conv, 320, 3)
        self.up_level2_block = DEBlock(default_conv, 128, 3)
        self.up_level1_block = DEBlock(default_conv, 64, 3)


        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 320, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up4 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))


        self.mix1 = CGAFusion(320, reduction=8)
        self.mix2 = CGAFusion(128, reduction=4)
        self.mix3 = CGAFusion(64, reduction=2)

        self.output = nn.Conv2d(64, 1, 1, 1, 0)
        self.conv3_512 = nn.Conv2d(3, 512, 1, 1, 0)

        norm_mean_std = ([.485, .456, .406], [.229, .224, .225])

        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])
        self.SE = MultiEmbedding(3)
        self.EVC_64 = EVCBlock(
            int(64),
            int(64),
            channel_ratio=2, base_channel=8,
            )
        self.EVC_512 = EVCBlock(
            int(512),
            int(512),
            channel_ratio=2, base_channel=8,
            )
        self.conv_320 = nn.Conv2d(320 * 2, 320, kernel_size=3, stride=1, padding=1)
        self.conv_128 = nn.Conv2d(128 * 2, 128, kernel_size=3, stride=1, padding=1)
        self.conv_64 = nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(64 + 64 + 3, 3, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(3, 6, kernel_size=1, stride=1)
        self.conv_lbp = nn.Conv2d(1, 64, kernel_size=1, stride=1)
        self.conv_1 = nn.Conv2d(6, 3, kernel_size=1, stride=1)

        routing_function1 = RountingFunction(in_channels=320, kernel_number=4)
        self.conv_ARC4 = AdaptiveRotatedConv2d(in_channels=320, out_channels=320,
                                             kernel_size=3, padding=1, rounting_func=routing_function1, bias=False,
                                             kernel_number=4)
        routing_function2 = RountingFunction(in_channels=128, kernel_number=4)
        self.conv_ARC3 = AdaptiveRotatedConv2d(in_channels=128, out_channels=128,
                                             kernel_size=3, padding=1, rounting_func=routing_function2, bias=False,
                                             kernel_number=4)
        routing_function3 = RountingFunction(in_channels=64, kernel_number=4)
        self.conv_ARC2 = AdaptiveRotatedConv2d(in_channels=64, out_channels=64,
                                             kernel_size=3, padding=1, rounting_func=routing_function3, bias=False,
                                             kernel_number=4)

    def prepare_input(self, image):
        image = self.normalization(image)
        return image

    def forward(self, imgs, att, lbp):

        image = self.prepare_input(imgs)
        lbp64 = self.conv_lbp(lbp)
        img1, img2, img3, img4 = self.rgb_net(image)
        img4_320 = self.up1(img4)
        img4_128 = self.up2(img4_320)
        img4_64 =  self.up3(img4_128)
        img4_64_low, img4_64_sod = self.EVC_64(img4_64)
        img4_512_low, img4_512_sod = self.EVC_512(img4)
        att_96 = F.interpolate(att, (96, 96), mode='bilinear', align_corners=True)
        lbp64_96 = F.interpolate(lbp64, (96, 96), mode='bilinear', align_corners=True)
        out_low_sp = self.conv_3(torch.cat([img4_64_low, lbp64_96, att_96], 1))
        out_low_sp = F.interpolate(out_low_sp, (384, 384), mode='bilinear', align_corners=True)
        img4_up = self.up1(img4_512_sod)
        img4_up = self.conv_ARC4(img4_up)
        x_level4_mix = self.conv_320(torch.cat([img4_up, img4_320], 1))
        img3_up = self.up2(x_level4_mix)
        img3_up = self.conv_ARC3(img3_up)
        x_level3_mix = self.conv_128(torch.cat([img3_up, img4_128], 1))
        img2_up = self.up3(x_level3_mix)
        img2_up = self.conv_ARC2(img2_up)
        x_level2_mix = self.conv_64(torch.cat([img2_up, img4_64], 1))
        sod_sp = x_level2_mix
        out_sod_sp = self.up4(sod_sp)
        output1_sod_sp = F.interpolate(out_sod_sp, (384, 384), mode='bilinear', align_corners=True)
        output_sod_sp = F.sigmoid(output1_sod_sp)

        return out_low_sp, output1_sod_sp, output_sod_sp


