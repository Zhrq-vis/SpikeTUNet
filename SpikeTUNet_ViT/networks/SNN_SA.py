import torch
import torch.nn as nn
import torch.nn.functional as F
from .quan_w import Conv2dLSQ

# -------------------------------------------------------
# 批归一化 + Padding 层
# -------------------------------------------------------
class BNAndPadLayer(nn.Module):
    def __init__(self, pad_pixels, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)

            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self): return self.bn.weight

    @property
    def bias(self): return self.bn.bias

    @property
    def running_mean(self): return self.bn.running_mean

    @property
    def running_var(self): return self.bn.running_var

    @property
    def eps(self): return self.bn.eps


# -------------------------------------------------------
# 局部感知卷积（LpConv）
# -------------------------------------------------------
class LpConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        self.body = nn.Sequential(
            Conv2dLSQ(in_channel, in_channel, 1, 1, 0, bias=False, groups=1),
            BNAndPadLayer(pad_pixels=1, num_features=in_channel),
            nn.Sequential(
                Conv2dLSQ(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
                Conv2dLSQ(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        )

    def forward(self, x):
        return self.body(x)


# -------------------------------------------------------
# 自定义 ReLU 激活，带阈值上限
# -------------------------------------------------------
class ReLUX(nn.Module):
    def __init__(self, thre=8):
        super().__init__()
        self.thre = thre

    def forward(self, input):
        return torch.clamp(input, 0, self.thre)

relu4 = ReLUX(thre=4)


# -------------------------------------------------------
# 多脉冲激活函数及模块（Multispike）
# -------------------------------------------------------
class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(relu4(input) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input * (0 < input).float() * (input < ctx.lens).float(), None


class Multispike(nn.Module):
    def __init__(self, lens=4):
        super().__init__()
        self.lens = lens
        self.spike = multispike

    def forward(self, inputs):
        return self.spike.apply(4 * inputs, self.lens) / 4


class Multispike_att(nn.Module):
    def __init__(self, lens=4):
        super().__init__()
        self.lens = lens
        self.spike = multispike

    def forward(self, inputs):
        return self.spike.apply(4 * inputs, self.lens) / 2


# -------------------------------------------------------
# 基于多脉冲机制的自注意力模块
# -------------------------------------------------------
class SASG(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.head_lif = Multispike()
        self.q_conv = nn.Sequential(LpConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(LpConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(LpConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_lif = Multispike()
        self.k_lif = Multispike()
        self.v_lif = Multispike()
        self.attn_lif = Multispike_att()

        self.proj_conv = nn.Sequential(LpConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

    def forward(self, x):
        x = x.unsqueeze(0)  # Add time dimension
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 1, 3, 2, 4).contiguous()

        k = self.k_lif(k).flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        k = k.permute(0, 1, 3, 2, 4).contiguous()

        v = self.v_lif(v).flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        v = v.permute(0, 1, 3, 2, 4).contiguous()

        x = (q @ (k.transpose(-2, -1) @ v)) * self.scale
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)

        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        x = x.squeeze(0)  # Remove time dimension

        return x
