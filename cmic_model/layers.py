from torch import nn
import torch.nn.functional as F
import torch
from itertools import accumulate
from typing import Any, Dict, List, Mapping, Optional, Tuple
from torch import Tensor

from compressai.layers import (
    CheckerboardMaskedConv2d,
)
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)





############################################################
############################################################
###                Transform                             ###
###                  Utils                               ###
############################################################
############################################################


class GatedFFN(nn.Module):
    def __init__(self, channels, expansion_factor=4):
        super(GatedFFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class Param_Gated(nn.Module):
    def __init__(self,
                 dim, dim_out, expansion_factor=4, **layer_kwargs):
        super().__init__()
        layer_scale = 1e-5
        self.norm2 = LayerNorm2d(dim_out)
        self.mixer = nn.Conv2d(dim, dim_out, kernel_size=1, stride=1)
        self.mlp = GatedFFN(dim_out, expansion_factor=expansion_factor)

    def forward(self, x):
        x = self.mixer(x)
        x = x + self.mlp(self.norm2(x))
        return x


class DepthConv_kernel5(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.01, inplace=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 5, padding=2, groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)

        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity


class GatedTransformCNN(nn.Module):
    def __init__(self,
                 dim, dim_out, expansion_factor=4, **layer_kwargs):
        super().__init__()
        layer_scale = 1e-5
        self.norm2 = LayerNorm2d(dim_out)
        self.mixer = DepthConv_kernel5(dim, dim_out)
        self.mlp = GatedFFN(dim_out, expansion_factor=expansion_factor)

    def forward(self, x):
        x = self.mixer(x)
        x = x + self.mlp(self.norm2(x))
        return x

############################################################
############################################################
###                Entropy                               ###
###                Utils                                 ###
############################################################
############################################################


class Param_Agg_Mask(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = CheckerboardMaskedConv2d(in_ch, in_ch, 5, padding=2, groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)

        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity

class Param_Agg_Block(nn.Module):
    def __init__(self,
                 dim, dim_out, expansion_factor=4, **layer_kwargs):
        super().__init__()
        self.norm2 = LayerNorm2d(dim_out)
        self.mixer = Param_Agg_Mask(dim, dim_out)
        self.mlp = GatedFFN(dim_out, expansion_factor=expansion_factor)

    def forward(self, x):
        x = self.mixer(x)
        x = x + self.mlp( self.norm2(x))
        return x







class Param_Agg_Block_double(nn.Module):
    def __init__(self,
                 dim, dim_out, expansion_factor=4, **layer_kwargs):
        super().__init__()
        self.ly1 = Param_Agg_Block(dim, dim)
        self.ly2 = Param_Agg_Block(dim, dim_out)

    def forward(self, x):
        x = self.ly1(x)
        x = self.ly2(x)
        return x


class FixedCheckerboardLatentCodec(CheckerboardLatentCodec):
    @torch.no_grad()
    def _y_ctx_zero(self, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_ctx = self.context_prediction(y.detach().clone())
        return y.new_zeros(y_ctx.shape)









############################################################
############################################################
###                accelerate                            ###
###                Utils                                 ###
############################################################
############################################################


from .pywave import DWT_2D, IDWT_2D
class OLP(nn.Module):
    """Orthogonal Linear Projection.

    A linear layer with an auxiliary orthogonality regularizer:
        || W W^T - I ||_F^2  (or W^T W depending on shape).
    """

    def __init__(self, in_features: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_dim, bias=bias)
        self.in_dim = in_features
        self.out_dim = out_dim

        eye_dim = min(in_features, out_dim)
        self.register_buffer("identity_matrix", torch.eye(eye_dim), persistent=False)

    def loss(self) -> Tensor:
        W = self.linear.weight
        gram = W @ W.t() if self.in_dim > self.out_dim else W.t() @ W
        return F.mse_loss(gram, self.identity_matrix.to(gram.device))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class WLS(nn.Module):
    """Wavelet Linear Scaling (analysis side).

    Applies DWT, scales subbands, then projects with OLP.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.dwt = DWT_2D(wave="haar")
        self.OLP = OLP(in_dim * 4, out_dim)

        factors = torch.cat(
            (
                torch.zeros(1, 1, in_dim) + 0.5,
                torch.zeros(1, 1, in_dim) + 0.5,
                torch.zeros(1, 1, in_dim) + 0.5,
                torch.zeros(1, 1, in_dim),
            ),
            dim=2,
        )
        self.scaling_factors = nn.Parameter(factors)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dwt(x)
        b, _, h, w = x.shape
        x = x.view(b, -1, h * w).permute(0, 2, 1)  # (B, HW, 4C)
        x = x * torch.exp(self.scaling_factors)
        x = self.OLP(x)
        return x.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


class iWLS(nn.Module):
    """Inverse WLS (synthesis side)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.idwt = IDWT_2D(wave="haar")
        self.OLP = OLP(in_dim, out_dim * 4)

        factors = torch.cat(
            (
                torch.zeros(1, 1, out_dim) + 0.5,
                torch.zeros(1, 1, out_dim) + 0.5,
                torch.zeros(1, 1, out_dim) + 0.5,
                torch.zeros(1, 1, out_dim),
            ),
            dim=2,
        )
        self.scaling_factors = nn.Parameter(factors)

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        x = x.view(b, -1, h * w).permute(0, 2, 1)  # (B, HW, C)
        x = self.OLP(x)
        x = x / torch.exp(self.scaling_factors)
        x = x.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return self.idwt(x)
