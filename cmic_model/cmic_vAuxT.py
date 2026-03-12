
from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from compressai.models.sensetime import Elic2022Official, ResidualBottleneckBlock

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers import sequential_channel_ramp, subpel_conv3x3
from compressai.models.utils import conv, deconv

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from .layers import (
    FixedCheckerboardLatentCodec,
    GatedTransformCNN,
    Param_Gated,
    Param_Agg_Block_double,
    OLP, WLS, iWLS

)



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
from .cmic_utils import CMIC_stage  

class Analysis_cmic(nn.Module):
    """Analysis transform g_a with AuxT fusion at multiple scales."""

    def __init__(self, N: int, M: int):
        super().__init__()

        embed_dim0, embed_dim1, embed_dim2 = 128, 192, 256

        self.AuxT_enc = nn.Sequential(
            WLS(3, embed_dim0),
            WLS(embed_dim0, embed_dim1),
            WLS(embed_dim1, embed_dim2),
            WLS(embed_dim2, M),
        )

        # Main branch
        self.g1 = nn.Sequential(
            GatedTransformCNN(embed_dim0, embed_dim0, expansion_factor=3),
            GatedTransformCNN(embed_dim0, embed_dim0, expansion_factor=3),
            GatedTransformCNN(embed_dim0, embed_dim0, expansion_factor=3),
        )
        self.g2 = CMIC_stage(
            dim=embed_dim1,
            d_state=8,
            depth=2,
            num_heads=8,
            window_size=8,
            inner_rank=32,
            cluster_num=64,
            convffn_kernel_size=3,
            mlp_ratio=3,
        )
        self.g3 = CMIC_stage(
            dim=embed_dim2,
            d_state=8,
            depth=2,
            num_heads=8,
            window_size=8,
            inner_rank=32,
            cluster_num=64,
            convffn_kernel_size=3,
            mlp_ratio=3,
        )

        # Downsampling
        self.down0 = nn.Conv2d(3, embed_dim0, 3, stride=2, padding=1)
        self.down1 = nn.Conv2d(embed_dim0, embed_dim1, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(embed_dim1, embed_dim2, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(embed_dim2, M, 3, stride=2, padding=1)

    def forward(self, x: Tensor, draw: bool = False) -> Tensor:
        aux_x = x

        x = self.down0(x)
        x = self.g1(x)
        aux_x = self.AuxT_enc[0](aux_x)
        x = x + aux_x

        x = self.down1(x)
        _, _, h, w = x.shape
        x = self.g2(x, (h, w), draw)
        aux_x = self.AuxT_enc[1](aux_x)
        x = x + aux_x

        x = self.down2(x)
        _, _, h, w = x.shape
        x = self.g3(x, (h, w), draw)
        aux_x = self.AuxT_enc[2](aux_x)
        x = x + aux_x

        x = self.down3(x)
        aux_x = self.AuxT_enc[3](aux_x)
        x = x + aux_x
        return x

class Synthesis_cmic(nn.Module):
    """Synthesis transform g_s with AuxT fusion at multiple scales."""

    def __init__(self, N: int, M: int):
        super().__init__()

        embed_dim1, embed_dim2, embed_dim3 = 128, 192, 256

        self.AuxT_dec = nn.Sequential(
            iWLS(M, embed_dim3),
            iWLS(embed_dim3, embed_dim2),
            iWLS(embed_dim2, embed_dim1),
            iWLS(embed_dim1, 3),
        )

        self.g1 = CMIC_stage(
            dim=embed_dim3,
            d_state=8,
            depth=2,
            num_heads=8,
            window_size=8,
            inner_rank=32,
            cluster_num=64,
            convffn_kernel_size=3,
            mlp_ratio=3,
        )
        self.g2 = CMIC_stage(
            dim=embed_dim2,
            d_state=8,
            depth=2,
            num_heads=8,
            window_size=8,
            inner_rank=32,
            cluster_num=64,
            convffn_kernel_size=3,
            mlp_ratio=3,
        )
        self.g3 = nn.Sequential(
            GatedTransformCNN(embed_dim1, embed_dim1, expansion_factor=3),
            GatedTransformCNN(embed_dim1, embed_dim1, expansion_factor=3),
            GatedTransformCNN(embed_dim1, embed_dim1, expansion_factor=3),
        )

        # Upsampling
        self.up0 = deconv(M, embed_dim3, kernel_size=3)
        self.up1 = deconv(embed_dim3, embed_dim2, kernel_size=3)
        self.up2 = deconv(embed_dim2, embed_dim1, kernel_size=3)
        self.up3 = subpel_conv3x3(embed_dim1, 3, 2)

    def forward(self, x: Tensor) -> Tensor:
        aux_x = x

        x = self.up0(x)
        _, _, h, w = x.shape
        x = self.g1(x, (h, w))
        aux_x = self.AuxT_dec[0](aux_x)
        x = x + aux_x

        x = self.up1(x)
        _, _, h, w = x.shape
        x = self.g2(x, (h, w))
        aux_x = self.AuxT_dec[1](aux_x)
        x = x + aux_x

        x = self.up2(x)
        x = self.g3(x)
        aux_x = self.AuxT_dec[2](aux_x)
        x = x + aux_x

        x = self.up3(x)
        aux_x = self.AuxT_dec[3](aux_x)
        x = x + aux_x
        return x


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class CMIC_AuxT(Elic2022Official):
    def __init__(self, N: int = 192, M: int = 320, groups=None, **kwargs):
        super().__init__(**kwargs)

        self.g_a = Analysis_cmic(N, M)
        self.g_s = Synthesis_cmic(N, M)

        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N, kernel_size=3, stride=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N, kernel_size=3, stride=2),
        )

        h_s = nn.Sequential(
            subpel_conv3x3(N, N, 2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            subpel_conv3x3(N, N, 2),
            GatedTransformCNN(N, N, expansion_factor=2),
            GatedTransformCNN(N, N, expansion_factor=2),
            conv(N, N * 2, kernel_size=3, stride=1),
        )

        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=GatedTransformCNN,
                make_act=lambda: nn.Identity(),
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(1, len(self.groups))
        }

        spatial_context = [
            Param_Agg_Block_double(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        param_aggregation = [
            sequential_channel_ramp(
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=Param_Gated,
                make_act=lambda: nn.Identity(),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        scctx_latent_codec = {
            f"y{k}": FixedCheckerboardLatentCodec(
                latent_codec={"y": GaussianConditionalLatentCodec(quantizer="ste")},
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )


    def ortho_loss(self) -> Tensor:
        loss = sum(m.loss() for m in self.modules() if isinstance(m, OLP))
        return cast(Tensor, loss)

    def forward(self, x: Tensor, draw: bool = False):
        y = self.g_a(x, draw)
        y_out = self.latent_codec(y)
        x_hat = self.g_s(y_out["y_hat"])
        return {"x_hat": x_hat, "likelihoods": y_out["likelihoods"]}
