from __future__ import annotations

import math
from inspect import isfunction
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from tqdm import tqdm 


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def exists(val) -> bool:
    return val is not None


def is_empty(t: torch.Tensor) -> bool:
    return t.nelement() == 0


def expand_dim(t: torch.Tensor, dim: int, k: int) -> torch.Tensor:
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def default(x, d):
    return d() if (not exists(x) and isfunction(d)) else (d if not exists(x) else x)


def ema(old: Optional[torch.Tensor], new: torch.Tensor, decay: float) -> torch.Tensor:
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    """In-place EMA update for buffers."""
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def similarity(x: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """Cosine-like similarity assuming x/means are already normalized."""
    return torch.einsum("bld,cd->blc", x, means)


def dists_and_buckets(x: torch.Tensor, means: torch.Tensor):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets


def batched_bincount(index: torch.Tensor, num_classes: int, dim: int = -1) -> torch.Tensor:
    """Count per-class assignments for a batched index tensor."""
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def center_iter(x: torch.Tensor, means: torch.Tensor, buckets: Optional[torch.Tensor] = None) -> torch.Tensor:
    """One iteration of centroid re-estimation (k-means style) with safe empty handling.

    Shapes:
        x:     (B, L, D)
        means: (K, D)
        buckets (optional): (B, L) hard assignment in [0, K-1]
    """
    b, l, d = x.shape
    dtype = x.dtype
    cluster_num = means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, cluster_num).sum(0, keepdim=True)  # (1, K)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, cluster_num, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)

    means_ = means_.sum(0) / bins.clamp(min=1).squeeze(0).unsqueeze(-1)  # (K, D)
    means_ = F.normalize(means_, dim=-1).type(dtype)

    means = torch.where(zero_mask.squeeze(0).unsqueeze(-1), means, means_)
    return means


def index_reverse(index: torch.Tensor) -> torch.Tensor:
    """Compute inverse permutation for each batch row.
    """
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1], device=index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def apply_permute(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather x according to index along the token dimension.
    """
    dim = index.dim()
    assert x.shape[:dim] == index.shape, f"x ({x.shape}) and index ({index.shape}) shape incompatible"

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    return torch.gather(x, dim=dim - 1, index=index)



class GatedCNNFFN(nn.Module):
    def __init__(self, channels: int, expansion_factor: float):
        super().__init__()
        self.hidden_features = channels
        hidden_channels = int(channels * expansion_factor)

        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, h, w).contiguous()
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x.flatten(2).transpose(1, 2).contiguous()



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """x: (B, H, W, C) -> windows: (num_windows*B, ws, ws, C)"""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """windows: (num_windows*B, ws, ws, C) -> x: (B, H, W, C)"""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv: torch.Tensor, rpi=None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}"


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class TokenPermutation:
    def __init__(self, stable: bool = True):
        self.stable = stable

    def sort_indices(self, cluster_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cluster_index: (B, N) integer cluster id per token
        Returns:
            sort_idx: (B, N) indices that group tokens by cluster id
            inv_idx:  (B, N) inverse permutation so that x == gather(gather(x, sort_idx), inv_idx)
        """
        _, sort_idx = torch.sort(cluster_index, dim=-1, stable=self.stable)
        inv_idx = index_reverse(sort_idx)
        return sort_idx, inv_idx


class PromptingOps:
    def __init__(self, cluster_num: int, ema_decay: float, n_iter: int):
        self.cluster_num = cluster_num
        self.ema_decay = ema_decay
        self.n_iter = n_iter

    def update_centroids(
        self,
        x: torch.Tensor,                 # (B, N, C)
        means: torch.Tensor,             # (K, C) buffer
        initted: torch.Tensor,           # () bool buffer
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_means:   (K, C) detached centroids used for this forward
            new_initted: bool tensor (same object usually updated in-place by caller)
        """
        B, N, C = x.shape
        K = self.cluster_num

        if not bool(initted.item()):
            pad_n = (K - N % K) % K
            padded_x = F.pad(x, (0, 0, 0, pad_n))
            # (K, B * (N/K), C) -> mean over batch/tokens per group
            x_means = torch.mean(rearrange(padded_x, "b (cnt n) c -> cnt (b n) c", cnt=K), dim=-2).detach()
        else:
            x_means = means.detach()

        if training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))

        x_means = x_means.detach()

        if training:
            with torch.no_grad():
                new_means = x_means
                if not bool(initted.item()):
                    means.data.copy_(new_means)
                    initted.data.copy_(torch.tensor(True, device=initted.device))
                else:
                    ema_inplace(means, new_means, self.ema_decay)

        return x_means, initted

    @staticmethod
    def hard_assign(x: torch.Tensor, x_means: torch.Tensor) -> torch.Tensor:
        """Compute hard cluster index: (B,N)"""
        with torch.no_grad():
            scores = torch.einsum("b i c, j c -> b i j", F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
            return torch.argmax(scores, dim=-1)


class CAM(nn.Module):

    def __init__(
        self,
        dim: int,
        d_state: int,
        cluster_num: int = 64,
        inner_rank: int = 128,
        mlp_ratio: float = 2.0,
        n_iter: int = 5,
        ema_decay: float = 0.999,
        permute_stable: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.cluster_num = cluster_num
        self.inner_rank = inner_rank
        self.d_state = d_state

        # Permutation / prompting helpers
        self.permutation = TokenPermutation(stable=permute_stable)
        self.prompting = PromptingOps(cluster_num=cluster_num, ema_decay=ema_decay, n_iter=n_iter)

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(nn.Conv2d(self.dim, hidden, 1, 1, 0))
        self.CPE = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden))

        self.register_buffer("means", torch.randn(cluster_num, dim))
        self.register_buffer("initted", torch.tensor(False))

        self.cal_embedding = nn.Linear(dim, d_state)


    def forward(
        self,
        x: torch.Tensor,                       # (B, N, C)
        x_size: Tuple[int, int],               # (H, W)

    ) -> torch.Tensor:
        B, N, C = x.shape
        H, W = x_size

        # -------- Prompting: centroid update + hard assignment --------
        x2d = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x_tokens = x2d.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, N, C)

        x_means, _ = self.prompting.update_centroids(
            x=x_tokens,
            means=self.means,
            initted=self.initted,
            training=self.training,
        )
        cluster_idx = self.prompting.hard_assign(x_tokens, x_means)  # (B, N)
        cls_policy = F.one_hot(cluster_idx, num_classes=self.cluster_num).float()  # (B, N, K)


        full_embedding = self.cal_embedding(x_means)      # (K, d_state)
        prompt = torch.matmul(cls_policy, full_embedding) # (B, N, d_state)
        
        # -------- Permutation--------
        sort_idx, inv_idx = self.permutation.sort_indices(cluster_idx)

        # -------- Mamba scan--------
        x_m = x_tokens.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x_m = self.in_proj(x_m)
        x_m = x_m * torch.sigmoid(self.CPE(x_m))
        cc = x_m.shape[1]
        x_m = x_m.view(B, cc, -1).permute(0, 2, 1).contiguous()  # (B, N, hidden)

        semantic_x = apply_permute(x_m, sort_idx)  
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))
        x_out = apply_permute(y, inv_idx)        
        return x_out

class Selective_Scan(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: float = 2.0,
        dt_rank="auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        **factory_kwargs,
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True  
        return dt_proj

    @staticmethod
    def A_log_init(d_state: int, d_inner: int, copies: int = 1, device=None, merge: bool = True) -> nn.Parameter:
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True  
        return A_log

    @staticmethod
    def D_init(d_inner: int, copies: int = 1, device=None, merge: bool = True) -> nn.Parameter:
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True  
        return D

    def forward_core(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """x: (B, L, hidden), prompt: (B, 1, d_state, L)"""
        B, L, C = x.shape
        K = 1

        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) + prompt  # inject prompt into C

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt: torch.Tensor, **kwargs) -> torch.Tensor:
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)  # (B, hidden, L)
        return y.permute(0, 2, 1).contiguous()  # (B, L, hidden)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class CMIC_block(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        inner_rank: int,
        cluster_num: int,
        convffn_kernel_size: int,
        mlp_ratio: float,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)


        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.win_mhsa = WindowAttention(dim, window_size=to_2tuple(window_size), num_heads=num_heads, qkv_bias=qkv_bias)


        self.assm = CAM(dim, d_state, cluster_num=cluster_num, inner_rank=inner_rank, mlp_ratio=mlp_ratio)

        self.convffn1 = GatedCNNFFN(dim, expansion_factor=mlp_ratio)
        self.convffn2 = GatedCNNFFN(dim, expansion_factor=mlp_ratio)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int], params=None, draw: bool = False) -> torch.Tensor:
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        # (1) Window Attention
        shortcut = x
        x_norm = self.norm1(x)
        qkv = self.wqkv(x_norm).reshape(b, h, w, c3)

        x_windows = window_partition(qkv, self.window_size).view(-1, self.window_size * self.window_size, c3)
        attn_windows = self.win_mhsa(x_windows, rpi=None, mask=None).view(-1, self.window_size, self.window_size, c)
        attn_x = window_reverse(attn_windows, self.window_size, h, w)

        x = attn_x.view(b, n, c) + shortcut
        x = self.convffn1(self.norm2(x), x_size) + x

        # (2) Content-adaptive Mamba
        shortcut = x
        x = self.assm(self.norm3(x), x_size,) + shortcut
        x = self.convffn2(self.norm4(x), x_size) + x
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        depth: int,
        num_heads: int,
        window_size: int,
        inner_rank: int,
        cluster_num: int,
        convffn_kernel_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                CMIC_block(
                    dim=dim,
                    d_state=d_state,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0,
                    inner_rank=inner_rank,
                    cluster_num=cluster_num,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int], params=None, draw: bool = False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, x_size, params, draw)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x) if self.norm is not None else x


class PatchUnEmbed(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, h, w)


class CMIC_stage(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        depth: int,
        num_heads: int,
        window_size: int,
        inner_rank: int,
        cluster_num: int,
        convffn_kernel_size: int,
        mlp_ratio: float,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.patch_embed = PatchEmbed(in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim,
            d_state=d_state,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            cluster_num=cluster_num,
            inner_rank=inner_rank,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int], draw: bool = False) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = self.residual_group(x, x_size,)
        return self.patch_unembed(x, x_size)

