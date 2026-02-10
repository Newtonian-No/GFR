import torch
import math
from functools import partial
from typing import Callable, Any

import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
try:
    import selective_scan_cuda_core
    import selective_scan_cuda_oflex
    import selective_scan_cuda_ndstate
    import selective_scan_cuda_nrow
    import selective_scan_cuda
except:
    pass

try:
    "sscore acts the same as mamba_ssm"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# ==================== Bidirectional Scan (双向) K=2 ====================
class BiScan(torch.autograd.Function):
    """双向扫描: 行方向正序 + 逆序"""
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 2, C, H * W))
        xs[:, 0] = x.flatten(2, 3)  # 正向: 从左到右, 从上到下
        xs[:, 1] = torch.flip(x.flatten(2, 3), dims=[-1])  # 反向
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1])
        return y.view(B, -1, H, W)


class BiMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1])
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.flip(dims=[-1])
        return xs.view(B, 2, C, H, W), None, None


# ==================== Cross Scan (十字) K=4 ====================
class CrossScan(torch.autograd.Function):
    """十字扫描: 行/列方向 + 正逆序, 共4方向"""
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


# ==================== Spiral/Ring Scan (回形) K=2 ====================
class SpiralScan(torch.autograd.Function):
    """回形扫描: 蛇形(之字形)顺序 - 奇数行反向, 形成回字形遍历"""
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        
        # 复制一份以防修改原图
        x_spiral = x.clone() 
        # 蛇形变换：奇数行翻转 (Index从0开始，1, 3, 5...行是奇数行)
        x_spiral[:, :, 1::2, :] = x_spiral[:, :, 1::2, :].flip(dims=[-1])
        
        L = H * W
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x_spiral.view(B, C, L)
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        # 1. 合并两个方向
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1]) 
        y = y.view(B, C, H, W)
        # 2. 逆蛇形变换：把奇数行再翻转回来
        y[:, :, 1::2, :] = y[:, :, 1::2, :].flip(dims=[-1])
        return y.contiguous()


class SpiralMerge(torch.autograd.Function):
    """回形 merge: 将蛇形顺序的 SSM 输出还原为行优先, 以配合 view(B,H,W,-1)"""
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1])  # (B, D, L) 蛇形顺序
        # 蛇形 -> 行优先, 供后续 view(B,H,W,-1) 使用
        out = y.new_empty((B, D, H * W))
        for h in range(H):
            start = h * W
            if h % 2 == 0:
                out[:, :, start:start + W] = y[:, :, start:start + W]
            else:
                out[:, :, start:start + W] = y[:, :, start + W - 1:start - 1:-1]
        return out  # (B, D, L) 行优先

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        # x 是行优先, 需转为蛇形顺序以匹配 ys
        y = x.new_empty((B, C, L))
        for h in range(H):
            start = h * W
            if h % 2 == 0:
                y[:, :, start:start + W] = x[:, :, start:start + W]
            else:
                y[:, :, start:start + W] = x[:, :, start + W - 1:start - 1:-1]
        xs = y.new_empty((B, 2, C, L))
        xs[:, 0] = y
        xs[:, 1] = y.flip(dims=[-1])
        return xs.view(B, 2, C, H, W), None, None


# selective scan (support multiple scan modes) ===============================
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


# Scan mode: 'bidirectional'(K=2), 'cross'(K=4), 'spiral'(K=2)
SCAN_MODE_MAP = {
    'bidirectional': ('BiScan', 'BiMerge', 2),
    'cross': ('CrossScan', 'CrossMerge', 4),
    'spiral': ('SpiralScan', 'SpiralMerge', 2),
}


def selective_scan_by_mode(
        x: torch.Tensor,
        x_proj_weight: torch.Tensor,
        x_proj_bias: torch.Tensor,
        dt_projs_weight: torch.Tensor,
        dt_projs_bias: torch.Tensor,
        A_logs: torch.Tensor,
        Ds: torch.Tensor,
        out_norm: torch.nn.Module,
        out_norm_shape: str,
        scan_mode: str = 'cross',
        nrows: int = -1,
        backnrows: int = -1,
        delta_softplus: bool = True,
        to_dtype: bool = True,
        force_fp32: bool = False,
        ssoflex: bool = True,
        SelectiveScan=None,
):
    """支持多种扫描模式的 selective scan: bidirectional/cross/spiral"""
    scan_name, merge_name, expected_k = SCAN_MODE_MAP.get(scan_mode, ('CrossScan', 'CrossMerge', 4))
    ScanCls = globals()[scan_name]
    MergeCls = globals()[merge_name]

    B, D, H, W = x.shape
    D_dim, N = A_logs.shape
    K, D_dim, R = dt_projs_weight.shape
    L = H * W

    def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = ScanCls.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys = selective_scan_fn(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, H, W)
    y = MergeCls.apply(ys)

    if out_norm_shape in ["v1"]:
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)
    else:
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        nrows=-1,
        backnrows=-1,
        delta_softplus=True,
        to_dtype=True,
        force_fp32=False,
        ssoflex=True,
        SelectiveScan=None,
        scan_mode_type='cross',
):
    """原始接口, 默认十字扫描。scan_mode_type 支持: bidirectional, cross, spiral"""
    return selective_scan_by_mode(
        x, x_proj_weight, x_proj_bias, dt_projs_weight, dt_projs_bias,
        A_logs, Ds, out_norm, out_norm_shape,
        scan_mode=scan_mode_type,
        nrows=nrows, backnrows=backnrows,
        delta_softplus=delta_softplus, to_dtype=to_dtype,
        force_fp32=force_fp32, ssoflex=ssoflex,
        SelectiveScan=SelectiveScan,
    )