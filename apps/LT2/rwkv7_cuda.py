"""Optional CUDA backend for LT2's RWKV-7 mixer.

The kernels are vendored from BlinkDL/RWKV-LM RWKV-v7/train_temp. They are
loaded lazily so CPU tests and machines without nvcc can still use the pure
PyTorch fallback in ``transformer.py``.
"""

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


_CUDA_DIR = Path(__file__).resolve().parent / "cuda" / "rwkv7"
_CFLAGS = ["-O3"]
_CUDA_FLAGS = [
    "-res-usage",
    "--use_fast_math",
    "-O3",
    "-Xptxas",
    "-O3",
    "--extra-device-vectorization",
]


def _verbose() -> bool:
    return os.environ.get("LT2_RWKV7_CUDA_VERBOSE", "0") == "1"


def _ensure_cuda_buildable() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit was not found; install nvcc and set CUDA_HOME")


def _load_extension(name: str, sources, extra_cuda_flags=None, is_python_module: bool = False):
    _ensure_cuda_buildable()
    paths = [_CUDA_DIR / source for source in sources]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing RWKV-7 CUDA source file(s): {', '.join(missing)}")
    return load(
        name=name,
        sources=[str(path) for path in paths],
        extra_cflags=_CFLAGS,
        extra_cuda_cflags=_CUDA_FLAGS + list(extra_cuda_flags or []),
        is_python_module=is_python_module,
        verbose=_verbose(),
    )


@lru_cache(maxsize=None)
def load_rwkv7_sequence_kernel(head_size: int, chunk_len: int = 16) -> None:
    if head_size != 64:
        raise RuntimeError("The optimized RWKV-7 bf16 sequence kernel currently requires head_size=64")
    if hasattr(torch.ops, "rwkv7_clampw") and hasattr(torch.ops.rwkv7_clampw, "forward"):
        return
    _load_extension(
        name=f"lt2_rwkv7_clampw_h{head_size}_c{chunk_len}",
        sources=["rwkv7_clampw.cu", "rwkv7_clampw.cpp"],
        extra_cuda_flags=[f"-D_N_={head_size}", f"-D_CHUNK_LEN_={chunk_len}"],
        is_python_module=False,
    )


@lru_cache(maxsize=None)
def load_tmix_kernels() -> None:
    specs = [
        ("lt2_rwkv7_tmix_mix6_bf16_v5", ["rwkv7_tmix_mix6_bf16_v5.cpp", "rwkv7_tmix_mix6_bf16_v5.cu"]),
        ("lt2_rwkv7_tmix_kk_pre_bf16_v5", ["rwkv7_tmix_kk_pre_bf16_v5.cpp", "rwkv7_tmix_kk_pre_bf16_v5.cu"]),
        (
            "lt2_rwkv7_tmix_lnx_rkvres_xg_bf16_v1",
            ["rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cpp", "rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cu"],
        ),
        ("lt2_rwkv7_tmix_a_gate_bf16", ["rwkv7_tmix_a_gate_bf16.cpp", "rwkv7_tmix_a_gate_bf16.cu"]),
        (
            "lt2_rwkv7_tmix_vres_gate_bf16_v1",
            ["rwkv7_tmix_vres_gate_bf16_v1.cpp", "rwkv7_tmix_vres_gate_bf16_v1.cu"],
        ),
    ]
    for name, sources in specs:
        _load_extension(name=name, sources=sources, is_python_module=False)


@lru_cache(maxsize=None)
def load_cmix_kernel() -> None:
    _load_extension(
        name="lt2_rwkv7_cmix_bf16_v5",
        sources=["rwkv7_cmix_bf16_v5.cpp", "rwkv7_cmix_bf16_v5.cu"],
        is_python_module=False,
    )


@lru_cache(maxsize=None)
def load_l2wrap_ce_kernel():
    return _load_extension(
        name="lt2_rwkv7_l2wrap_ce_bf16_v2",
        sources=["rwkv7_l2wrap_ce_bf16_v2.cpp", "rwkv7_l2wrap_ce_bf16_v2.cu"],
        is_python_module=True,
    )


def pad_to_chunk(x: torch.Tensor, chunk_len: int) -> Tuple[torch.Tensor, int]:
    pad_len = (-x.size(1)) % chunk_len
    if pad_len == 0:
        return x, 0
    return torch.nn.functional.pad(x, (0, 0, 0, pad_len)), pad_len


class RWKV7ClampwFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b, head_size: int, chunk_len: int):
        load_rwkv7_sequence_kernel(head_size, chunk_len)
        B, T, C = r.shape
        H = C // head_size
        r, w, k, v, a, b = [
            tensor.reshape(B, T, H, head_size).contiguous()
            for tensor in (r, w, k, v, a, b)
        ]
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // chunk_len, head_size, head_size, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, head_size, dtype=torch.float32, device=w.device)
        torch.ops.rwkv7_clampw.forward(r, w, k, v, a, b, y, s, sa)
        ctx.save_for_backward(r, w, k, v, a, b, s, sa)
        ctx.shape = (B, T, C)
        return y.reshape(B, T, C)

    @staticmethod
    def backward(ctx, dy):
        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        dy = dy.reshape_as(v).contiguous()
        dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in (r, w, k, v, a, b)]
        torch.ops.rwkv7_clampw.backward(r, w, k, v, a, b, dy, s, sa, dr, dw, dk, dv, da, db)
        B, T, C = ctx.shape
        return (
            dr.reshape(B, T, C),
            dw.reshape(B, T, C),
            dk.reshape(B, T, C),
            dv.reshape(B, T, C),
            da.reshape(B, T, C),
            db.reshape(B, T, C),
            None,
            None,
        )


def rwkv7_recurrence_cuda_bf16(r, w, k, v, a, b, head_size: int, chunk_len: int = 16):
    original_t = r.size(1)
    padded = []
    for tensor in (r, w, k, v, a, b):
        tensor, _ = pad_to_chunk(tensor.contiguous(), chunk_len)
        padded.append(tensor)
    out = RWKV7ClampwFn.apply(*padded, head_size, chunk_len)
    return out[:, :original_t]


class TmixMix6Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_r, x_w, x_k, x_v, x_a, x_g):
        load_tmix_kernels()
        ctx.save_for_backward(x, x_r, x_w, x_k, x_v, x_a, x_g)
        outs = torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(
            x.contiguous(),
            x_r.contiguous(),
            x_w.contiguous(),
            x_k.contiguous(),
            x_v.contiguous(),
            x_a.contiguous(),
            x_g.contiguous(),
        )
        return tuple(outs)

    @staticmethod
    def backward(ctx, grad_r, grad_w, grad_k, grad_v, grad_a, grad_g):
        outs = torch.ops.rwkv7_tmix_mix6_bf16_v5.backward(
            grad_r.contiguous(),
            grad_w.contiguous(),
            grad_k.contiguous(),
            grad_v.contiguous(),
            grad_a.contiguous(),
            grad_g.contiguous(),
            *ctx.saved_tensors,
        )
        return tuple(outs)


def tmix_mix6(x, x_r, x_w, x_k, x_v, x_a, x_g):
    return TmixMix6Fn.apply(x, x_r, x_w, x_k, x_v, x_a, x_g)


class TmixAGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a0, a12):
        load_tmix_kernels()
        ctx.save_for_backward(a0, a12)
        return torch.ops.rwkv7_tmix_a_gate_bf16.forward(a0.contiguous(), a12.contiguous())

    @staticmethod
    def backward(ctx, grad_out):
        a0, a12 = ctx.saved_tensors
        return tuple(torch.ops.rwkv7_tmix_a_gate_bf16.backward(grad_out.contiguous(), a0, a12))


def tmix_a_gate(a0, a12):
    return TmixAGateFn.apply(a0, a12)


class TmixVresGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_first, v0, v12):
        load_tmix_kernels()
        ctx.save_for_backward(v, v_first, v0, v12)
        return torch.ops.rwkv7_tmix_vres_gate_bf16_v1.forward(
            v.contiguous(),
            v_first.contiguous(),
            v0.contiguous(),
            v12.contiguous(),
        )

    @staticmethod
    def backward(ctx, grad_out):
        v, v_first, v0, v12 = ctx.saved_tensors
        grad_v, grad_v_first, grad_pre = torch.ops.rwkv7_tmix_vres_gate_bf16_v1.backward(
            grad_out.contiguous(),
            v,
            v_first,
            v0,
            v12,
        )
        grad_v0 = grad_pre.sum(dim=(0, 1))
        return grad_v, grad_v_first, grad_v0.to(v0.dtype), grad_pre.to(v12.dtype)


def tmix_vres_gate(v, v_first, v0, v12):
    return TmixVresGateFn.apply(v, v_first, v0, v12)


class TmixKkPreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, k_k, a, k_a, head_size: int):
        load_tmix_kernels()
        outs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(
            k.contiguous(),
            k_k.contiguous(),
            a.contiguous(),
            k_a.contiguous(),
            head_size,
        )
        ctx.save_for_backward(k, k_k, a, k_a, outs[3])
        ctx.head_size = head_size
        return outs[0], outs[1], outs[2]

    @staticmethod
    def backward(ctx, grad_new_k, grad_neg_kk, grad_kka):
        k, k_k, a, k_a, inv_d = ctx.saved_tensors
        outs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.backward(
            grad_new_k.contiguous(),
            grad_neg_kk.contiguous(),
            grad_kka.contiguous(),
            k,
            k_k,
            a,
            k_a,
            inv_d,
            ctx.head_size,
        )
        return outs[0], outs[1], outs[2], outs[3], None


def tmix_kk_pre(k, k_k, a, k_a, head_size: int):
    return TmixKkPreFn.apply(k, k_k, a, k_a, head_size)


class TmixLnxRkvresXgFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, r, k, v, r_k, weight, bias, g):
        load_tmix_kernels()
        outs = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.forward(
            x.contiguous(),
            r.contiguous(),
            k.contiguous(),
            v.contiguous(),
            r_k.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            g.contiguous(),
        )
        ctx.save_for_backward(x, r, k, v, r_k, weight, bias, g, outs[1], outs[2])
        return outs[0]

    @staticmethod
    def backward(ctx, grad_xg):
        x, r, k, v, r_k, weight, bias, g, mean, rstd = ctx.saved_tensors
        outs = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.backward(
            grad_xg.contiguous(),
            x,
            r,
            k,
            v,
            r_k,
            weight,
            bias,
            g,
            mean,
            rstd,
        )
        return tuple(outs)


def tmix_lnx_rkvres_xg(x, r, k, v, r_k, weight, bias, g):
    return TmixLnxRkvresXgFn.apply(x, r, k, v, r_k, weight, bias, g)


class CmixLayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_k, key_weight, value_weight):
        load_cmix_kernel()
        out, mixed, act = torch.ops.rwkv7_cmix_bf16_v5.forward(
            x.contiguous(),
            x_k.contiguous(),
            key_weight.contiguous(),
            value_weight.contiguous(),
        )
        ctx.save_for_backward(x, x_k, key_weight, value_weight, mixed, act)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, x_k, key_weight, value_weight, mixed, act = ctx.saved_tensors
        grad_x, grad_x_k, grad_key_weight, grad_value_weight = torch.ops.rwkv7_cmix_bf16_v5.backward(
            grad_out.contiguous(),
            x,
            x_k,
            key_weight,
            value_weight,
            mixed,
            act,
        )
        return grad_x, grad_x_k, grad_key_weight, grad_value_weight


def cmix_layer(x, x_k, key_weight, value_weight):
    return CmixLayerFn.apply(x, x_k, key_weight, value_weight)


def l2wrap_cross_entropy(logits, targets):
    module = load_l2wrap_ce_kernel()
    logits = logits.contiguous()
    targets = targets.contiguous()

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits_, targets_):
            loss, lse, max_vals, argmax = module.forward(logits_, targets_)
            ctx.save_for_backward(logits_, targets_.reshape(-1), lse, max_vals, argmax)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            logits_, targets_, lse, max_vals, argmax = ctx.saved_tensors
            grad_logits = module.backward(
                grad_output.contiguous().float(),
                logits_,
                targets_,
                lse,
                max_vals,
                argmax,
            )
            return grad_logits, None

    return _Fn.apply(logits, targets)


def warn_fast_backend_failure(exc: Exception) -> None:
    warnings.warn(
        f"Falling back to pure PyTorch RWKV-7 backend because CUDA kernels are unavailable: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )
