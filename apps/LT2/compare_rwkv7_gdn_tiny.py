"""Compare LT2 RWKV-7 and GDN on identical synthetic token batches.

This is a fast local ablation harness, not a substitute for full FineWeb
training. It keeps the model shape, random batches, optimizer, and token budget
fixed while changing only ``model.layer_pattern``.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.LT2.transformer import LoopedWindowTransformer, LoopedWindowTransformerArgs


def mixer_ffn_multiplier(parsed: argparse.Namespace, mixer: str) -> float:
    overrides = {
        "gdn": parsed.gdn_ffn_dim_multiplier,
        "rwkv7": parsed.rwkv7_ffn_dim_multiplier,
        "rwkv7_native": parsed.rwkv7_native_ffn_dim_multiplier,
    }
    value = overrides.get(mixer)
    return parsed.ffn_dim_multiplier if value is None else value


def build_model_args(parsed: argparse.Namespace, mixer: str) -> LoopedWindowTransformerArgs:
    return LoopedWindowTransformerArgs(
        dim=parsed.dim,
        n_layers=parsed.n_layers,
        n_heads=parsed.n_heads,
        n_kv_heads=parsed.n_heads,
        vocab_size=parsed.vocab_size,
        max_seqlen=parsed.seq_len,
        loop_count=parsed.loop_count,
        layer_pattern=mixer,
        attention_pattern="1:0",
        default_sliding_window=parsed.seq_len,
        rwkv7_head_size=parsed.rwkv7_head_size,
        rwkv7_enable_v_first_mix=True,
        rwkv7_backend=parsed.rwkv7_backend,
        rwkv7_chunk_len=parsed.rwkv7_chunk_len,
        rwkv7_use_l2wrap_ce=parsed.rwkv7_use_l2wrap_ce,
        gdn_allow_neg_eigval=parsed.gdn_allow_neg_eigval,
        ffn_dim_multiplier=mixer_ffn_multiplier(parsed, mixer),
        multiple_of=parsed.multiple_of,
        weight_tying=True,
        use_residual=True,
        use_block_residual=True,
        attn_impl="sdpa",
    )


def count_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def grad_norm(model: torch.nn.Module) -> float:
    grad_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm_sq += float(param.grad.detach().float().pow(2).sum().item())
    return math.sqrt(grad_norm_sq)


def make_batches(parsed: argparse.Namespace, device: torch.device) -> List[torch.Tensor]:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(parsed.data_seed)
    batches = []
    for _ in range(parsed.warmup_steps + parsed.steps):
        batches.append(
            torch.randint(
                0,
                parsed.vocab_size,
                (parsed.batch_size, parsed.seq_len),
                generator=generator,
                device=device,
            )
        )
    return batches


def run_mixer(parsed: argparse.Namespace, mixer: str, batches: List[torch.Tensor]) -> Dict[str, object]:
    torch.manual_seed(parsed.model_seed)
    device = torch.device(parsed.device)
    model = LoopedWindowTransformer(build_model_args(parsed, mixer)).to(device)
    model.init_weights()
    if parsed.dtype == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif parsed.dtype == "fp32":
        model = model.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported dtype: {parsed.dtype}")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=parsed.lr, weight_decay=parsed.weight_decay)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    losses = []
    grad_norms = []
    start = None
    for step, x in enumerate(batches):
        y = torch.roll(x, shifts=-1, dims=1)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y, attn_impl="sdpa")
        if not torch.isfinite(loss):
            raise RuntimeError(f"{mixer}: non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        norm = grad_norm(model)
        if not math.isfinite(norm) or norm == 0.0:
            raise RuntimeError(f"{mixer}: invalid gradient norm at step {step}: {norm}")
        optimizer.step()
        if step + 1 == parsed.warmup_steps and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        if step >= parsed.warmup_steps:
            if start is None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(norm)
        if parsed.verbose:
            phase = "warmup" if step < parsed.warmup_steps else "timed"
            print(f"{mixer} {phase}_step={step} loss={loss.item():.6f} grad_norm={norm:.6f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_memory_mb = None
    if start is None:
        raise RuntimeError("No timed steps were run; set --steps > 0")
    elapsed = time.perf_counter() - start
    tokens = parsed.steps * parsed.batch_size * parsed.seq_len
    return {
        "mixer": mixer,
        "params": count_params(model),
        "ffn_dim_multiplier": mixer_ffn_multiplier(parsed, mixer),
        "warmup_steps": parsed.warmup_steps,
        "steps": parsed.steps,
        "tokens": tokens,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "loss_delta": losses[-1] - losses[0],
        "mean_loss": sum(losses) / len(losses),
        "last_grad_norm": grad_norms[-1],
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "peak_memory_mb": peak_memory_mb,
        "losses": losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LT2 RWKV-7 and GDN on synthetic batches.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--mixers", default="rwkv7,gdn")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--loop-count", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--rwkv7-head-size", type=int, default=32)
    parser.add_argument("--rwkv7-backend", default="auto", choices=["auto", "cuda", "torch"])
    parser.add_argument("--rwkv7-chunk-len", type=int, default=16)
    parser.add_argument("--rwkv7-use-l2wrap-ce", action="store_true")
    parser.add_argument("--ffn-dim-multiplier", type=float, default=0.5)
    parser.add_argument("--gdn-ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--rwkv7-ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--rwkv7-native-ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--multiple-of", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--model-seed", type=int, default=1234)
    parser.add_argument("--data-seed", type=int, default=5678)
    parser.add_argument("--gdn-allow-neg-eigval", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--verbose", action="store_true")
    parsed = parser.parse_args()

    if parsed.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    device = torch.device(parsed.device)
    batches = make_batches(parsed, device)
    mixers = [m.strip().lower() for m in parsed.mixers.split(",") if m.strip()]
    results = []
    for mixer in mixers:
        results.append(run_mixer(parsed, mixer, batches))

    print("mixer,params,ffn_dim_multiplier,tokens_per_second,first_loss,last_loss,loss_delta,mean_loss,last_grad_norm,peak_memory_mb")
    for result in results:
        peak = "" if result["peak_memory_mb"] is None else f"{result['peak_memory_mb']:.2f}"
        print(
            f"{result['mixer']},{result['params']},{result['ffn_dim_multiplier']:.6f},"
            f"{result['tokens_per_second']:.2f},{result['first_loss']:.6f},{result['last_loss']:.6f},{result['loss_delta']:.6f},"
            f"{result['mean_loss']:.6f},{result['last_grad_norm']:.6f},{peak}"
        )

    if parsed.json_out:
        payload = {"args": vars(parsed), "results": results}
        Path(parsed.json_out).write_text(json.dumps(payload, indent=2) + "\n")

    print("compare_rwkv7_gdn_tiny_ok")


if __name__ == "__main__":
    main()
