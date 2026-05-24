"""Tiny synthetic training smoke test for LT2 + RWKV-7.

This intentionally avoids the full distributed LT2 training stack and dataset.
It validates model construction, init, forward, backward, and optimizer steps
for ``layer_pattern="rwkv7"`` on random token sequences.
"""

import argparse
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.LT2.transformer import LoopedWindowTransformer, LoopedWindowTransformerArgs


def build_args(args: argparse.Namespace) -> LoopedWindowTransformerArgs:
    return LoopedWindowTransformerArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_heads,
        vocab_size=args.vocab_size,
        max_seqlen=args.seq_len,
        loop_count=args.loop_count,
        layer_pattern="rwkv7",
        attention_pattern="1:0",
        default_sliding_window=args.seq_len,
        rwkv7_head_size=args.rwkv7_head_size,
        rwkv7_enable_v_first_mix=True,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        multiple_of=args.multiple_of,
        weight_tying=True,
        use_residual=True,
        use_block_residual=True,
        attn_impl="sdpa",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny LT2 RWKV-7 synthetic train smoke test.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--loop-count", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--rwkv7-head-size", type=int, default=32)
    parser.add_argument("--ffn-dim-multiplier", type=float, default=0.5)
    parser.add_argument("--multiple-of", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parsed = parser.parse_args()

    if parsed.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(parsed.seed)
    device = torch.device(parsed.device)
    model = LoopedWindowTransformer(build_args(parsed)).to(device)
    model.init_weights()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=parsed.lr)

    losses = []
    for step in range(parsed.steps):
        x = torch.randint(
            0,
            parsed.vocab_size,
            (parsed.batch_size, parsed.seq_len),
            device=device,
        )
        y = torch.roll(x, shifts=-1, dims=1)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y, attn_impl="sdpa")
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_sq += float(param.grad.detach().float().pow(2).sum().item())
        grad_norm = math.sqrt(grad_norm_sq)
        if not math.isfinite(grad_norm) or grad_norm == 0.0:
            raise RuntimeError(f"Invalid gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f"step={step} loss={losses[-1]:.6f} grad_norm={grad_norm:.6f}")

    print(
        "rwkv7_tiny_train_ok "
        f"steps={parsed.steps} first_loss={losses[0]:.6f} last_loss={losses[-1]:.6f}"
    )


if __name__ == "__main__":
    main()
