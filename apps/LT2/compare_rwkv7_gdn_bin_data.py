"""Compare LT2 mixers on parameter-golf FineWeb .bin token shards.

The parameter-golf shards are uint16 token streams with a 256-int32 header:
magic=20240520, version=1, token_count=header[2]. This harness keeps the
comparison local and deterministic without requiring the full LT2 JSONL data
pipeline.
"""

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.LT2.transformer import LoopedWindowTransformer, LoopedWindowTransformerArgs


def load_data_shard(path: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected parameter-golf shard header: {path}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if path.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {path}: expected {expected_size}")
    tokens_np = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def shard_token_count(path: Path) -> int:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected parameter-golf shard header: {path}")
    return int(header[2])


def resolve_files(pattern: str, max_shards: int = 0) -> List[Path]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files[:max_shards] if max_shards > 0 else files


class TokenStream:
    """Sequential deterministic token stream that wraps around shard files."""

    def __init__(self, files: Iterable[Path]):
        self.files = list(files)
        if not self.files:
            raise ValueError("TokenStream requires at least one shard")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n_tokens: int) -> torch.Tensor:
        chunks = []
        remaining = n_tokens
        while remaining > 0:
            available = self.tokens.numel() - self.pos
            if available <= 0:
                self._advance_file()
                continue
            n = min(remaining, available)
            chunks.append(self.tokens[self.pos : self.pos + n])
            self.pos += n
            remaining -= n
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


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
        use_block_residual=False,
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


def next_batch(
    stream: TokenStream,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = stream.take(batch_size * seq_len + 1).to(dtype=torch.int64)
    x = tokens[:-1].reshape(batch_size, seq_len)
    y = tokens[1:].reshape(batch_size, seq_len)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    files: List[Path],
    parsed: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    stream = TokenStream(files)
    total = 0.0
    for _ in range(parsed.eval_batches):
        x, y = next_batch(stream, parsed.eval_batch_size, parsed.seq_len, device)
        loss = model(x, y, attn_impl="sdpa")
        total += float(loss.detach().cpu())
    model.train()
    return total / parsed.eval_batches


def run_mixer(
    parsed: argparse.Namespace,
    mixer: str,
    train_files: List[Path],
    val_files: List[Path],
) -> Dict[str, object]:
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
    train_stream = TokenStream(train_files)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    initial_val_loss = evaluate(model, val_files, parsed, device) if parsed.eval_batches > 0 else None

    losses = []
    grad_norms = []
    start = None
    for step in range(parsed.warmup_steps + parsed.steps):
        x, y = next_batch(train_stream, parsed.batch_size, parsed.seq_len, device)
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
            print(f"{mixer} {phase}_step={step} loss={loss.item():.6f} grad_norm={norm:.6f}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_memory_mb = None
    if start is None:
        raise RuntimeError("No timed steps were run; set --steps > 0")
    elapsed = time.perf_counter() - start
    final_val_loss = evaluate(model, val_files, parsed, device) if parsed.eval_batches > 0 else None
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
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "val_loss_delta": None if initial_val_loss is None else final_val_loss - initial_val_loss,
        "last_grad_norm": grad_norms[-1],
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "peak_memory_mb": peak_memory_mb,
        "losses": losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LT2 mixers on parameter-golf FineWeb .bin shards.")
    parser.add_argument("--data-dir", default="/home/xiaol/X/parameter-golf/data/datasets/fineweb10B_sp1024")
    parser.add_argument("--train-pattern", default="")
    parser.add_argument("--val-pattern", default="")
    parser.add_argument("--max-train-shards", type=int, default=0)
    parser.add_argument("--max-val-shards", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--mixers", default="gdn,rwkv7_native")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--loop-count", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--rwkv7-head-size", type=int, default=64)
    parser.add_argument("--rwkv7-backend", default="auto", choices=["auto", "cuda", "torch"])
    parser.add_argument("--rwkv7-chunk-len", type=int, default=16)
    parser.add_argument("--rwkv7-use-l2wrap-ce", action="store_true")
    parser.add_argument("--ffn-dim-multiplier", type=float, default=0.5)
    parser.add_argument("--gdn-ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--rwkv7-ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--rwkv7-native-ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--multiple-of", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--model-seed", type=int, default=1234)
    parser.add_argument("--gdn-allow-neg-eigval", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--verbose", action="store_true")
    parsed = parser.parse_args()

    if parsed.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    train_pattern = parsed.train_pattern or str(Path(parsed.data_dir) / "fineweb_train_*.bin")
    val_pattern = parsed.val_pattern or str(Path(parsed.data_dir) / "fineweb_val_*.bin")
    train_files = resolve_files(train_pattern, parsed.max_train_shards)
    val_files = resolve_files(val_pattern, parsed.max_val_shards)
    train_tokens_available = sum(shard_token_count(path) for path in train_files)
    val_tokens_available = sum(shard_token_count(path) for path in val_files)
    mixers = [m.strip().lower() for m in parsed.mixers.split(",") if m.strip()]

    results = []
    for mixer in mixers:
        results.append(run_mixer(parsed, mixer, train_files, val_files))

    print(
        "mixer,params,ffn_dim_multiplier,tokens_per_second,first_loss,last_loss,"
        "loss_delta,mean_loss,initial_val_loss,final_val_loss,val_loss_delta,"
        "last_grad_norm,peak_memory_mb"
    )
    for result in results:
        peak = "" if result["peak_memory_mb"] is None else f"{result['peak_memory_mb']:.2f}"
        val0 = "" if result["initial_val_loss"] is None else f"{result['initial_val_loss']:.6f}"
        val1 = "" if result["final_val_loss"] is None else f"{result['final_val_loss']:.6f}"
        vald = "" if result["val_loss_delta"] is None else f"{result['val_loss_delta']:.6f}"
        print(
            f"{result['mixer']},{result['params']},{result['ffn_dim_multiplier']:.6f},"
            f"{result['tokens_per_second']:.2f},{result['first_loss']:.6f},"
            f"{result['last_loss']:.6f},{result['loss_delta']:.6f},{result['mean_loss']:.6f},"
            f"{val0},{val1},{vald},{result['last_grad_norm']:.6f},{peak}"
        )

    if parsed.json_out:
        payload = {
            "args": vars(parsed),
            "train_files": [str(path) for path in train_files],
            "val_files": [str(path) for path in val_files],
            "train_tokens_available": train_tokens_available,
            "val_tokens_available": val_tokens_available,
            "results": results,
        }
        Path(parsed.json_out).write_text(json.dumps(payload, indent=2) + "\n")

    print(
        "compare_rwkv7_gdn_bin_data_ok "
        f"train_tokens_available={train_tokens_available} val_tokens_available={val_tokens_available}"
    )


if __name__ == "__main__":
    main()
