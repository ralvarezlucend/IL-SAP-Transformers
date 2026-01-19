"""Utility helpers for logging transformer self-attention during training.

Provides two independent representations:
* `log_attention_table` – structured numeric weights in a wandb.Table for analysis
* `log_attention_heatmap` – static per-head heatmaps logged as wandb.Images

Use `log_attention` as a wrapper when you want either or both.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb


def _get_attention(
    attn_weights: torch.Tensor, layer: int = 0, batch_idx: int = -1
) -> np.ndarray:
    """Extract attention weights of shape (heads, seq_len, seq_len).

    Args:
        attn_weights: Full attention tensor from model
        layer: Which layer to extract (default: 0)
        batch_idx: Which batch item (-1 for batch average)

    Returns:
        Attention array of shape (heads, seq_len, seq_len)
    """
    try:
        attn = attn_weights[layer].detach().cpu().numpy()
    except IndexError as exc:
        raise IndexError(
            f"Layer index {layer} out of range for attn_weights with "
            f"{len(attn_weights)} layers"
        ) from exc

    if batch_idx == -1:  # average over batch
        return attn.mean(axis=0)

    if batch_idx >= attn.shape[0]:
        raise IndexError(
            f"batch_idx {batch_idx} out of range for batch size {attn.shape[0]}"
        )
    return attn[batch_idx]


def _default_token_seq(seq_len: int) -> List[str]:
    """Generate default token labels as 0-based indices."""
    return [str(i) for i in range(seq_len)]


def build_attention_table(
    attn: np.ndarray, token_seq: Optional[List[str]] = None
) -> wandb.Table:
    """Convert (heads, seq_len, seq_len) attention into structured wandb.Table.

    Args:
        attn: Attention weights of shape (heads, seq_len, seq_len)
        token_seq: Token labels (defaults to indices if None)

    Returns:
        wandb.Table with columns: head, query_idx, key_idx, query_token, key_token, weight
    """
    num_heads, seq_len, _ = attn.shape
    if token_seq is None:
        token_seq = _default_token_seq(seq_len)

    cols = ["head", "query_idx", "key_idx", "query_token", "key_token", "weight"]
    rows: List[List] = []

    for h in range(num_heads):
        for q in range(seq_len):
            for k in range(seq_len):
                rows.append([
                    h, q, k, token_seq[q], token_seq[k], float(attn[h, q, k])
                ])

    return wandb.Table(data=rows, columns=cols)


def log_attention_table(
    run: Optional["wandb.run"],
    attn_weights: torch.Tensor,
    token_seq: Optional[List[str]] = None,
    layer: int = 0,
    batch_idx: int = -1,
    step: Optional[int] = None,
    table_key: str = "attention_table",
) -> None:
    """Log structured attention weights as a wandb.Table.

    Args:
        run: Active wandb.run (skipped if None)
        attn_weights: Full attention tensor from model
        token_seq: Human-readable tokens (defaults to indices)
        layer: Which layer to visualize
        batch_idx: Which batch item (-1 for average)
        step: Training step for versioning
        table_key: Dashboard key for table versions
    """
    if run is None or step is None:
        return

    attn = _get_attention(attn_weights, layer, batch_idx)
    table = build_attention_table(attn, token_seq)

    # Log table with versioning at this step
    run.log({table_key: table}, step=step)


def log_attention_heatmap(
    run: Optional["wandb.run"],
    attn_weights: np.ndarray,
    log_key: str,
    token_seq: Optional[List[str]] = None,
    layer: int = 0,
    batch_idx: int = -1,
    step: Optional[int] = None,
) -> None:
    """Log per-head heatmaps plus an averaged heatmap.

    Args:
        run: wandb run instance (skipped if None)
        attn_weights: attention weights of shape (heads, seq_len, seq_len)
        log_key: key for logging
        token_seq: token labels (defaults to indices)
        layer: layer index (for consistency with table function)
        batch_idx: batch index (for consistency with table function)
        step: training step
    """
    if run is None or step is None:
        return

    # Handle both numpy arrays (current usage) and torch tensors (for consistency)
    if isinstance(attn_weights, torch.Tensor):
        attn = _get_attention(attn_weights, layer, batch_idx)
    else:
        attn = attn_weights  # Already processed numpy array

    num_heads, seq_len, _ = attn.shape
    if token_seq is None:
        token_seq = _default_token_seq(seq_len)

    images: List[wandb.Image] = []

    # Per-head heatmaps
    for h in range(num_heads):
        fig = plt.figure(figsize=(4, 4))
        sns.heatmap(
            attn[h],
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            xticklabels=token_seq,
            yticklabels=token_seq,
            cbar=True,
        )
        plt.title(f"Head {h}")
        plt.xlabel("Position")
        plt.ylabel("Position")
        plt.xticks(rotation=45)
        plt.tight_layout()
        images.append(wandb.Image(fig, caption=f"Head {h}"))
        plt.close(fig)

    # Average heatmap
    fig = plt.figure(figsize=(4, 4))
    sns.heatmap(
        attn.mean(axis=0),
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        xticklabels=token_seq,
        yticklabels=token_seq,
        cbar=True,
    )
    plt.title("Average Heads")
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.xticks(rotation=45)
    plt.tight_layout()
    images.append(wandb.Image(fig, caption="Average"))
    plt.close(fig)

    run.log({log_key: images}, step=step)


