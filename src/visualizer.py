"""Utility helpers for logging transformer self-attention during training.

Provides two independent representations:
* `log_attention_table` – structured numeric weights in a wandb.Table for analysis
* `log_attention_heatmap` – static per-head heatmaps logged as wandb.Images
* `compute_gt_attention_row` – ground-truth uniform attention distribution per head
* `log_value_matrix_alignment` – per-head cosine similarity of value weights vs. teacher
* `log_attention_alignment` – per-head attention alignment scalars and bar charts

Use `log_attention` as a wrapper when you want either or both attention functions.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
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


def compute_gt_attention_row(
    span_lengths: List[int],
    context_length: int,
    seq_len: int,
    stride: Optional[int] = None,
) -> np.ndarray:
    """Compute the ground-truth attention row for the last query position.

    For each head h, the ground truth is a uniform distribution over the
    span_lengths[h] positions that correspond to lag h in the trimmed attention
    sequence.  The context window occupies the last `context_length` columns of
    the trimmed sequence, and within that window the spans are laid out as:
        - without stride: span h starts at sum(span_lengths[:h])
        - with stride s:  span h starts at h * s

    Args:
        span_lengths: per-head span lengths (list of ints, length == num_heads)
        context_length: total context window length (sum of spans or stride-based)
        seq_len: length of the trimmed attention axis (the `length` dimension
                 after prefix trimming in _run_ar_model)
        stride: stride between spans; None means non-overlapping

    Returns:
        gt: np.ndarray of shape (num_heads, seq_len), each row sums to 1.0
    """
    num_heads = len(span_lengths)
    gt = np.zeros((num_heads, seq_len), dtype=np.float32)

    # The relevant context for the last query occupies these columns.
    context_start = seq_len - context_length

    for h in range(num_heads):
        if stride is not None:
            span_start_in_context = h * stride
        else:
            span_start_in_context = sum(span_lengths[:h])

        span_len = span_lengths[h]
        abs_start = context_start + span_start_in_context
        abs_end = abs_start + span_len

        abs_start = max(0, min(abs_start, seq_len))
        abs_end = max(0, min(abs_end, seq_len))

        if abs_end > abs_start:
            gt[h, abs_start:abs_end] = 1.0 / (abs_end - abs_start)

    return gt


def log_value_matrix_alignment(
    run: Optional["wandb.run"],
    teacher_matrices: torch.Tensor,
    student: torch.nn.Module,
    dim: int,
    step: int,
    split: str,
    layer: int = 0,
) -> None:
    """Log per-head cosine similarity between student value projections and teacher matrices.

    Operates only when the student block uses attention_disentanglement=True so that
    each head has its own nn.Linear value projection.  The effective (dim, dim) sub-block
    of each value weight is compared to the corresponding teacher matrix via the
    Frobenius inner product cosine similarity.

    Args:
        run: Active wandb run (skipped if None)
        teacher_matrices: shape (window, dim, dim) — teacher._params (may be on any device)
        student: TransformerDecoder instance
        dim: data / vocabulary dimensionality (not embed_dim)
        step: training step
        split: "train" or "val"
        layer: which transformer block to inspect (default 0)
    """
    if run is None:
        return

    block = student.transformer_blocks[layer]
    mha = block.self_attention

    if not getattr(mha, "attention_disentanglement", False):
        return
    if not isinstance(mha.value_proj, nn.ModuleList):
        return

    teacher_np = teacher_matrices.detach().cpu().numpy()  # (window, dim, dim)
    num_teacher = teacher_np.shape[0]
    num_heads = len(mha.value_proj)

    log_dict = {}
    for h in range(min(num_heads, num_teacher)):
        vp = mha.value_proj[h]
        if not isinstance(vp, nn.Linear):
            continue

        # Effective (dim, dim) block: decoder keeps first `dim` rows, token
        # embedding occupies first `dim` columns of the value weight.
        W = vp.weight.detach().cpu().numpy()[:dim, :dim]  # (dim, dim)
        A = teacher_np[h]                                   # (dim, dim)

        w_flat = W.ravel()
        a_flat = A.ravel()
        w_norm = float(np.linalg.norm(w_flat))
        a_norm = float(np.linalg.norm(a_flat))

        cos_sim = float(np.dot(w_flat, a_flat) / (w_norm * a_norm)) if (w_norm > 0 and a_norm > 0) else 0.0
        log_dict[f"{split}_value_cosine_sim_head{h}"] = cos_sim

    if log_dict:
        run.log(log_dict, step=step)


def log_attention_alignment(
    run: Optional["wandb.run"],
    attn_avg: np.ndarray,
    span_lengths: List[int],
    context_length: int,
    step: int,
    split: str,
    stride: Optional[int] = None,
) -> None:
    """Log alignment between the last-query attention row and the GT span distribution.

    For each head h, compares attn_avg[h, -1, :] (the last query's attention
    distribution) against the ground-truth uniform distribution over span h.
    Logs per-head cosine similarity and L1 distance scalars, plus a grouped
    bar chart image showing predicted vs GT weights restricted to the context window.

    Args:
        run: Active wandb run (skipped if None)
        attn_avg: batch-averaged attention, shape (num_heads, seq_len, seq_len)
        span_lengths: per-head span lengths
        context_length: total context window length
        step: training step
        split: "train" or "val"
        stride: stride between spans (None = non-overlapping)
    """
    if run is None:
        return

    num_heads, seq_len, _ = attn_avg.shape
    gt = compute_gt_attention_row(span_lengths, context_length, seq_len, stride=stride)
    # gt shape: (num_heads, seq_len)

    log_dict: dict = {}
    images: List[wandb.Image] = []

    for h in range(num_heads):
        pred_row = attn_avg[h, -1, :]  # (seq_len,) — last query position
        gt_row = gt[h] if h < len(span_lengths) else np.zeros(seq_len, dtype=np.float32)

        pred_norm = float(np.linalg.norm(pred_row))
        gt_norm = float(np.linalg.norm(gt_row))

        cos_sim = float(np.dot(pred_row, gt_row) / (pred_norm * gt_norm)) if (pred_norm > 0 and gt_norm > 0) else 0.0
        l1_dist = float(np.sum(np.abs(pred_row - gt_row)))

        log_dict[f"{split}_attn_alignment_cos_head{h}"] = cos_sim
        log_dict[f"{split}_attn_alignment_l1_head{h}"] = l1_dist

        # Bar chart: zoom in on the context window for readability.
        context_start = max(0, seq_len - context_length)
        positions = np.arange(context_start, seq_len)
        pred_ctx = pred_row[context_start:]
        gt_ctx = gt_row[context_start:]

        fig, ax = plt.subplots(figsize=(max(6, len(positions) // 2), 3))
        width = 0.4
        ax.bar(positions - width / 2, pred_ctx, width=width, label="Student", color="steelblue")
        ax.bar(positions + width / 2, gt_ctx, width=width, label="GT", color="coral", alpha=0.8)
        ax.set_title(f"Head {h} — last-query attention ({split}, step {step})")
        ax.set_xlabel("Key position (context window)")
        ax.set_ylabel("Attention weight")
        ax.legend(fontsize=8)
        ax.set_xlim(context_start - 0.5, seq_len - 0.5)
        plt.tight_layout()
        images.append(wandb.Image(fig, caption=f"Head {h}"))
        plt.close(fig)

    log_dict[f"{split}_attn_alignment_charts"] = images
    run.log(log_dict, step=step)
