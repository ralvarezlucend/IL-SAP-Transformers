"""Unified utilities for W&B data fetching and plotting."""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Sequence, Optional
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

DEFAULT_ENTITY = "r-alvarezlucendo16"
DEFAULT_PROJECT = "incremental-learning"


# ============================================================================
# W&B Data Fetching
# ============================================================================

def fetch_runs(
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    tags_any: Optional[Sequence[str]] = None,
    extra_filters: Optional[Dict[str, Any]] = None,
) -> List[wandb.apis.public.Run]:
    """Fetch W&B runs matching filters."""
    api = wandb.Api()
    filters: Dict[str, Any] = {}
    if tags_any:
        filters["tags"] = {"$in": list(tags_any)}
    if extra_filters:
        filters.update(extra_filters)
    return list(api.runs(f"{entity}/{project}", filters=filters))


def fetch_run_data(
    run_id: str,
    metrics: List[str],
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> Dict[str, Any]:
    """Fetch data for a specific run."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    all_metrics = list(set(metrics + ["_step"]))
    history_df = pd.DataFrame(run.scan_history(keys=all_metrics))
    return {
        "df": history_df,
        "config": dict(run.config),
        "name": run.name,
        "id": run.id,
    }


def get_runs_data(
    runs: Sequence[wandb.apis.public.Run],
    metrics: Sequence[str],
    step_key: str = "_step",
    include_config: bool = True,
    config_sep: str = ".",
    config_prefix: str = "cfg",
) -> pd.DataFrame:
    """Aggregate history and config from multiple runs."""
    dfs: List[pd.DataFrame] = []

    for r in runs:
        h = pd.DataFrame(r.scan_history(keys=list(metrics) + [step_key]))
        if h.empty:
            continue

        meta = {"_run_id": r.id, "_run_name": r.name}

        if include_config:
            flat_cfg = pd.json_normalize(dict(r.config), sep=config_sep)
            flat_cfg = flat_cfg.add_prefix(f"{config_prefix}{config_sep}")
            meta.update(flat_cfg.to_dict(orient="records")[0])

        meta_block = pd.DataFrame([meta] * len(h)).reset_index(drop=True)
        h = pd.concat([h.reset_index(drop=True), meta_block], axis=1)
        dfs.append(h)

    return pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()


def differing_config(
    df: pd.DataFrame,
    run_id_col: str = "_run_id",
    run_name_col: str = "_run_name",
) -> pd.DataFrame:
    """Return one row per run with only config columns that differ across runs."""
    if df.empty:
        return pd.DataFrame()

    cfg_cols = [c for c in df.columns if c.startswith("cfg.")]
    if not cfg_cols:
        return pd.DataFrame()

    id_cols = [run_id_col, run_name_col]
    per_run = df.groupby(id_cols, dropna=False)[cfg_cols].first().reset_index()

    varying_cols = per_run[cfg_cols].nunique(dropna=False)
    varying_cols = list(varying_cols.index[varying_cols > 1])

    return per_run[id_cols + varying_cols] if varying_cols else per_run[id_cols]


def get_table(
    artifact_path: str,
    step: int,
    split: str = "val",
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> pd.DataFrame:
    """Get attention table from W&B artifact."""
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{artifact_path}:v{step}")
    table_key = f"{split}_attention_weights"
    table = artifact.get(table_key)

    if table is None:
        raise ValueError(f"No table '{table_key}' in artifact {artifact_path}:v{step}")

    return pd.DataFrame(data=table.data, columns=table.columns)


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_combined_heads(
    artifact_path: str,
    steps: List[int],
    split: str = "val",
    save_name: Optional[str] = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Plot combined attention heads with color overlays."""
    if isinstance(steps, int):
        steps = [steps]

    n_cols = len(steps)
    fig, axes = plt.subplots(1, n_cols, figsize=(12 * n_cols, 12), sharey=False)
    if n_cols == 1:
        axes = [axes]

    colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]  # Green, Orange, Blue

    for col_idx, step in enumerate(steps):
        ax = axes[col_idx]
        df = get_table(artifact_path, step, split=split, entity=entity, project=project)

        head_indices = sorted(df["head"].unique())
        head_colors = {h: colors[i % len(colors)] for i, h in enumerate(head_indices)}

        first_head_df = df[df["head"] == head_indices[0]]
        query_indices = sorted(first_head_df["query_idx"].unique())
        key_indices = sorted(first_head_df["key_idx"].unique())

        combined_rgb = np.ones((len(query_indices), len(key_indices), 3))

        for head_idx in head_indices:
            df_head = df[df["head"] == head_idx]
            attn = df_head.pivot(index="query_idx", columns="key_idx", values="weight")
            color_rgb = np.array(plt.cm.colors.to_rgb(head_colors[head_idx]))

            for i, query_idx in enumerate(query_indices):
                for j, key_idx in enumerate(key_indices):
                    weight = attn.loc[query_idx, key_idx]
                    combined_rgb[i, j] = combined_rgb[i, j] * (1 - weight) + color_rgb * weight

        ax.imshow(combined_rgb, aspect="equal", interpolation="nearest", origin="upper")
        ax.set_xticks(range(len(key_indices)))
        ax.set_xticklabels(key_indices, fontsize=32, rotation=90)
        ax.set_yticks(range(len(query_indices)))
        ax.set_yticklabels(query_indices, fontsize=32)

        if col_idx == 0:
            ax.set_ylabel("Query Positions", fontsize=55)
        ax.set_xlabel("Key Positions", fontsize=55)
        ax.set_title(f"$\\mathbf{{Step~{step}}}$", fontsize=70, pad=20)

    plt.tight_layout()
    if save_name:
        fig.savefig(f"figures/{save_name}.pdf", bbox_inches="tight", dpi=300)
    else:
        plt.show()


def plot_combined_heads_individual(
    artifact_path: str,
    step: int,
    split: str = "val",
    gamma: float = 1.0,
    save_name: Optional[str] = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Plot combined attention heads with screen blending (individual step)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    df = get_table(artifact_path, step, split=split, entity=entity, project=project)
    colors = ("#2ca02c", "#1f77b4", "#ff7f0e")

    head_indices = sorted(df["head"].unique())
    head_colors = {
        h: np.array(mcolors.to_rgb(colors[i % len(colors)]), dtype=float)
        for i, h in enumerate(head_indices)
    }

    query_indices = sorted(df["query_idx"].unique())
    key_indices = sorted(df["key_idx"].unique())
    nq, nk = len(query_indices), len(key_indices)

    prod_color = np.ones((nq, nk, 3), dtype=float)
    prod_alpha = np.ones((nq, nk), dtype=float)

    for h in head_indices:
        df_h = df[df["head"] == h]
        A = (
            df_h.pivot(index="query_idx", columns="key_idx", values="weight")
            .reindex(index=query_indices, columns=key_indices)
            .fillna(0.0)
            .to_numpy()
            .astype(float)
        )

        if gamma != 1.0:
            A = np.power(A, gamma)

        W = np.clip(A, 0.0, 1.0)
        C = head_colors[h]

        prod_color *= 1.0 - W[..., None] * C[None, None, :]
        prod_alpha *= 1.0 - W

    S = 1.0 - prod_color
    t = 1.0 - prod_alpha
    combined_rgb = (1.0 - t)[..., None] + S
    combined_rgb = np.clip(combined_rgb, 0.0, 1.0)

    ax.imshow(combined_rgb, aspect="equal", interpolation="nearest", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if save_name:
        plt.savefig(f"figures/{save_name}.pdf", bbox_inches="tight", dpi=300)
    else:
        plt.show()


def plot_kl_divergence_simple(
    run_id: str,
    divergence_steps: Optional[List[int]] = None,
    max_steps: Optional[int] = None,
    figsize: tuple = (12, 8),
    learnable: bool = False,
    shift_steps: bool = True,
    save_name: Optional[str] = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Plot KL divergence over training steps."""
    if learnable:
        metrics = [
            "kl_div_unigram_learned_val",
            "kl_div_bigram_learned_val",
            "kl_div_teacher_val",
        ]
        kl_metrics = {
            "kl_div_unigram_learned_val": "4-gram",
            "kl_div_bigram_learned_val": "8-gram",
            "kl_div_teacher_val": "12-gram",
        }
    else:
        metrics = [
            "kl_div_prefix_1_teacher_val",
            "kl_div_prefix_2_teacher_val",
            "kl_div_prefix_3_teacher_val",
        ]
        kl_metrics = {
            "kl_div_prefix_1_teacher_val": r"${A^{*}_{1}}$",
            "kl_div_prefix_2_teacher_val": r"${A^{*}_{1:2}}$",
            "kl_div_prefix_3_teacher_val": r"${A^{*}_{1:3}}$",
        }

    data = fetch_run_data(run_id, metrics, entity=entity, project=project)
    df = data["df"]

    if df.empty:
        print(f"No data found for run {run_id}")
        return

    plot_df = df if max_steps is None else df[df["_step"] <= max_steps]
    plot_df = plot_df.copy()
    if shift_steps:
        plot_df["_step"] = plot_df["_step"] - 2000

    plt.figure(figsize=figsize)
    x_min, x_max = plot_df["_step"].min(), plot_df["_step"].max()
    plt.margins(x=0)
    plt.xlim(x_min, x_max)

    if divergence_steps and len(divergence_steps) >= 2:
        strategy_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        adjusted_steps = [s - 2000 for s in divergence_steps] if shift_steps else divergence_steps

        x_range = x_max - x_min
        x_min_ext = x_min - 0.01 * x_range
        x_max_ext = x_max + 0.01 * x_range

        plt.axvspan(x_min_ext, adjusted_steps[0], alpha=0.2, color=strategy_colors[0])
        plt.axvspan(adjusted_steps[0], adjusted_steps[1], alpha=0.2, color=strategy_colors[1])
        plt.axvspan(adjusted_steps[1], x_max_ext, alpha=0.2, color=strategy_colors[2])

    for metric, label in kl_metrics.items():
        if metric in df.columns:
            plt.plot(plot_df["_step"], plot_df[metric], label=label, linewidth=4)

    plt.xlabel("Training Step", fontsize=30)
    plt.ylabel("KL Divergence", fontsize=30)
    plt.legend(fontsize=32, loc="upper right", framealpha=1)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_name:
        plt.savefig(f"figures/{save_name}.pdf", bbox_inches="tight", dpi=300)
    else:
        plt.show()


def plot_val_loss_simple(
    run_id: str,
    divergence_steps: Optional[List[int]] = None,
    max_steps: Optional[int] = None,
    figsize: tuple = (12, 8),
    shift_steps: bool = True,
    save_name: Optional[str] = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Plot validation loss over training steps."""
    data = fetch_run_data(run_id, ["val_loss"], entity=entity, project=project)
    df = data["df"]
    plot_df = df if max_steps is None else df[df["_step"] <= max_steps]
    plot_df = plot_df.copy()
    if shift_steps:
        plot_df["_step"] = plot_df["_step"] - 2000

    plt.figure(figsize=figsize)
    plt.margins(x=0)

    if divergence_steps and len(divergence_steps) >= 2:
        strategy_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        adjusted_steps = [s - 2000 for s in divergence_steps] if shift_steps else divergence_steps

        x_min, x_max = plot_df["_step"].min(), plot_df["_step"].max()
        x_range = x_max - x_min
        x_min_ext = x_min - 0.01 * x_range
        x_max_ext = x_max + 0.01 * x_range

        plt.axvspan(x_min_ext, adjusted_steps[0], alpha=0.2, color=strategy_colors[0])
        plt.axvspan(adjusted_steps[0], adjusted_steps[1], alpha=0.2, color=strategy_colors[1])
        plt.axvspan(adjusted_steps[1], x_max_ext, alpha=0.2, color=strategy_colors[2])

    if "val_loss" in plot_df.columns:
        plt.plot(plot_df["_step"], plot_df["val_loss"], linewidth=2, color="black")

    plt.xlabel("Training Step", fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    if divergence_steps and len(divergence_steps) >= 2:
        strategy_labels = [r"$A^*_1$", r"$A^*_{1:2}$", r"$A^*_{1:3}$"]
        legend_patches = [Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.2) for c in strategy_colors]
        plt.legend(legend_patches, strategy_labels, fontsize=32, loc="upper right", framealpha=1)

    plt.tight_layout()

    if save_name:
        plt.savefig(f"figures/{save_name}.pdf", bbox_inches="tight", dpi=300)
    else:
        plt.show()
