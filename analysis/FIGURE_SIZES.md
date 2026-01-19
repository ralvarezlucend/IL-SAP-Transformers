# Figure Sizes Reference

This document lists all figure sizes used in the notebooks for easy reference and consistency.

## Summary of Figure Sizes

### Individual Notebooks

**init-scales.ipynb**
- Main plot: `figsize=(10, 6)` - Excess loss comparison

**multiplicative-constant.ipynb**
- Main plot: `figsize=(10, 6)` - Ablation study on MC

**dataset-validation.ipynb**
- Bar chart: `figsize=(12, 6)` - Best validation loss by model and dataset size

**dataset-experiments.ipynb**
- Val loss plot: Uses matplotlib default
- KL divergence comparison: `figsize=(12, 4)` - Three panel subplot
- Combined figure: `figsize=(24, 6)` - Four panel subplot (1 val loss + 3 KL)

**full-model.ipynb**
- All plots: `figsize=(12, 8)` - KL divergence and validation loss plots

**figure1.ipynb**
- KL/Val loss plots: `figsize=(10, 8)` - Main figure plots
- Ideal attention patterns: `figsize=(8, 8)` - Three heatmap plots (square for proper aspect ratio)

**infinite-data.ipynb**
- All plots: `figsize=(12, 8)` - Infinite data regime experiments

**reverse-constants.ipynb**
- All plots: `figsize=(12, 8)` - Reverse constant experiments

### Utility Functions (analysis/utils.py)

**plot_combined_heads()**
- Default: `figsize=(12 * n_cols, 12)` - Scales with number of time steps

**plot_combined_heads_individual()**
- Default: `figsize=(12, 12)` - Single attention head combination

**plot_kl_divergence_simple()**
- Default: `figsize=(12, 8)` - Can be overridden per call

**plot_val_loss_simple()**
- Default: `figsize=(12, 8)` - Can be overridden per call

## Guidelines

1. **Line plots**: Use `(10-12, 6-8)` for single plots, wider for multiple subplots
2. **Heatmaps**: Use square dimensions `(8, 8)` or `(12, 12)` to maintain aspect ratio
3. **Multi-panel figures**: Scale width proportionally - e.g., `(12, 4)` for 3 panels, `(24, 6)` for 4 panels
4. **All PDFs**: Saved with `bbox_inches="tight"` and `dpi=300` for publication quality
