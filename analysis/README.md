# Analysis Notebooks

This folder contains cleaned and organized analysis notebooks for the AR classification project.

## Structure

- **utils.py**: Unified utilities for W&B data fetching and plotting
- **__init__.py**: Makes `analysis` a proper Python package for clean imports
- **figures/**: Output directory for generated PDFs (gitignored)

### Available Functions

- `fetch_runs()`: Fetch W&B runs by tags/filters
- `fetch_run_data()`: Get data for a specific run
- `get_runs_data()`: Aggregate history and config from multiple runs
- `differing_config()`: Show config parameters that differ across runs
- `get_table()`: Get attention tables from W&B artifacts
- `plot_combined_heads()`: Plot combined attention heads with color overlays
- `plot_combined_heads_individual()`: Plot individual attention head combinations
- `plot_kl_divergence_simple()`: Plot KL divergence over training
- `plot_val_loss_simple()`: Plot validation loss over training

## Notebooks

### Experiment Analysis
- **init-scales.ipynb**: Analysis of query initialization scales
- **multiplicative-constant.ipynb**: Ablation study on multiplicative constants
- **dataset-validation.ipynb**: Dataset size validation experiments
- **dataset-experiments.ipynb**: Comprehensive dataset size experiments with combined plots

### Model Visualization
- **full-model.ipynb**: Full model analysis with attention patterns and KL divergence
- **figure1.ipynb**: Main figure plots including attention patterns and ideal attention
- **infinite-data.ipynb**: Infinite data regime experiments
- **reverse-constants.ipynb**: Reverse constant experiments

## Usage

All notebooks use clean imports from the `analysis` package:

```python
from analysis.utils import fetch_runs, get_runs_data, plot_kl_divergence_simple
import matplotlib.pyplot as plt
import numpy as np
```

**To run a notebook:**
1. Ensure you're in the project root directory
2. Open any notebook in Jupyter
3. Run cells normally

All generated PDFs are saved to `analysis/figures/` and are automatically excluded from git.

### Requirements

The `analysis` package requires:
- wandb
- pandas
- matplotlib
- seaborn
- numpy

Install with: `pip install wandb pandas matplotlib seaborn numpy`
