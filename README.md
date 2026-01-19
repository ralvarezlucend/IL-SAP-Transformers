# Incremental Learning of Sparse Attention Patterns in Transformers

This is the official code for the paper on [Incremental Learning of Sparse Attention Patterns in Transformers](https://okyksl.github.io/slides/prigm-2025/#/1) presented at [EurIPS 2025 Workshop
on Principles of Generative Modeling](https://sites.google.com/view/prigm-eurips-2025/home).

## Installation

```bash
uv sync
```

## Running Experiments

```bash
# List available experiments
bash run.sh

# Run a specific experiment
bash run.sh <experiment_name>
```

## Configuration

Experiments are configured using [Hydra](https://hydra.cc/) with configs located in `conf/`.

- **Experiment configs** in `conf/experiments/` override base settings from `conf/train.yaml`
- **Component configs** can be customized: `model/`, `dataset/`, `optimizer/`, `scheduler/`, `loss/`