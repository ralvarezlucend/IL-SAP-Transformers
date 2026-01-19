Removed / Clean up:

- Removed spectral norm
- Removed the following unused metrics: 
    [
        "rel_val_true_loss",
        "rel_val_loss",
        "val_logits_hessian_2",
        "val_logits_hessian",
        "val_logits_grad_2",
        "val_logits_grad",
        "val_logits_l2_2",
        "val_logits_l2",
        "rel_train_true_loss",
        "rel_train_loss",
        "train_logits_hessian_2",
        "train_logits_hessian",
        "train_logits_grad_2",
        "train_logits_grad",
        "train_logits_l2_2",
        "train_logits_l2",
    ]
    - Removed optimizer.py with Adsgd optimizer and conf/optimizer/adsgd.yaml
    - Removed evaluator.py, eval.py, conf/eval.yaml, and conf/evaluator/
    - Removed state.py and all checkpoint functionality (no longer saving or loading model checkpoints)
    - Removed stability.py and conf/stability.yaml (stability analysis script)
    - Removed stats.py, conf/stats.yaml, and conf/trainer/stats.yaml (statistics analysis script) 
    - Add uv enviroment for better dependecy management. 
    - Refactor analysis notebooks that include the figures from the paper. 
    - Removed unused positional encodings: relative and rope.
    - Improved experiments launcher with Hydra sweep and Ray launcher. TODO: launch experiments and make sure results are reproducible.
    - Removed LowRankOperator / LowRankModel
    - Log attention heatmaps / tables with a frequency
    - Add tags for experiments
    - Switch to step-only training

<!-- - TODO: rerun all experiments in clean wandb project and make it public
- TODO: reformat notebooks. -->