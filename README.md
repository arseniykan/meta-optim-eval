# Meta-Optim-Eval

This repository combines `optim4rl` and `rl-learned-optimization` to evaluate meta-learned optimizers in reinforcement learning.

## Setup

1.  **Install Dependencies**:
    ```bash
    cd optim4rl
    pip install -r requirements.txt
    ```
    *Note: Ensure you have a compatible JAX installation with GPU support if available.*

2.  **Download Datasets** (if required):
    ```bash
    python download.py
    ```

## Experimental Protocol

We follow the experimental protocol focused on four primary objectives.

### 1. Reproduction of Baselines

To reproduce experiments from Optim4RL and OPEN on the standard Ant environment:

**Baselines (PPO with Adam/RMSProp):**
Run PPO with standard optimizers using `ppo_ant.json`. This config sweeps over Adam and RMSProp.
```bash
# Run PPO baseline (sweeps over seeds and optimizers)
python main.py --config_file ./configs/ppo_ant.json --config_idx 1
```
*Use `seq 1 20` to run all configuration indices (seeds/optimizers).*

**Optim4RL (Meta-Training & Evaluation):**
First, meta-train the optimizer on Ant:
```bash
python main.py --config_file ./configs/meta_rl_ant.json --config_idx 1
```
Then, evaluate the learned optimizer:
```bash
python main.py --config_file ./configs/lopt_rl_ant.json --config_idx 1
```

### 2. Multi-Task Generalization ("Multi-OPEN")

To evaluate generalization across discrete games and the continuous Ant environment:

**Training on Gridworld Tasks:**
Train the optimizer on a suite of Gridworld variants (proxy for "Atari-like" discrete tasks in this codebase):
```bash
python main.py --config_file ./configs/meta_rl_grid.json --config_idx 1
```

**Baselines on Gridworld:**
To compare against baselines (A2C with Adam/RMSProp) on Gridworld:
```bash
python main.py --config_file ./configs/a2c_grid.json --config_idx 1
```

**Evaluation on Ant (Generalization):**
Evaluate the optimizer learned on Gridworld directly on the Ant environment:
```bash
python main.py --config_file ./configs/lopt_rl_grid_ant.json --config_idx 1
```

### 3. Hyperparameter Sensitivity Analysis

To investigate the "hyperparameter-free" claim, we vary the meta-hyperparameters during training.

**Key Meta-Hyperparameters:**
-   `meta_optim.kwargs.learning_rate`: The learning rate of the outer optimizer (Adam).
-   `agent.reset_interval`: The interval at which the inner agent is reset.

**Running Sensitivity Sweeps:**
The `meta_rl_ant.json` configuration is already set up to sweep over these values.
-   `learning_rate`: `[1e-5, 3e-5, 1e-4, 3e-4, 1e-3]`
-   `reset_interval`: `[32, 64, 128, 256, 512]`

Run specific indices to test different combinations. For example:
```bash
# Index 1 might correspond to default LR/Reset
python main.py --config_file ./configs/meta_rl_ant.json --config_idx 1

# Varing indices will automatically select different combinations defined in the config.
# Check 'configs/meta_rl_ant.json' to map indices to specific parameter values.
```

## Running Experiments

All scripts should be run from the `optim4rl` directory.
Use the `run.sh` script as a reference for parallel execution commands.
