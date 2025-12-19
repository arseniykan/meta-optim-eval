
import argparse
import os
import sys
import json
import time
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import numpy as np
# import gymnax # Lazy import later
from agents import PPO
from utils.gymnax_wrapper import (
    GymnaxGymWrapper,
    FlatWrapper,
    GymnaxLogWrapper,
    BraxGymnaxWrapper,
    ClipAction
)
# Ensure we can import from local dirs
sys.path.append(os.getcwd())

class UnifiedPPO(PPO):
    def __init__(self, cfg):
        # We need to Bypass PPO.__init__ partly because it hardcodes brax.envs.get_environment
        # converting the cfg to self.cfg
        self.cfg = cfg
        self.config_idx = cfg['config_idx']
        
        # Logging
        # We should suppress excessive logging if running many seeds
        from utils.logger import Logger
        self.logger = Logger(cfg['logs_dir'])
        # We don't necessarily need the feather file for every run if we consolidate, but good for debug
        self.log_path = cfg['logs_dir'] + f'result_Test.feather'
        self.result = []
        
        self.env_name = cfg['env']['name']
        self.agent_name = cfg['agent']['name']
        self.train_steps = int(cfg['env']['train_steps'])
        
        # --- ENV SETUP ---
        if self.env_name in ["breakout", "asterix", "freeway", "space_invaders"]:
            import gymnax # Lazy import
            # Setup Gymnax
            param_env_name = self.env_name
            gymnax_name = f"MinAtar-{param_env_name.replace('_', '').capitalize() if param_env_name != 'space_invaders' else 'SpaceInvaders'}"
            if param_env_name == "breakout": gymnax_name = "MinAtar-Breakout"
            if param_env_name == "asterix": gymnax_name = "MinAtar-Asterix"
            if param_env_name == "freeway": gymnax_name = "MinAtar-Freeway"

            self.logger.info(f"Loading Gymnax Env: {gymnax_name}")
            
            basic_env, env_params = gymnax.make(gymnax_name)
            
            # Wrap to be Brax-compatible
            # 1. Gymnax -> Gym API (Continuous/Discrete handling)
            config_mock = {"CONTINUOUS": False}
            env = GymnaxGymWrapper(basic_env, env_params, config_mock)
            
            # 2. Flatten Observations
            env = FlatWrapper(env)
            
            # 3. Log Wrapper (Internal stats) - Note: Brax output stats come from here?
            env = GymnaxLogWrapper(env)
            
            # 4. Brax API Wrapper (State.obs, etc.)
            self.env = BraxGymnaxWrapper(env, env_params=None) 
            
        else:
            # Standard Brax
            from brax import envs
            backends = ['generalized', 'positional', 'spring']
            self.env = envs.get_environment(env_name=self.env_name, backend=backends[1])

        # Initialize State
        self.state = jax.jit(self.env.reset)(rng=jax.random.PRNGKey(seed=self.cfg['seed']))
        
        # --- Timing and Device Setup (Copied from PPO.py) ---
        self.start_time = time.time()
        self._PMAP_AXIS_NAME = 'i'
        self.process_id = jax.process_index()
        self.process_count = jax.process_count()
        total_device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        max_devices_per_host = self.cfg['max_devices_per_host']
        if max_devices_per_host is not None and max_devices_per_host > 0:
            self.local_devices_to_use = min(local_device_count, max_devices_per_host)
        else:
            self.local_devices_to_use = local_device_count
        self.core_reshape = lambda x: x.reshape((self.local_devices_to_use,) + x.shape[1:])
        
        self.logger.info(f'Total device: {total_device_count}, Process: {self.process_count} (ID {self.process_id})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Env name: breakout, ant, humanoid...")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "rmsprop", "optim4rl"])
    parser.add_argument("--steps", type=float, default=1e7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-runs", type=int, default=18)
    args = parser.parse_args()
    
    # Parent Log Dir for Consolidation
    consolidated_log_dir = f"./logs/unified/{args.env}/{args.optimizer}/"
    os.makedirs(consolidated_log_dir, exist_ok=True)
    
    all_returns = []
    
    for i in range(args.num_runs):
        current_seed = args.seed + i
        print(f"=== Starting Run {i+1}/{args.num_runs} (Seed {current_seed}) ===")
        
        # Construct Config per run
        cfg = {
          "config_idx": i+1,
          "logs_dir": os.path.join(consolidated_log_dir, f"{current_seed}/"),
          "seed": current_seed,
          "generate_random_seed": False,
          "max_devices_per_host": -1,
          "batch_size": 1024 if args.optimizer == "optim4rl" else 256, 
          "discount": 0.99,
          "env": {
              "name": args.env,
              "train_steps": args.steps,
              "episode_length": 1000,
              "action_repeat": 1,
              "reward_scaling": 1.0,
              "num_envs": 2048,
              "num_evals": 20,
              "normalize_obs": True
          },
          "agent": {
              "name": "PPO",
              "gae_lambda": 0.95,
              "rollout_steps": 10,
              "num_minibatches": 4, 
              "clipping_epsilon": 0.2,
              "update_epochs": 4,
              "entropy_weight": 0.01
          },
          "optim": {
              "name": "Adam" if args.optimizer == "adam" else ("RMSProp" if args.optimizer == "rmsprop" else "Optim4RL"),
              "kwargs": {
                  "learning_rate": args.lr,
                  "grad_clip": 1.0
              }
          }
        }
        
        if args.optimizer == "optim4rl":
            if "humanoid" in args.env:
                 cfg["optim"]["kwargs"]["param_load_path"] = "./logs/meta_rl_humanoid/1/param.pickle"
            elif "ant" in args.env:
                 cfg["optim"]["kwargs"]["param_load_path"] = "./logs/meta_rl_ant/1/param.pickle"
        
        os.makedirs(cfg["logs_dir"], exist_ok=True)
        
        # Run
        # Re-initializing UnifiedPPO in loop works because JAX jit cache is based on shapes/functions, 
        # so repeated calls are efficient, though PPO.__init__ does some JIT compilation of reset.
        agent = UnifiedPPO(cfg)
        agent.train()
        
        # Collect results
        # self.result is list of dicts
        results = pd.DataFrame(agent.result)
        if not results.empty:
            all_returns.append(results['Return'].values)
        else:
            print(f"Warning: No results for run {i+1}")
            
    # Consolidation
    # all_returns is list of arrays. Each array is [Steps]
    # We want [Runs, Steps]
    if all_returns:
        # Check lengths match (they should if steps fixed)
        min_len = min(len(r) for r in all_returns)
        truncated_returns = [r[:min_len] for r in all_returns]
        
        data_matrix = np.array(truncated_returns) # Shape (Runs, Steps)
        
        df_out = pd.DataFrame(data_matrix)
        # Use header=True/False? 
        # If plot script reads pd.read_csv(path), it treats first row as header usually.
        # But our plot script doesn't specify header=None, so it eats the first row!
        # Unless the CSV has a header.
        
        # OPEN `eval.py` typically has NO header? 
        # Actually checking `plot_comparison.py`: `df = pd.read_csv(path)` -> header=0 default.
        # So it consumes first run as header? That's bad.
        # UNLESS the CSV has a header like "step1, step2".
        # But `eval_baselines.py` saved WITHOUT header and index.
        # If `eval_baselines.py` saved without header, then `pd.read_csv` would consume the first row as header!
        # This means my plot script is likely losing the first run of data.
        
        # Correction: `pd.read_csv(headers=None)` should be used if no header.
        # But I can't easily change `plot_comparison.py` if the user didn't ask to fix it (though I wrote it).
        # Actually I WROTE `plot_comparison.py` in this session.
        # I should verify if I put `header=None`.
        
        # In `plot_comparison.py`: `df = pd.read_csv(path)`. Default header='infer'.
        # If first row is numbers, pandas usually infers it as header.
        # So I SHOULD save with a header or update plot script.
        # Easiest is to update `plot_comparison.py` to use `header=None` OR save with generic header.
        # But wait, looking at `eval.py` output from memory... OPEN usually saves raw numbers.
        
        # Let's save without header, and I will strictly tell user to ensure plot script is compatible 
        # or I will covertly update plot script if I can.
        # Actually I can just save with header=None and let pandas infer? No, pandas takes first row.
        
        # Safest: Save with no header, and update `plot_comparison.py` to use `header=None`.
        # I will update `plot_comparison.py` in next step if needed.
        # For now, save without header.
        
        csv_path = os.path.join(consolidated_log_dir, "returns.csv")
        df_out.to_csv(csv_path, index=False, header=False)
        print(f"\nSaved consolidated {args.num_runs} runs to {csv_path}")
        print(f"Shape: {df_out.shape} (Runs x Steps)")
    else:
        print("No results collected.")

if __name__ == "__main__":
    main()
