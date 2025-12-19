
import os
import argparse
import time
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial
from flax.training.train_state import TrainState

# Reuse existing training components
from train import Actor, Critic, Transition, symlog
from utils import (
    GymnaxGymWrapper,
    GymnaxLogWrapper,
    FlatWrapper,
    BraxGymnaxWrapper,
    ClipAction,
    TransformObservation,
    NormalizeObservation,
    NormalizeReward,
    VecEnv,
)
from configs import all_configs
import gymnax

# Hyperparameter Tables (from User Image)
ADAM_PARAMS = {
    "asterix": {"LR": 0.003, "b1": 0.9, "b2": 0.999, "ANNEAL_LR": True},
    "freeway": {"LR": 0.001, "b1": 0.9, "b2": 0.99, "ANNEAL_LR": True},
    "breakout": {"LR": 0.01, "b1": 0.9, "b2": 0.99, "ANNEAL_LR": True},
    "spaceinvaders": {"LR": 0.007, "b1": 0.9, "b2": 0.99, "ANNEAL_LR": True},
    "ant": {"LR": 0.0003, "b1": 0.99, "b2": 0.99, "ANNEAL_LR": True},
    # Defaults for new envs
    "humanoid": {"LR": 0.0003, "b1": 0.99, "b2": 0.99, "ANNEAL_LR": True},
    "halfcheetah": {"LR": 0.0003, "b1": 0.99, "b2": 0.99, "ANNEAL_LR": True},
    "pendulum": {"LR": 0.001, "b1": 0.9, "b2": 0.99, "ANNEAL_LR": True},
}

RMSPROP_PARAMS = {
    "asterix": {"LR": 0.002, "decay": 0.99, "ANNEAL_LR": True},
    "freeway": {"LR": 0.001, "decay": 0.999, "ANNEAL_LR": True},
    "breakout": {"LR": 0.002, "decay": 0.99, "ANNEAL_LR": False},
    "spaceinvaders": {"LR": 0.009, "decay": 0.99, "ANNEAL_LR": True},
    "ant": {"LR": 0.0008, "decay": 0.99, "ANNEAL_LR": False},
    # Defaults
    "humanoid": {"LR": 0.0008, "decay": 0.99, "ANNEAL_LR": False},
    "halfcheetah": {"LR": 0.0008, "decay": 0.99, "ANNEAL_LR": False},
    "pendulum": {"LR": 0.001, "decay": 0.99, "ANNEAL_LR": False},
}


def make_train(config, optimizer_name="adam"):
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // int(config["NUM_STEPS"]) // int(config["NUM_ENVS"])
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env_name = config["ENV_NAME_SHORT"] # e.g. 'breakout'
    
    # Select Hyperparams
    if optimizer_name == "adam":
        hparams = ADAM_PARAMS.get(env_name, ADAM_PARAMS["ant"]) # Fallback to ant
        lr_val = hparams["LR"]
        anneal = hparams["ANNEAL_LR"]
        kwargs = {"b1": hparams.get("b1", 0.9), "b2": hparams.get("b2", 0.999)}
    elif optimizer_name == "rmsprop":
        hparams = RMSPROP_PARAMS.get(env_name, RMSPROP_PARAMS["ant"])
        lr_val = hparams["LR"]
        anneal = hparams["ANNEAL_LR"]
        kwargs = {"decay": hparams.get("decay", 0.99)}
    else:
        # Default fallback
        lr_val = 3e-4
        anneal = True
        kwargs = {}

    print(f"Setting up {optimizer_name} for {env_name}: LR={lr_val}, Anneal={anneal}, {kwargs}")

    # OPTIMIZER SETUP
    if anneal:
        lr_schedule = optax.linear_schedule(
            init_value=lr_val,
            end_value=0.0,
            transition_steps=config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
        )
        if optimizer_name == "adam":
            tx = optax.adam(learning_rate=lr_schedule, **kwargs)
        elif optimizer_name == "rmsprop":
            tx = optax.rmsprop(learning_rate=lr_schedule, **kwargs)
        else:
             tx = optax.adam(learning_rate=lr_schedule)
    else:
        if optimizer_name == "adam":
            tx = optax.adam(learning_rate=lr_val, **kwargs)
        elif optimizer_name == "rmsprop":
            tx = optax.rmsprop(learning_rate=lr_val, **kwargs)
        else:
            tx = optax.adam(learning_rate=lr_val)
            
    # Combine with Gradient Clipping
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        tx
    )


    @jax.jit
    def train(rng):
        # 1. INIT ENV
        if "Brax-" in config["ENV_NAME"]:
            name = config["ENV_NAME"].split("Brax-")[1]
            env, env_params = BraxGymnaxWrapper(env_name=name), None
            if config.get("CLIP_ACTION"): env = ClipAction(env)
            env = GymnaxLogWrapper(env)
            if config.get("SYMLOG_OBS"): env = TransformObservation(env, transform_obs=symlog)
            env = VecEnv(env)
            if config.get("NORMALIZE"):
                env = NormalizeObservation(env)
                env = NormalizeReward(env, config["GAMMA"])
            actor = Actor(env.action_space(env_params).shape[0], config=config)
            critic = Critic(config=config)
            init_x = jnp.zeros(env.observation_space(env_params).shape)
        else:
            env, env_params = gymnax.make(config["ENV_NAME"])
            env = GymnaxGymWrapper(env, env_params, config)
            env = FlatWrapper(env)
            env = GymnaxLogWrapper(env)
            env = VecEnv(env)
            actor = Actor(env.action_space, config=config)
            critic = Critic(config=config)
            init_x = jnp.zeros(env.observation_space)

        # 2. INIT PARAMS
        rng, _rng = jax.random.split(rng)
        actor_params = actor.init(_rng, init_x)
        critic_params = critic.init(_rng, init_x)

        rng, rng_act, rng_crit = jax.random.split(rng, 3)
        train_state_actor = TrainState.create(
            apply_fn=actor.apply, params=actor_params, tx=tx
        )
        train_state_critic = TrainState.create(
            apply_fn=critic.apply, params=critic_params, tx=tx
        )

        # 3. TRAIN LOOP
        all_rng = jax.random.split(_rng, config["NUM_ENVS"] + 1)
        rng, _rng = all_rng[0], all_rng[1:]
        obsv, env_state = env.reset(_rng, env_params)

        runner_state = (
            train_state_actor,
            train_state_critic,
            env_state,
            obsv,
            rng,
        )

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state_actor, train_state_critic, env_state, last_obs, rng = runner_state
                rng, _rng = jax.random.split(rng)
                
                # Action
                pi, _ = actor.apply(train_state_actor.params, last_obs)
                value, _ = critic.apply(train_state_critic.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # Step
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
                
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                new_runner_state = (train_state_actor, train_state_critic, env_state, obsv, rng)
                return new_runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
            
            # PPO UPDATE
            train_state_actor, train_state_critic, env_state, last_obs, rng = runner_state
            
            # GAE
            last_val, _ = critic.apply(train_state_critic.params, last_obs)
            # Handle done mask approx (traj_batch.done is [T, N])
            # For simplicity, we use the done from the last step of trajectory for bootstrapping? 
            # Actually standard PPO bootstrapping usually uses 'done' from specific step.
            # Here we follow train.py style logic implicitly
            
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae
                
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    train_state_actor, train_state_critic = train_state
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(actor_params, critic_params, traj_batch, gae, targets):
                        pi, _ = actor.apply(actor_params, traj_batch.obs)
                        value, _ = critic.apply(critic_params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value Loss
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Actor Loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, None

                    grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)
                    (loss, _), (grads_actor, grads_critic) = grad_fn(
                        train_state_actor.params, train_state_critic.params, traj_batch, advantages, targets
                    )
                    
                    train_state_actor = train_state_actor.apply_gradients(grads=grads_actor)
                    train_state_critic = train_state_critic.apply_gradients(grads=grads_critic)
                    return (train_state_actor, train_state_critic), loss

                # Batching logic
                train_state_actor, train_state_critic, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                
                # Flatten batch [T, N, ...] -> [T*N, ...]
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                
                permutation = jax.random.permutation(_rng, batch_size)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch
                )
                
                (train_state_actor, train_state_critic), _ = jax.lax.scan(
                    _update_minbatch, 
                    (train_state_actor, train_state_critic), 
                    minibatches
                )
                return (train_state_actor, train_state_critic, rng), None

            update_state = (train_state_actor, train_state_critic, rng)
            update_state, _ = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            
            train_state_actor, train_state_critic, rng = update_state
            
            # Metrics
            metric = traj_batch.info
            fitness = metric["returned_episode_returns"][-1].mean() # Average across envs
            
            return (train_state_actor, train_state_critic, env_state, last_obs, rng), fitness

        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return metrics

    return train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", nargs="+", required=True, help="List of environment names (e.g. breakout ant)")
    parser.add_argument("--optimizer", nargs="+", default=["adam", "rmsprop"], choices=["adam", "rmsprop", "sgd"], help="List of optimizers")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of parallel runs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Convert single string args to lists if needed
    envs = args.env if isinstance(args.env, list) else [args.env]
    optimizers = args.optimizer if isinstance(args.optimizer, list) else [args.optimizer]

    print(f"Running evaluation for Envs: {envs}")
    print(f"Optimizers: {optimizers}")
    
    for environment in envs:
        for optimizer in optimizers:
            if environment not in all_configs:
                print(f"Skipping {environment}: Not found in configs.")
                continue
            
            config = all_configs[environment].copy() # Copy to avoid mutation issues across loops
            config["ENV_NAME_SHORT"] = environment
            
            print(f"\n{'='*50}")
            print(f"Evaluating {environment} with {optimizer}")
            print(f"{'='*50}")
            
            # Create Directories
            save_path = f"baselines/{environment}/{optimizer}"
            os.makedirs(save_path, exist_ok=True)
            
            # Check if already done? (For now, overwrite)

            # Compilation & Execution
            try:
                # Re-JIT per config/optimizer pair as shapes/logic might change
                # Note: We rebuild make_train to capture new valid config closures
                train_fn = make_train(config, optimizer)
            
                rng = jax.random.PRNGKey(args.seed)
                rngs = jax.random.split(rng, args.num_runs)
                
                start_time = time.time()
                
                # Jit + Vmap running N runs
                # We JIT here inside the loop to avoid caching incorrect shapes from previous envs
                vmap_train = jax.jit(jax.vmap(train_fn, in_axes=(0)))
                
                print("Compiling and Training...")
                metrics = vmap_train(rngs) # Shape: [Num_Runs, Num_Updates]
                
                elapsed = time.time() - start_time
                print(f"Training Complete. Time: {elapsed:.2f}s")
                
                # Metrics
                returns = metrics 
                final_return = returns[:, -1].mean()
                print(f"Final Mean Return ({environment}/{optimizer}): {final_return:.2f}")

                # Save Data
                df = pd.DataFrame(returns.T) 
                csv_file = f"{save_path}/returns.csv"
                df.to_csv(csv_file, index=False)
                print(f"Saved results to {csv_file}")
                
            except Exception as e:
                print(f"FAILED to run {environment} with {optimizer}")
                print(e)
                import traceback
                traceback.print_exc()
