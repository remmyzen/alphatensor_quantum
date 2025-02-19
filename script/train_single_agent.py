# # Training Single Agent with AlphaTensor-Quantum
# 
# Train AlphaTensor-Quantum with random circuits of a fixed qubit number.

import sys
sys.path.append(r'../../')

import time
import warnings
warnings.filterwarnings("ignore", message=".*Explicitly requested dtype <class 'jax.numpy.float64'>.*")

from absl import app
import jax
import jax.numpy as jnp
import os
import random
import numpy as np
from alphatensor_quantum.src.experiment import agent as agent_lib
from alphatensor_quantum.src.experiment import config
import pickle
import copy

np.random.seed(2024)
random.seed(2024)

## Number of training data
num_data= 10000

## Number of qubits
num_qubits = 5

## Use gadgets
use_gadgets = True

# exp_type = 0 -> combination of RL + demonstrations, 1 -> only demonstrations, 2 -> only RL.
exp_type = 0

## Set up the hyperparameters and training data.
configuration, baseline_tgates, baseline_times = config.get_demo_config(
    use_gadgets=use_gadgets, 
    num_data = num_data,
    num_qubits = num_qubits,
    todd_path = '/TOpt/bin/TOpt', ##TODO: Change with your TODD path
    exp_type = exp_type
)
exp_config = configuration.exp_config

print(f'Average baseline T-count: {np.mean(baseline_tgates):.4f}, std: {np.std(baseline_tgates):.4f}')
print(f'Average baseline optimization time: {np.mean(baseline_times):.4f}, std: {np.std(baseline_times):.4f}')

## Initialize the agent and the run state.
agent = agent_lib.Agent(configuration)
run_state = agent.init_run_state(jax.random.PRNGKey(2024))
tcounts = []
avg_returns = []

## Main training loop.
start = time.time()
for step in range(
    0, exp_config.num_training_steps, exp_config.eval_frequency_steps
):
    time_start = time.time()
    run_state = agent.run_agent_env_interaction(step, run_state)
    time_taken = (time.time() - time_start) / exp_config.eval_frequency_steps
    # Keep track of the average return (for reporting purposes). We use a
    # debiased version of `avg_return` that only includes batch elements with at
    # least one completed episode.
    num_games = run_state.game_stats.num_games
    avg_return = run_state.game_stats.avg_return
    avg_return = jnp.sum(
      jnp.where(
          num_games > 0,
          avg_return / (1.0 - exp_config.avg_return_smoothing ** num_games),
          0.0
      ),
      axis=0
    ) / jnp.sum(num_games > 0, axis=0)
    print(
      f'=============== Step: {step + exp_config.eval_frequency_steps} .. '
      f'Running Average Returns: {jnp.mean(avg_return)} .. '
      f'Time taken: {time_taken} seconds/step'
    )
 
    tcounts.append(-run_state.game_stats.best_return)
    avg_returns.append(avg_return)   

    print(f'Average T-count for training circuits: {jnp.mean(-run_state.game_stats.best_return)}, std: {jnp.std(-run_state.game_stats.best_return)}')
    print(f'Average baseline T-count for training circuits: {np.mean(baseline_tgates)},  std: {np.std(baseline_tgates)}')    
    
## Save the runstate for running the agent.
pickle.dump(run_state, open(f'runstate-single-{num_qubits}.p', 'wb'))

total_time = time.time() - start
print(f'Total training time: {total_time}s')



