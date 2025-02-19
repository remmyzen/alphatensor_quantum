import sys
sys.path.append(r'../../')

import time
import warnings
warnings.filterwarnings("ignore", message=".*Explicitly requested dtype <class 'jax.numpy.float64'>.*")

from absl import app
import jax
import jax.numpy as jnp

from alphatensor_quantum.src.experiment import agent as agent_lib
from alphatensor_quantum.src.experiment import config
import pickle

def evaluation(configuration_evals, run_state, eval_type):
  ##Currently loop over environment with 1 circuit. 
  ##NOTE: could be optimized with vmap

  tcounts_eval = []
  for ii, configuration_eval in enumerate(configuration_evals):
    if ii % 10 == 0:
      print(f'Evaluation {ii}/{len(configuration_evals)}')

    agent = agent_lib.Agent(configuration_eval)
    ## Initialize just to create the environment but do not need the run state
    run_state_new = agent.init_run_state(jax.random.PRNGKey(2024))
    run_state_new = run_state_new._replace(params = run_state.params)
    if eval_type == 'policy':
        run_state_eval = agent.evaluate_policy(run_state_new)
    if eval_type == 'greedy':
        run_state_eval = agent.evaluate_greedy_step(run_state_new)
    if eval_type == 'random':
        run_state_eval = agent.evaluate_random_step(run_state_new)

    tcounts_eval.append(-run_state_eval.game_stats.best_return)

  return tcounts_eval

## Eval data filename
eval_data_filename = 'eval-data-100-5-gadgets.p'

## Model filename
model_filename = 'runstate-single-5.p'

use_gadgets = True
eval_type = 'greedy' 
optimize_input_circuit = True

## Open eval data
configuration_evals, tgate_baseline_evals, _ = pickle.load(open(eval_data_filename, 'rb'))
          
start = time.time()

run_state_eval = pickle.load(open(model_filename, 'rb'))

tcounts_eval = jnp.array(evaluation(configuration_evals, run_state_eval, eval_type))

print(f'Average T-count for eval circuit: {jnp.mean(tcounts_eval)}, std: {jnp.std(tcounts_eval)}')
print(f'Average baseline T-count for eval circuit: {jnp.mean(tgate_baseline_evals)}, std: {jnp.std(tgate_baseline_evals)}')

total_time = time.time() - start
print(f'Total evaluation time: {total_time}s')     
print(f'Evaluation time per circuit:  {total_time / len(configuration_evals)}s')      
