import sys
sys.path.append(r'../../')

import jax.numpy as jnp
from alphatensor_quantum.src.experiment import config
import pickle
import numpy as np
import random

np.random.seed(2024)
random.seed(2024)

num_eval_data = 100#0
num_qubits = 5
use_gadgets = True

configuration_evals = []
baseline_tgates_evals = []
baseline_times_evals = []


for ii in range(num_eval_data):
    if ii % 100 == 0:
        print(f'=== Generate eval data {ii}/{num_eval_data}')

    configuration_eval, baseline_tgates, baseline_times = config.get_demo_config(
        use_gadgets=use_gadgets, 
        num_data = 1,
        num_qubits = num_qubits,
        todd_path = '/u/rzen/TOpt/bin/TOpt', ##TODO: Change with your TODD path
    )

    configuration_evals.append(configuration_eval)
    baseline_tgates_evals.append(baseline_tgates[0])
    baseline_times_evals.append(baseline_times[0])

baseline_tgates_evals = np.array(baseline_tgates_evals)
print(f'Mean gate: {jnp.mean(baseline_tgates_evals):.4f}, std: {jnp.std(baseline_tgates_evals):.4f}')
print(f'Max gate: {jnp.max(baseline_tgates_evals):.4f}, std: {jnp.min(baseline_tgates_evals):.4f}')

times_evals = jnp.array(baseline_times_evals)

if use_gadgets:
    pickle.dump((configuration_evals, baseline_tgates_evals, baseline_times_evals), 
            open('eval-data-%d-%d-gadgets.p' % (num_eval_data,num_qubits), 'wb'))
else:
    pickle.dump((configuration_evals, baseline_tgates_evals, baseline_times_evals), 
            open('eval-data-%d-%d-nongadgets.p' % (num_eval_data,num_qubits), 'wb'))