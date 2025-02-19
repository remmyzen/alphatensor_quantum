# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Configuration hyperparameters for the AlphaTensor-Quantum demo."""

import dataclasses
import os

from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import tensors
from alphatensor_quantum.src.experiment import utils
import numpy as np
import time

@dataclasses.dataclass(frozen=True, kw_only=True)
class LossParams:
  """Hyperparameters for the loss.

  Attributes:
    init_demonstrations_weight: The initial weight of the loss corresponding to
      the episodes from synthetic demonstrations.
    demonstrations_boundaries_and_scales: The boundaries and scales for the
      synthetic demonstrations weight, to be used in a
      `piecewise_constant_schedule` Optax schedule.
  """
  init_demonstrations_weight: float
  demonstrations_boundaries_and_scales: dict[int, float]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExperimentParams:
  """Hyperparameters for the experiment.

  Attributes:
    batch_size: The batch size.
    num_mcts_simulations: The number of MCTS simulations to run per each action
      taken.
    num_training_steps: The total number of training steps.
    avg_return_smoothing: The smoothing factor for the average return, for
      reporting purposes only.
    eval_frequency_steps: The frequency (expressed in number of training steps)
      to report the running statistics. This is for reporting purposes only.
    loss: The loss parameters.
  """
  batch_size: int = 2_048
  num_mcts_simulations: int = 800
  num_training_steps: int = 1_000_000
  avg_return_smoothing: float = 0.9
  eval_frequency_steps: int = 1_000
  loss: LossParams


@dataclasses.dataclass(frozen=True, kw_only=True)
class DemoConfig:
  """All the hyperparameters for the demo."""
  exp_config: ExperimentParams
  env_config: config_lib.EnvironmentParams
  net_config: config_lib.NetworkParams
  opt_config: config_lib.OptimizerParams
  dem_config: config_lib.DemonstrationsParams




def get_demo_config(use_gadgets: bool, num_data: int, optimize_input_circuit:bool = True, todd_path:str = None,
                    num_qubits = 5, exp_type = 0) -> DemoConfig:
  """Returns the config hyperparameters for the demo.

  Args:
    use_gadgets: Whether to consider gadgetization. This parameter affects not
      only the environment, but also the default target circuits.
    num_data: The number of data points to generate.
    optimize_input_circuit: Whether to optimize the input circuit with PyZX and TODD
      before generating the Waring matrix.
    todd_path: The path to the TODD binary. If `None`, TODD is not used.
    num_qubits: The number of qubits for the target circuits. Could be list of integers for training general agent.
    exp_type = 0 -> combination of RL + demonstrations, 1 -> only demonstrations, 2 -> only RL.

  Returns:
    The hyperparameters for the demo.
  """

  target_circuit_types = []
  tgates_baseline = []
  times = []


  ## Generate data
  for _ in range(num_data):
    ## Randomly sample the number of qubits for general agent
    if isinstance(num_qubits, list):
      num_qubit = np.random.choice(num_qubits)
    else:
      num_qubit = num_qubits

    ## circuit_size sample from 5n to 15n
    ## num_t sample from 20% to 60% of circuit size
    circuit_size = np.random.choice(range(5*num_qubit, 15*num_qubit))
    num_t_ratio =  np.random.choice(range(20, 60))
    num_t = circuit_size * num_t_ratio // 100

    random_circuit, qiskit_circuit = utils.generate_random_circuit(num_qubit, num_t, circuit_size)
    
    ## Optimize with PyZX and TODD 
    time_start = time.time()
    pyzx_circuit = utils.optimize_circuit_pyzx(qiskit_circuit)
    pyzx_todd_circuit = utils.optimize_circuit_todd(todd_path, pyzx_circuit)
    tgate_pyzx_todd = utils.count_t_gate(pyzx_todd_circuit)
    times.append(time.time() - time_start)

    ## Count T gates for logging
    tgates_baseline.append(tgate_pyzx_todd)
         
    ## Get the Waring matrix for alpha tensor
    ### PyZX optimized circuit as input for AlphaTensor
    if optimize_input_circuit:
      ## Need to process the circuit to include only the T + CX gate. 
      ## Because PyZX introduce S and Z gates. 
      processed_circuit = utils.process_circuit(pyzx_circuit)
      
      A = utils.generate_waring_matrix(utils.circuit_to_gate_list(processed_circuit))

      T = np.einsum('im,jm,km->ijk',A,A,A) % 2
    else: 
      A = utils.generate_waring_matrix(random_circuit)
      T = np.einsum('im,jm,km->ijk',A,A,A) % 2
    
    target_circuit_types.append(T)
  
  ## Both RL and Demonstrations 
  if exp_type == 0:
    exp_config = ExperimentParams(
        batch_size=128,
        num_mcts_simulations=80,
        num_training_steps=2_000, #100_000,
        eval_frequency_steps=1000,
        loss=LossParams(
            init_demonstrations_weight=1.0,
            # Progressively reduce the weight of the demonstrations in favour of
            # the acting episodes.
            demonstrations_boundaries_and_scales={
                60: 0.99, 200: 0.5, 5_000: 0.2, 10_000: 0.1
            },
        ),
    )
  ## Only Demonstrations
  elif exp_type == 1:
    exp_config = ExperimentParams(
        batch_size=128,
        num_mcts_simulations=80,
        num_training_steps= 100_000,
        eval_frequency_steps=1000,
        loss=LossParams(
            init_demonstrations_weight=1.0,
            # Progressively reduce the weight of the demonstrations in favour of
            # the acting episodes.
            demonstrations_boundaries_and_scales={
                60: 1.0, 200: 1.0, 5_000: 1.0, 10_000: 1.0
            },
        ),
    )
  ## Only RL
  elif exp_type == 2:
    exp_config = ExperimentParams(
        batch_size=128,
        num_mcts_simulations=80,
        num_training_steps=100_000,
        eval_frequency_steps=1000,
        loss=LossParams(
            init_demonstrations_weight=0.0,
            # Progressively reduce the weight of the demonstrations in favour of
            # the acting episodes.
            demonstrations_boundaries_and_scales={
                60: 0.0, 200: 0.0, 5_000: 0.0, 10_000: 0.0
            },
        ),
    )

  if isinstance(num_qubits, list):
    max_size = max(num_qubits)
  else:
    max_size = num_qubits

  env_config = config_lib.EnvironmentParams(
      max_num_moves=30,
      target_circuit_types=target_circuit_types,
      num_past_factors_to_observe=6,
      change_of_basis=config_lib.ChangeOfBasisParams(
          prob_zero_entry=0.9,
          num_change_of_basis_matrices=80,
          prob_canonical_basis=0.16,
      ),
      max_size = max_size,
      use_gadgets=use_gadgets
  )
  net_config = config_lib.NetworkParams(
      num_layers_torso=4,
      attention_params=config_lib.AttentionParams(
          num_heads=8,
          head_depth=8,
          mlp_widening_factor=2,
      ),
  )
  opt_config = config_lib.OptimizerParams(
      init_lr=1e-3,
      lr_scheduler_transition_steps=5_000,
  )
  dem_config = config_lib.DemonstrationsParams(
      max_num_factors=30,
      max_num_gadgets=5,
      prob_include_gadget=0.9 if use_gadgets else 0.0,
  )
  return DemoConfig(
      exp_config=exp_config,
      env_config=env_config,
      net_config=net_config,
      opt_config=opt_config,
      dem_config=dem_config,
  ), tgates_baseline, times
