# Optimizing T-count in General Quantum Circuits with AlphaTensor-Quantum

This repository accompanies the paper

> Zen, Remmy and Nägele, Maximilian and Marquardt, Florian. Reusability Report: Optimizing T-count in General Quantum Circuits with AlphaTensor-Quantum.
Submitted to *Nature Machine Intelligence*.

, which is a reusability report for the paper

> Ruiz, F. J. R. et al. Quantum Circuit Optimization with AlphaTensor.
*Nature Machine Intelligence* (Under Review).

There are two *new* directories:

- `/script` contains the scripts to train and evaluate general agent with random circuits of varying qubit numbers and single agent with random circuit of fixed qubit number using AlphaTensor-Quantum.

- `/src/experiment` contains the code for the configuration, the agent, and utility files.

## Installation

A machine with Python 3 installed is required, ideally with a
hardware accelerator such as a GPU or TPU. The required dependencies (assuming
an Nvidia GPU is available) can be installed by executing
`pip3 install -r alphatensor_quantum/src/experiment/requirements.txt`.

You would need to install TODD for the baseline:

- Clone TODD Github somewhere in your home folder.
```
git clone git@github.com:Luke-Heyfron/TOpt.git
```

- Follow the installation process
```
cd TOpt
mkdir bin
make all
```

- You need change the path     
```
todd_path = '/u/rzen/TOpt/bin/TOpt', ##TODO: Change with your TODD path
```

in the code to your installed path.


## Usage

- `/script/train_general_agent.py`: Train a general agent with random circuits for varying qubit numbers. 

- `/script/train_single_agent.py`: Train a single agent with random circuits for a fixed qubit number. 

- `/script/generate_eval_data.py`: Generate random circuits for evaluation. 

- `/script/evaluation_general.py`: Evaluate a general agent for a given evaluation data and trained model. 

- `/script/evaluation_single.py`: Evaluate a single agent for a given evaluation data and trained model. 

## Citing this work

If you use the code or data in this package, please cite:

```latex
@article{alphatensor_quantum_general,
      author={Zen, Remmy and Nägele, Maximilian and Marquardt, Florian},
      title={Reusability Report: Optimizing T-count in General Quantum Circuits with AlphaTensor-Quantum},
      journal = {Nature Machine Intelligence (Under Review)},
      year={2025},
}
```

and the original paper: 

```latex
@article{alphatensor_quantum,
      author={Ruiz, Francisco J. R. and Laakkonen, Tuomas and Bausch, Johannes and Balog, Matej and Barekatain, Mohammadamin and Heras, Francisco J. H. and Novikov, Alexander and Fitzpatrick, Nathan and Romera-Paredes, Bernardino and van de Wetering, John and Fawzi, Alhussein and Meichanetzidis, Konstantinos and Kohli, Pushmeet},
      title={Quantum Circuit Optimization with {A}lpha{T}ensor},
      journal = {Nature Machine Intelligence (Under Review)},
      year={2024},
}
```
