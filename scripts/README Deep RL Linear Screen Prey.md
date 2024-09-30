# Result Notebook Documentation

This notebook implements Reinforcement Learning (RL) experiments using a Deep Deterministic Policy Gradient (DDPG) agent in a multi-modal maze environment. The agent's primary objective is to capture prey under various static and dynamic scenarios, with training and evaluation focused on different environmental conditions and agent hyperparameters. This document outlines the key components of the notebook, including the parameters, agent setup, RL training, and hyperparameter optimization using Optuna.

## Imports
The following libraries are imported to facilitate the environment simulation, agent training, and visualization of results:

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multimodal_mazes  # Custom module for RL environment
import seaborn as sns
```

- **numpy**: For numerical computations.
- **matplotlib**: For plotting and visualizing results.
- **tqdm**: For progress tracking during training.
- **multimodal_mazes**: A custom module for creating and running RL experiments.
- **seaborn**: For enhanced data visualization.

## Parameters, Hyperparameters, and Helper Functions

### Environment Hyperparameters
The environment dimensions and other environmental settings are defined by the `environment_hyperparameters()` function, which includes:
- `width`, `height`: Dimensions of the environment.
- `pk`, `pk_hw`: Parameters defining the local sensory information available to the agent.
- `capture_radius`: The radius within which the agent can capture prey.

### Agent Hyperparameters
The agent-specific settings are provided by the `agent_hyperparameters()` function, which includes:
- `input_dim`, `hidden_dim`, `action_dim`: Dimensions for the input, hidden layers, and output for the agent’s neural networks.
- `actor_lr`, `critic_lr`: Learning rates for the actor and critic networks.
- `gamma`: Discount factor for future rewards.
- `tau`: Parameter for the soft update of target networks.
- `channels`, `location`: Additional parameters defining the agent's sensory input and starting location.

### Scenario-Specific Hyperparameters
There are two scenario types covered in the experiments, each with their own set of hyperparameters:
- **Static Scenario**: Prey does not move, and agent training occurs with a fixed environment.
- **Constant Scenario**: Prey follows a linear movement pattern, and the agent is trained to handle constant motion.

The respective functions `static_hyperparamters()` and `constant_hyperparameters()` define these parameters:
- `noise`, `n_prey`, `n_steps`: Noise in sensory data, number of prey, and number of steps per trial.
- `scenario`, `motion`, `multisensory`, `speed`: Scenario type, prey motion, and sensory input type.

## Static Training Test
In this section, the agent is trained in a static environment where the prey remains in a fixed position. The training process is conducted using the `train_RL()` method for 10,000 training trials, after which performance metrics are plotted to assess the training progress.

```python
training_evaluator.train_RL(training_trials=10000)
training_evaluator.training_plots(plot_training_lengths=True, plot_first_5_last_5=True)
```

## Optuna Hyperparameter Search
The Optuna framework is used to optimize the hyperparameters of the DDPG agent. Key parameters such as actor learning rate, critic learning rate, hidden dimensions, and exploration noise are explored to find the best configuration for maximizing the agent's reward during training.

### Objective Function
The `ddpg_objective(trial)` function defines the search space for Optuna to explore:
- `trial.suggest_float()` and `trial.suggest_int()` are used to sample the learning rates, dimensions, and noise scale for the agent.
- The agent is trained for 1000 episodes, and the total reward is reported to Optuna.
- The `MedianPruner` is used to prune unpromising trials early based on the reward metrics.

### Optuna Study
A study is created to run 100 trials of hyperparameter tuning:
```python
study.optimize(ddpg_objective, n_trials=100)
```
After the optimization, the best hyperparameters are printed for use in future training experiments.

```python
Best parameters: {'actor_lr': 1.0506603198418105e-05, 'critic_lr': 0.0005747127451203138, 'hidden_dim': 10, 'gamma': 0.9197484689502473, 'tau': 0.0009655446721377356, 'noise_scale': 0.49798430512987657}
```

## Key Functions
- **`train_RL()`**: Trains the RL agent over a specified number of trials.
- **`training_plots()`**: Visualizes the training process, including the performance over time and comparisons between the first and last trials.
- **`ddpg_objective()`**: Defines the objective for optimizing the DDPG agent's hyperparameters using Optuna.

## Key Metrics
- **Total Reward**: The cumulative reward over all episodes, used as the primary metric for assessing the performance of different hyperparameter configurations.
- **Training Lengths**: Length of time (or number of steps) for which the agent is trained.
- **Parameter Importance**: Visualizations provided by Optuna to display the importance of different hyperparameters in determining agent success.

This notebook demonstrates a systematic approach to training a DDPG agent in a multi-modal maze environment, optimizing its performance using Reinforcement Learning techniques and hyperparameter tuning.
