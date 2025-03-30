# Result Notebook Documentation

This notebook implements Reinforcement Learning (RL) experiments using a Deep Deterministic Policy Gradient (DDPG) agent in a multi-modal maze environment. The agent's primary objective is to capture prey under various static and dynamic scenarios, with training and evaluation focused on different environmental conditions and agent hyperparameters. This document outlines the key components of the notebook, including the parameters, agent setup, RL training, and hyperparameter optimization using Optuna.

**Note: This implementation is not yet fully completed, and further refinements are needed to finalize the results and analysis.**

## Imports
The following libraries are imported to facilitate the environment simulation, agent training, and visualization of results:

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multimodal_mazes  # Custom module for RL environment
import seaborn as sns
import h5py
import gc
```

- **numpy**: For numerical computations.
- **matplotlib**: For plotting and visualizing results.
- **tqdm**: For progress tracking during training.
- **multimodal_mazes**: A custom module for creating and running RL experiments.
- **seaborn**: For enhanced data visualization.
- **h5py**: For saving trajectory data in HDF5 format.
- **gc**: For garbage collection and memory management.

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
In this section, the agent is trained in a static environment where the prey remains in a fixed position. The training process is conducted using the `run_trials()` method for 400 trials, after which performance metrics are plotted to assess the training progress.

```python
evaluator.run_trials(400, True)
evaluator.training_plots(paths=True)
evaluator.training_plots(reward=True)
evaluator.training_plots(animate=True)
```

## Optuna Hyperparameter Search
The Optuna framework is used to optimize the hyperparameters of the DDPG agent. Key parameters such as actor learning rate, critic learning rate, hidden dimensions, and exploration noise are explored to find the best configuration for maximizing the agent's reward during training.
*Note: While Optuna has been used for hyperparameter tuning, the current script version does not include the Optuna setup or calls.*

## Constant Movement Training Test
This section evaluates agent performance when prey moves linearly. Multiple scenarios (`case 1`, `case 2`, `case 3`) are tested across a range of prey speeds. Each scenario's results are aggregated, processed, and visualized to assess capture rates and robustness to motion:
- Trial runs for each speed and scenario.
- Capture success rates and standard deviations are calculated.
- Line plots and error bars display performance metrics.
    
## Result Processing
After all trials, the results are processed:
- Capture success across scenarios is calculated and plotted.
- A figure visualizes capture success against prey speed for all test cases.
- Averages and standard deviations are computed to compare performance stability.
    
## Trajectory Plotting
Agent trajectories are visualized to understand spatial behavior and strategies:
- Prey-agent paths are analyzed for each speed and scenario.
- Mean trajectories are plotted to highlight consistent patterns.
- Path directionality and agent response to prey movement are captured.
    
## Trajectory Grouping
Trajectories are grouped by prey starting direction (`LEFT`, `RIGHT`, `CENTER`) and categorized by effective prey movement direction:
- Coordinates are binned and averaged.
- This enables trajectory comparisons across directional biases.
    
## Saving Trajectories as h5
All trajectory data is saved in HDF5 format using `h5py` for future analysis and reproducibility:
```python
with h5py.File('deep_rl_mean_coords.h5', 'w') as h5file:
    recursively_save_dict_contents(h5file, direction_coords)
```

## Key Functions
- **`run_trials()`**: Trains the RL agent over a specified number of trials.
- **`training_plots()`**: Visualizes the training process, including performance over time and animations of agent behavior.
- **`ddpg_objective()`**: Defines the objective for optimizing the DDPG agent's hyperparameters using Optuna.
- **`recursively_save_dict_contents()`**: Helper function to save nested trajectory data in HDF5 format.

## Key Metrics
- **Total Reward**: The cumulative reward over all episodes, used as the primary metric for assessing the performance of different hyperparameter configurations.
- **Capture Success**: The rate at which the agent successfully captures the prey.
- **Training Lengths**: Number of steps taken to complete episodes.
- **Parameter Importance**: Visualizations provided by Optuna to display the influence of different hyperparameters on success.
- **Trajectory Patterns**: Spatial distribution and consistency of agent movements across trials.

This notebook demonstrates a systematic approach to training a DDPG agent in a multi-modal maze environment, optimizing its performance using Reinforcement Learning techniques and hyperparameter tuning.
