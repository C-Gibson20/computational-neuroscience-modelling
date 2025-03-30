# Computational Neuroscience Modelling

This repository is part of an ongoing research project that models predator-prey interactions within a neuroscience experimental setup using multi-modal maze environments. The project explores agent behavior in these environments using rule-based approaches, reinforcement learning (RL) function approximation, and Deep Deterministic Policy Gradient (DDPG) agents.

The repository contains a range of files that focus on the implementation and evaluation of different agent models, including custom RL environments, predator-prey dynamics, and experimental results for various scenarios. The included code reflects my personal contribution to the project, covering my work on agent implementation, experimental trial setups, and the collection of results. The broader **multimodal_mazes** library, which includes additional environments and scenarios, is not included as the project is still under development and not yet public.

# Repository Structure

## 1. multimodal_mazes/RL_Agents
This directory contains the core RL agent implementations, including DDPG agents and RL function approximation methods.

### Files:
- **Deep_RL.py**: Implements a DDPG agent with continuous action space, designed to capture prey in the environment.
- **RL_function_method.py**: Implements an RL agent using function approximation methods to capture prey in the environment.

### Key Classes:
- **DDPGAgent**: A DDPG agent that handles continuous control tasks for the predator-prey simulation.
- **RLApproximationAgent**: An agent using RL function approximation methods (such as Q-learning) to optimize its behavior in the environment.

## 2. multimodal_mazes/agents/prey
This directory models prey behavior and interactions in the environment.

### Files:
- **prey_continuous.py**: Defines prey behavior when moving continuously through the environment.
- **prey_linear.py**: Implements prey following a discrete linear trajectory, providing a simple target for predator agents to capture.

### Key Classes:
- **ContinuousPrey**: A class defining prey that moves continuously through the environment.
- **LinearPrey**: Models prey with discrete movement through the environment.

## 3. multimodal_mazes/predator_prey
This directory includes files for trial setups where predator agents interact with prey under different scenarios.

### Files:
- **RL_linear_prey_trial.py**: A class for running RL-based predator agents against prey moving linearly in the environment.
- **continuous_linear_prey_trial.py**: Defines a trial where prey moves continuously in a linear pattern and is chased by RL agents.
- **rule_based_linear_prey_trial.py**: Implements a trial where a rule-based agent attempts to capture prey with linear motion.

### Key Classes:
- **RLLinearPreyTrial**: A trial configuration where RL agents are tested against linearly moving prey.
- **ContinuousLinearPreyTrial**: Evaluates agents against continuously moving prey.
- **RuleBasedLinearPreyTrial**: Evaluates a rule-based predator agent in a scenario with linearly moving prey.

## 4. analysis
This directory contains a script to evaluate model-generated trajectories against actual animal data using Dynamic Time Warping (DTW).

### Purpose
DTW provides a quantitive method for comparing predicted and actual prey trajectories, highlighting alignment quality across different models and experimental conditions.

### Key Features:
- **Trajectory Preprocessing** - Trajectories from both experimental data and model outputs are resampled, normalised, and averaged across trials.
- **Comparison Visualisation**: Side-by-side plots for each start position and speed. Curves for model-to-data DTW distances.
- **Model Evaluation**: Area Under Curve (AUC) metrics summarise DTW performance across models.

## 5. results
This directory stores experimental results, organized by the type of agent and scenario being tested. The results are separated into different folders for easy access and analysis.

### Folders:
- **Multi-Prey Results**: Results from experiments with multiple prey in the environment.
- **RL Results**: Results from experiments using RL agents (DDPG or function approximation).
- **Rulebased Results**: Results from experiments using rule-based agents.

## 6. scripts
This directory contains Jupyter notebooks and script files for running specific experiments and generating results.

### Files:
- **Deep_RL_linear_screen_prey.ipynb**: A Jupyter notebook for testing DDPG agents against prey moving in a linear pattern across the screen.
- **RL_linear_screen_prey.ipynb**: A notebook that runs RL-based predator agents in trials with linear prey motion.
- **linear_screen_prey.ipynb**: A notebook for evaluating rulebased agent performance on prey with linear motion.

### README Files:
- Several README files providing descriptions of specific notebooks and experiment setups are included:
  - **README Deep RL Linear Screen Prey.md**
  - **README Linear Screen Prey.md**
  - **README RL Linear Screen Prey.md**

---

## Ongoing Work

This is an active and ongoing project.
