# Reinforcement Learning Agent Training with Gymnasium

This repository contains implementations of reinforcement learning agents trained in Gymnasium environments using both value-based and policy-based methods. The goal is to explore and compare different RL approaches to understand their strengths, weaknesses, and applicable scenarios.

## Table of Contents

- [What is Reinforcement Learning?](#what-is-reinforcement-learning)
- [Gymnasium Environment](#gymnasium-environment)
- [Value-Based Methods](#value-based-methods)
- [Policy-Based Methods](#policy-based-methods)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Contributing](#contributing)

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning that focuses on training agents to make decisions by interacting with an environment[1][2]. Unlike supervised learning, which relies on labeled input-output pairs, RL agents learn through trial and error by receiving rewards or penalties based on their actions[4][13].

### Key Components

**Agent**: The decision-maker that interacts with the environment and learns to optimize its behavior[1][5].

**Environment**: The external world or system with which the agent interacts, including all conditions, contexts, and dynamics that the agent must respond to[12][18].

**State**: The current situation or condition of the environment at a specific time, representing all information needed for decision-making[2][15].

**Action**: The choices or decisions that the agent can make in response to the current state[5][15].

**Reward**: The feedback signal from the environment that indicates how good or bad an action was, guiding the agent toward its goal[4][15].

**Policy**: A strategy or rule set that maps states to actions, determining what action the agent should take in each state[1][6].

### How RL Works

The RL process follows a continuous loop[13][18]:

1. The agent observes the current state of the environment
2. Based on its policy, the agent selects an action
3. The environment responds to the action by transitioning to a new state
4. The agent receives a reward signal from the environment
5. The agent updates its policy based on the experience
6. The process repeats until the episode ends or a goal is achieved

This trial-and-error learning approach allows agents to discover optimal strategies without explicit programming, making RL particularly powerful for complex decision-making tasks[7][10].

## Gymnasium Environment

Gymnasium is an open-source Python library that provides a standard API for reinforcement learning environments[21][27]. It is a maintained fork of OpenAI's Gym library, developed by the Farama Foundation to ensure continued development and maintenance[22][25].

### What is Gymnasium?

Gymnasium offers a unified interface for RL experiments, making it easier for researchers and developers to:

- **Standardize Environment Interactions**: Provides consistent methods for initializing environments, taking actions, and receiving observations[29][32]
- **Focus on Algorithm Development**: Abstracts away the complexity of environment implementation, allowing researchers to concentrate on developing RL algorithms[26][38]
- **Ensure Reproducibility**: Includes environment versioning and seeding capabilities to ensure consistent and reproducible results[32]

### Key Features

**Standard API**: All environments follow the same interface pattern:
```python
import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

**Diverse Environment Collection**: Gymnasium includes several families of environments[27][33]:
- **Classic Control**: Physics-based control tasks (CartPole, Pendulum, etc.)
- **Box2D**: 2D physics-based games and simulations
- **Toy Text**: Simple discrete environments for debugging algorithms
- **MuJoCo**: Complex physics simulations with multi-joint control
- **Atari**: Classic Atari 2600 games for testing RL algorithms

**Action and Observation Spaces**: Environments define structured spaces that specify the format and bounds of actions and observations[24].

### Why Use Gymnasium?

1. **Industry Standard**: Gymnasium's API has become the de facto standard for RL research and development[31][34]
2. **Easy Integration**: Compatible with popular RL libraries like Stable-Baselines3, RLlib, and others[35]
3. **Extensive Documentation**: Comprehensive guides and examples for getting started
4. **Active Community**: Regular updates, bug fixes, and new environment additions

## Value-Based Methods

Value-based methods in reinforcement learning focus on learning the value of states or actions to guide decision-making[41][44]. Instead of directly learning a policy, these methods estimate how beneficial it is to be in specific states or take specific actions, measured by expected cumulative future rewards[50][53].

### Core Concept

The fundamental idea is to build value functions that estimate expected returns:

- **State-Value Function V(s)**: Expected reward for being in state s and following the current policy
- **Action-Value Function Q(s,a)**: Expected reward for taking action a in state s and then following the policy[41][50]

Once learned, these value functions guide action selection by choosing actions with the highest estimated values[44][47].

### Key Algorithms

#### Q-Learning
Q-Learning is a model-free, off-policy algorithm that learns optimal action-values using the Bellman equation[44]:

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- α is the learning rate
- γ is the discount factor
- r is the immediate reward
- s' is the next state

#### Deep Q-Networks (DQN)
For complex environments with large state spaces, DQN uses neural networks to approximate Q-values instead of maintaining Q-tables[45][51]:

- **Experience Replay**: Stores past experiences to break temporal correlations and improve stability[54]
- **Target Network**: Uses a separate network for computing target values to stabilize training[54]
- **ε-greedy Exploration**: Balances exploration and exploitation during learning[57]

#### Temporal Difference Learning
TD learning updates value estimates based on the difference between predicted and observed values[43][46]:

- Updates predictions to match future predictions at each time step
- Combines aspects of Monte Carlo and Dynamic Programming methods
- Enables learning without waiting for episode completion[52][55]

### Advantages

1. **Discrete Action Efficiency**: Excellent performance in environments with discrete, manageable action spaces[41][53]
2. **Sample Efficiency**: Generally requires fewer interactions with the environment compared to policy methods[81]
3. **Proven Convergence**: Theoretical guarantees for convergence in tabular settings[44]
4. **Computational Efficiency**: Direct value estimation can be computationally efficient for appropriate problems[81]

### Disadvantages

1. **Continuous Action Limitations**: Struggle with continuous or high-dimensional action spaces where maximizing over all actions becomes impractical[41][67]
2. **Function Approximation Issues**: Can suffer from instability and convergence problems when using neural networks[82][85]
3. **Exploration Challenges**: Require explicit exploration strategies (like ε-greedy) that may not be optimal[87]
4. **Deterministic Policies**: Typically learn quasi-deterministic policies, which can be suboptimal in partially observable environments[87][93]
5. **Overestimation Bias**: DQN can overestimate action values, leading to suboptimal policies[42][60]

## Policy-Based Methods

Policy-based methods directly learn a policy—a strategy for choosing actions—without relying on intermediate value function estimates[61][67]. These methods parameterize the policy (often using neural networks) and optimize it to maximize expected cumulative rewards[64][73].

### Core Concept

Rather than learning "how good" states or actions are, policy methods directly learn "what to do" by optimizing a parameterized policy π_θ(a|s) that outputs action probabilities[64][76]. The goal is to find parameters θ that maximize the expected return:

```
J(θ) = E[Σ γ^t R_t]
```

### Key Algorithms

#### REINFORCE Algorithm
REINFORCE is a Monte Carlo policy gradient method that updates policy parameters using complete episode returns[68][77]:

1. **Collect Episode**: Run policy to generate trajectory of states, actions, and rewards
2. **Calculate Returns**: Compute cumulative discounted rewards G_t for each time step
3. **Update Policy**: Adjust parameters using: θ ← θ + α ∇log π_θ(a_t|s_t) G_t
4. **Repeat**: Continue for multiple episodes until convergence[71][80]

#### Actor-Critic Methods
Actor-Critic algorithms combine policy-based and value-based approaches[63][66]:

- **Actor**: The policy network that selects actions based on current state
- **Critic**: Value function that evaluates the actor's actions
- **Advantage**: Uses the difference between actual and expected returns to reduce variance[65][69]

Popular variants include:
- **A2C/A3C**: Advantage Actor-Critic with synchronous/asynchronous updates
- **PPO**: Proximal Policy Optimization with clipped objectives for stability
- **DDPG**: Deep Deterministic Policy Gradient for continuous control[66][69]

### Advantages

1. **Continuous Action Spaces**: Naturally handle continuous and high-dimensional action spaces by directly outputting actions[67][73]
2. **Stochastic Policies**: Can learn probabilistic policies, enabling better exploration and handling of partially observable environments[64][87]
3. **No Perceptual Aliasing**: Stochastic policies avoid getting stuck in situations where different states appear identical[87][98]
4. **Better Convergence Properties**: Often have superior convergence characteristics compared to value-based methods[87]
5. **Direct Optimization**: Optimize the quantity we care about (policy performance) directly[70][76]

### Disadvantages

1. **High Variance**: Policy gradient estimates can have high variance, leading to unstable training[67][83]
2. **Sample Inefficiency**: Often require many episodes to learn effective policies[83][97]
3. **Local Optima**: Can get trapped in local optima during gradient ascent[70][90]
4. **Slow Convergence**: May converge slowly compared to value-based methods in some environments[83][97]
5. **Hyperparameter Sensitivity**: Performance can be highly sensitive to learning rates and other hyperparameters[92][95]
6. **Exploration-Exploitation Balance**: Balancing exploration of new strategies with exploitation of known successful approaches[83][97]

## Method Comparison Summary

| Aspect | Value-Based Methods | Policy-Based Methods |
|--------|-------------------|-------------------|
| **Learning Target** | Value functions (Q-values, V-values) | Policy parameters directly |
| **Action Spaces** | Best for discrete actions | Excellent for continuous actions |
| **Policy Type** | Typically deterministic (ε-greedy) | Can learn stochastic policies |
| **Sample Efficiency** | Generally more sample efficient | Often requires more samples |
| **Convergence** | Faster initial learning | May have better long-term convergence |
| **Exploration** | Requires explicit exploration strategy | Natural exploration through stochasticity |
| **Computational Cost** | Lower for discrete actions | Higher due to gradient computation |
| **Stability** | Can suffer from function approximation issues | High variance in updates |

## Project Structure

```
reinforcement-learning-agents/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── src/
│   ├── environments/         # Gymnasium environment configurations
│   ├── value_based/         # Value-based method implementations
│   │   ├── q_learning.py    # Tabular Q-learning
│   │   ├── dqn.py          # Deep Q-Network
│   │   └── double_dqn.py   # Double DQN
│   ├── policy_based/        # Policy-based method implementations
│   │   ├── reinforce.py    # REINFORCE algorithm
│   │   ├── actor_critic.py # Actor-Critic methods
│   │   └── ppo.py          # Proximal Policy Optimization
│   └── utils/              # Utility functions and helpers
├── experiments/            # Experiment scripts and configurations
├── results/               # Training results and visualizations
└── docs/                 # Additional documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/reinforcement-learning-agents.git
cd reinforcement-learning-agents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run a basic example:
```bash
python src/value_based/dqn.py --env CartPole-v1
```

### Basic Usage

Train a DQN agent:
```python
from src.value_based.dqn import DQNAgent
import gymnasium as gym

env = gym.make("CartPole-v1")
agent = DQNAgent(state_size=4, action_size=2)
agent.train(env, episodes=1000)
```

Train a REINFORCE agent:
```python
from src.policy_based.reinforce import REINFORCEAgent
import gymnasium as gym

env = gym.make("CartPole-v1")
agent = REINFORCEAgent(state_size=4, action_size=2)
agent.train(env, episodes=1000)
```

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests. Make sure to:

1. Follow the existing code style and structure
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass before submitting

## References

This README is based on comprehensive research from multiple sources in the reinforcement learning literature. The implementation focuses on practical applications while maintaining theoretical rigor in the algorithm designs.

---

*Happy Learning and Training!* 🤖🎯
