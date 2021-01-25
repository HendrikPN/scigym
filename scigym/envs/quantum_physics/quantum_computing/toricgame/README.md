# The Toric Game
This repository implements a reinforcement learning environment for learning to
perform quantum error decoding on the toric code with bitflip or depolarizing noise.
The corresponding publication can be found here: [https://arxiv.org/abs/2101.08093](https://arxiv.org/abs/2101.08093). If you use this environment, or any of the code provided here, we would appreciate it
if you could spare a citation :)

## Usage
This environment can be used via [SciGym](https://github.com/hendrikpn/scigym), or via a [standalone repository](https://github.com/condensedAI/neat-qec) that also includes our training scripts etc.
Using [SciGym](https://github.com/hendrikpn/scigym), you can initialize this environment through

**Initializing the environment**
```python
import scigym
env = scigym.make("toricgame-v0", board_size=3, error_model=0)

# Add errors (stores default error rate, so calling env.reset() reuses this error rate)
current_state = env.generate_errors(error_rate = 0.05)

# Using perspectives is not required at all, but if you'd like to, here's how:
locations, actions, action_probabilities=[], [], []
for p in env.state.syndrome_pos:
    # Get the indices for the qubit array, as if 'plaq' where at the center
    indices = env.get_perspective(p)
    # Update the input
    input = current_state[indices]
    # Ask the network for the action probabilities
    action_probabilities += list(nn.activate(input)) # nn is your network
```

## Environment description
Work In Progress
