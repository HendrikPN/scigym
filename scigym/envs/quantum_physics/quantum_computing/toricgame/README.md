# The Toric Game

This repository implements a reinforcement learning environment for learning to
perform quantum error decoding on the toric code with depolarizing noise.
Game states are represented using perspectives from [here](arxiv.org)
be trained.

## Usage
The best way to use this environment, is to use [SciGym](https://github.com/hendrikpn/scigym) instead of this repo as a standalone.
Using [SciGym](https://github.com/hendrikpn/scigym), you can initialize this environment through

**Initializing the environment**
```python
import scigym
env = scigym.make("toricgame-v0", board_size=3, error_model=0, error_rate = 0.05)
```

## Environment description
TODO
