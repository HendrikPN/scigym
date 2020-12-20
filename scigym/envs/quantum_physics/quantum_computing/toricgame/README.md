# The Toric Game

This repository implements a reinforcement learning environment for the toric code,
with perspectives (see [arxiv](arxiv.org)), on which quantum error correction can 
be trained.

## Usage
The best way to use this environment, is to use [SciGym](https://github.com/hendrikpn/scigym) instead of this repo as a standalone.
Using [SciGym](https://github.com/hendrikpn/scigym), you can initialize this environment through

**Initializing the environment**
```python
import scigym
env = scigym.make("toricgame-v0")
```

## Environment description
TODO

## TRAINING
Launch the file train.py with corresponding arguments (see in the file)

## EVALUATION
Launch the file evaluate.py with corresponding arguments (see in the file)
