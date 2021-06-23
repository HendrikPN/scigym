# The Toric Game
This repository implements a reinforcement learning environment for learning to
perform quantum error decoding on the toric code with bitflip or depolarizing 
noise.
The corresponding publication can be found here: [https://arxiv.org/abs/2101.08093](https://arxiv.org/abs/2101.08093). 
If you use this environment, or any of the code provided here, we would 
appreciate it if you could spare a citation :)

## Usage
This environment can be used via [SciGym](https://github.com/hendrikpn/scigym), 
or via a [standalone repository](https://github.com/condensedAI/neat-qec) that 
also includes our training scripts etc.
Using [SciGym](https://github.com/hendrikpn/scigym), you can initialize this 
environment through

**Initializing the environment**
```python
import scigym
KWARGS = {'board_size': 3, 'error_model': 0, 'error_rate': 0.001}
env = scigym.make("toricgame-v0", **KWARGS)
```

## Environment description
The player is tasked to correct errors in a quantum error correction
code based on the toric code by applying Pauli operators while only
observing error syndromes.

The player wins only if all syndromes have been removed and no logical
error was introduced.
The player looses if...
+ ... they remove all syndromes but introduce a logical error.
+ ... exceed the maximum number of steps.
+ ... they perform the same action twice (if `allow_illegal_actions` is `False`).

## Example for Human Play
The following code can be used to play the game if the environment has already
been initialized.

```python
# Initialize a game that is not already solved
observation = env.reset()
    
# Render the initial state
env.render()

done = False
while not done:
    action = int(input(f"Pick action from {env.action_space.n} actions: "))
    observation, reward, done, info = env.step(action)
    print(f"You received a reward of {reward}.")
    env.render()

print("Combined with the original physical qubit errors, this is the total error string:")
for q in env.initial_qubits_flips[0]:
    env.state.act(q, 0, update_syndrome=False)
env.render()
    
if reward == 1:
    print("You win!")
else: 
    print("You've lost")
```
