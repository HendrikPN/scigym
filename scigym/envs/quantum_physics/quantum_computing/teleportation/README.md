### Teleportation

The task in this environment is for the agent to find a protocol that uses an already distributed entangled state to teleport an arbitrary quantum state, i.e. the agent should learn to reconstruct the well-known quantum teleportation protocol when provided with a universal gate set and quantum measurements.

**Initializing the environment**
```python
import scigym
env = scigym.make("teleportation-v0")
```

#### Environment description

The agent is presented with the following situation:

*Image goes here*

where the input qubit is an arbitrary state Ψ.

*Available actions:*

* T gate T = diag(1, exp(iπ/4)) on each qubit (3 actions)
* Hadamard gate H on each qubit (3 actions)
* CNOT-gate on the two qubits at location A (1 actions)
* Z-measurements on each qubit (3 actions)
In total: **10 actions**

The Z-measurements are considered to be destructive, i.e. the qubit is removed and actions acting on that qubit will are no longer valid (and will not change the state).
If the agent can handle changing action sets, the currently available actions can be accessed via the attribute `env.available_actions` or the info dictionary returned by the `env.step(action)` method.

*Observations*
The observations perceived by the agent are a list of previous actions and measurement outcomes. Invalid actions (on qubits that have already been measured) are not added to this list.


#### Known solutions

#### Implementation details

#### Discussion
