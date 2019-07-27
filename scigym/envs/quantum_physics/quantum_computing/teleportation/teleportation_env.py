import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

H_0 = 0
T_0 = 1
H_1 = 2
T_1 = 3
H_2 = 4
T_2 = 5
CNOT_01 = 6
MEASURE_0 = 7
MEASURE_1 = 8
MEASURE_2 = 9
OUTCOME_PLUS_0 = 10
OUTCOME_MINUS_0 = 11
OUTCOME_PLUS_1 = 12
OUTCOME_MINUS_1 = 13
OUTCOME_PLUS_2 = 14
OUTCOME_MINUS_2 = 15
NOTHING = 16

# actions involving certain qubits
ACTIONS_Q0 = [0, 1, 6, 7]
ACTIONS_Q1 = [2, 3, 6, 8]
ACTIONS_Q2 = [4, 5, 9]


def _tensor(*args):
    res = np.array([[1]])
    for i in args:
        res = np.kron(res, i)
    return res


def _H(rho):
    return rho.conj().T


def _ptrace(rho, sys):
    # sys is a list of subsystems that should be traced over
    rho = np.copy(rho)  # just to be safe
    n = int(np.log2(rho.shape[0]))
    # old_shape = rho.shape
    rho = rho.reshape((2, 2) * n)
    sys = np.sort(sys)[::-1]  # sort highest to lowest, so we don't need to re-index after each trace
    for i in sys:
        rho = np.trace(rho, axis1=i, axis2=i + n)
        n -= 1
    return rho.reshape(2**n, 2**n)


Ha = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])  # Hadamard gate
T = np.array([[1, 0], [0, 1 / np.sqrt(2) * (1 + 1j)]], dtype=np.complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
Id = np.eye(2)

h0 = _tensor(Id, Ha, Id, Id)
t0 = _tensor(Id, T, Id, Id)
h1 = _tensor(Id, Id, Ha, Id)
t1 = _tensor(Id, Id, T, Id)
h2 = _tensor(Id, Id, Id, Ha)
t2 = _tensor(Id, Id, Id, T)
cnot01 = _tensor(Id, CNOT, Id)

z0 = np.array([1, 0]).reshape(2, 1)
z1 = np.array([0, 1]).reshape(2, 1)
Pz0 = np.dot(z0, _H(z0))
Pz1 = np.dot(z1, _H(z1))

proj_plus_0 = _tensor(Id, Pz0, Id, Id)
proj_minus_0 = _tensor(Id, Pz1, Id, Id)
proj_plus_1 = _tensor(Id, Id, Pz0, Id)
proj_minus_1 = _tensor(Id, Id, Pz1, Id)
proj_plus_2 = _tensor(Id, Id, Id, Pz0)
proj_minus_2 = _tensor(Id, Id, Id, Pz1)

phiplus = 1 / np.sqrt(2) * (_tensor(z0, z0) + _tensor(z1, z1))

# def _random_pure_state():
#     # pick a random point on the bloch sphere
#     phi = 2 * np.pi * np.random.random()
#     theta = np.arccos(2 * np.random.random() - 1)
#     return np.cos(theta / 2) * z0 + np.exp(1j * phi) * np.sin(theta / 2) * z1


def _norm(psi):
    return np.sqrt(np.sum(np.abs(psi)**2))


def _normalize(psi):
    return psi / _norm(psi)


def _measure(psi, i):
    if i == 0:
        aux1 = np.dot(proj_plus_0, psi)
        aux2 = np.dot(proj_minus_0, psi)
    elif i == 1:
        aux1 = np.dot(proj_plus_1, psi)
        aux2 = np.dot(proj_minus_1, psi)
    elif i == 2:
        aux1 = np.dot(proj_plus_2, psi)
        aux2 = np.dot(proj_minus_2, psi)
    p_plus = _norm(aux1)**2
    p_minus = _norm(aux2)**2
    if np.random.random() < p_plus:
        state = _normalize(aux1)
        outcome = 0
    else:
        state = _normalize(aux2)
        outcome = 1
    return state, outcome


class TeleportationEnv(gym.Env):
    """An environment for discovering the quantum teleportation protocol.

    Starting from an input qubit and a Bell pair, the agent should learn to
    use a universal gate set and measurements to teleport the state of the
    input qubit.

    Args:
        max_actions (int): The environment resets after this many effective actions. Defaults to 40.

    Attributes:
        n_actions (int): Number of actions. Is 10.
        action_space (gym.spaces.Space): gym.spaces.Discrete(`n_actions`)
        observation_space (gym.spaces.Space): gym.spaces.MutliDiscrete([17] * `max_actions`)
        state (np.ndarray): Current internal quantum state.
        percept_now (list of ints): Current percept, is an element of the `observation_space`.
        available_actions (list of ints): These actions are valid for the current state.
        metadata (dict)

    """
    # no rendering available
    metadata = {'render.modes': []}

    def __init__(self, max_actions=40):
        self.n_actions = 10
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self._max_actions = max_actions
        self.observation_space = gym.spaces.MultiDiscrete([17] * self._max_actions)
        self._target = phiplus
        self._target_rho = np.dot(self._target, _H(self._target))
        self.state = _tensor(self._target, phiplus)
        self.percept_now = [NOTHING] * self._max_actions
        self.available_actions = [i for i in range(self.n_actions)]
        self._actions_taken = 0

    def reset(self):
        self._target = phiplus
        self._target_rho = np.dot(self._target, _H(self._target))
        self.state = _tensor(self._target, phiplus)
        self.percept_now = [NOTHING] * self._max_actions
        self.available_actions = [i for i in range(self.n_actions)]
        return self.percept_now, {"available_actions": self.available_actions}

    def _check_success(self):
        aux = np.dot(self.state, _H(self.state))
        aux = _ptrace(aux, [1, 2])  # note that 1, 2 in this notation corresponds to qubits 0 and 1
        return np.allclose(aux, self._target_rho)

    def _remove_actions(self, actions):
        for action in actions:
            try:
                self.available_actions.remove(action)
            except ValueError:
                continue

    def _record_action(self, action):
        self.percept_now[self._actions_taken] = action
        self._actions_taken += 1

    def step(self, action):
        """Apply the chosen action.

        Args:
            action (int): An action in the action space that is performed.

        Returns:
            tuple: of the form (observation, reward, episode_finished, info)

        """
        if action in self.available_actions and self._actions_taken < self._max_actions:
            if action in range(7):
                self._record_action(action)

            if action == H_0:
                self.state = np.dot(h0, self.state)
            elif action == T_0:
                self.state = np.dot(t0, self.state)
            elif action == H_1:
                self.state = np.dot(h1, self.state)
            elif action == T_1:
                self.state = np.dot(t1, self.state)
            elif action == H_2:
                self.state = np.dot(h2, self.state)
            elif action == T_2:
                self.state = np.dot(t2, self.state)
            elif action == CNOT_01:
                self.state = np.dot(cnot01, self.state)
            elif action == MEASURE_0:
                self.state, outcome = _measure(self.state, 0)
                if outcome == 0:
                    self._record_action(OUTCOME_PLUS_0)
                else:
                    self._record_action(OUTCOME_MINUS_0)
                self._remove_actions(ACTIONS_Q0)
            elif action == MEASURE_1:
                self.state, outcome = _measure(self.state, 1)
                if outcome == 0:
                    self._record_action(OUTCOME_PLUS_1)
                else:
                    self._record_action(OUTCOME_MINUS_1)
                self._remove_actions(ACTIONS_Q1)
            elif action == MEASURE_2:
                self.state, outcome = _measure(self.state, 2)
                if outcome == 0:
                    self._record_action(OUTCOME_PLUS_2)
                else:
                    self._record_action(OUTCOME_MINUS_2)
                self._remove_actions(ACTIONS_Q2)

        if self._check_success():
            reward = 1
            episode_finished = 1
        else:
            reward = 0
            episode_finished = 0

        # if no actions remain, episode is over
        if not self.available_actions or self._actions_taken >= self._max_actions:
            episode_finished = 1

        return self.percept_now, reward, episode_finished, {"available_actions": self.available_actions}

    def render(self):
        raise NotImplementedError
