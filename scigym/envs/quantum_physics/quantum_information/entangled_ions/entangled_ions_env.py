import numpy as np
from functools import reduce
import gym

from scigym.envs.quantum_physics.quantum_information.entangled_ions.operations.qudit_qm import QuditQM
from scigym.envs.quantum_physics.quantum_information.entangled_ions.operations.laser_gates import LaserGates

class EntangledIonsEnv(gym.Env, QuditQM):
    def __init__(self, **kwargs):
        """
        This is the environment for a d-level ion trap quantum computer as 
        described in https://arxiv.org/pdf/1907.08569.pdf.
        An agent has a certain gate set at its disposal made up from elementary
        laser pulses and global molmer sorensen gates. 
        At each time step an agent can choose one gate to apply to the current
        quantum state of the environment. 
        The environment state is a d^n dimensional quantum state where d is the 
        local dimension of each ion and n is the number of ions. 
        The observation is a 2d^n dimensional vector representing real and 
        imaginary vector components.
        The initial state is |0...0>.
        The goal is to create multipartite high-dimensional entanglement between 
        ions characterizsed by a Schmidt rank vector. Once such a state has 
        been created, the agent receives a reward. 
        If a maximum number of time steps is reached, the episode ends and 
        should be restarted.

        Args:
            **kwargs:
                num_ions (int): The number of ions. Defaults to 3.
                dim (int): The local (odd) dimension of an ion. Defaults to 3.
                goal (list): List of SRVs that are rewarded. 
                             Defaults to [[3,3,3]].
                phases (dict): The phases defining the laser gate set.
                               Defaults to 
                               {'pulse_angles': [np.pi/2],
                                'pulse_phases': [0, np.pi/2, np.pi/6],
                                'ms_phases': [-np.pi/2]}
                max_steps (int): The maximum number of allowed time steps.
                                 Defaults to 10.
        """
        # this is just checking whether kwargs are provided and adds defaults
        if 'num_ions' in kwargs and type(kwargs['num_ions']) is int:
            setattr(self, 'num_ions', kwargs['num_ions'])
        else:
            setattr(self, 'num_ions', 3)
        if 'dim' in kwargs and type(kwargs['dim']) is int:
            setattr(self, 'dim', kwargs['dim'])
        else:
            setattr(self, 'dim', 3)
        if 'goal' in kwargs and type(kwargs['goal']) is list:
            setattr(self, 'goal', kwargs['goal'])
        else:
            setattr(self, 'goal', [[3,3,3]])
        if 'phases' in kwargs and type(kwargs['phases']) is dict:
            setattr(self, 'phases', kwargs['phases'])
        else:
            setattr(self, 'phases', {
                'pulse_angles': [np.pi/2],
                'pulse_phases': [0, np.pi/2, np.pi/6],
                'ms_phases': [-np.pi/2]
            })
        if 'max_steps' in kwargs and type(kwargs['max_steps']) is int:
            setattr(self, 'max_steps', kwargs['max_steps'])
        else:
            setattr(self, 'max_steps', 10)

        # add methods (such as `ket` or `srv`) from parent
        super().__init__(self.dim, self.num_ions)
        #spaces.Box: The observation space as defined by `gym` 
        o_size = self.dim**self.num_ions * 2
        self.observation_space = gym.spaces.Box(low=-1., 
                                                high=1., 
                                                shape=(o_size,1),
                                                dtype=np.float64
                                                )
        #:class:`LaserGates`: The laser gates used for actions.
        self.gates = LaserGates(self.dim, self.num_ions, self.phases)
        #list: The action gate set.
        self.actions = self.gates.gates
        #int: Number of gates.
        self.num_actions = len(self.gates.gates)
        #spaces.Discrete: The action space as defined by `gym`
        self.action_space = gym.spaces.Discrete(self.num_actions)
        #np.ndarray: Initial state.
        self.init_state = reduce(np.kron, 
                                 [self.ket(0) for i in range(self.num_ions)]
                                )
        #np.ndarray: Current state of the environment and observation.
        self.state = self.init_state
        #int: Current number of time steps.
        self.time_step = 0
    
    def reset(self):
        """
        Resets the environment to its initial state, i.e., |0...0>.
        
        Returns:
            observation (np.ndarray): Initial environment state.
        """
        self.state = self.init_state
        self.time_step = 0

        observation = np.append(self.state.real, self.state.imag, axis=0)

        return observation

    def step(self, action):
        """
        Performs and evaluates an action on the environment.
        
        (1) A quantum gate from our laser gate set is applied to the
            current state of the environment.
        (2) The Schmidt rank vector (SRV) is calculate for the current state
            of the environment. I
        (3) If the SRV is among the set of goal SRVs, a reward is provided and
            the trial is done. If not, no reward is given. The trial is ended if 
            the time exceed the maximum number of steps.

        Args:
            action (int): Index of action in gate set.

        Returns:
            observation (np.ndarray): Representation of environment state.
            reward (float): Current reward.
            done (bool): Whether or not the episode is done.
            info (dict): Additional information. Not provided.
        """
        done = False
        reward = 0.
        # increment counter
        self.time_step += 1
        # (1) Apply gate to state
        self.state = self.gates.gates[action]@self.state
        # (2) Calculate SRV
        srv = self.srv(self.state)
        # (3) Finish episode and give reward if appropriate
        if srv in self.goal:
            done = True
            reward = 1.
        elif self.time_step >= self.max_steps:
            done = True

        observation = np.append(self.state.real, self.state.imag, axis=0)

        return observation, reward, done, {}
