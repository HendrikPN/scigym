# Just in case a python version <3 is used
from __future__ import division

import sys, os
import numpy as np
import gym
from gym.utils import seeding

class ToricGameEnv(gym.Env):
    '''
    ToricGameEnv environment. Single player game.
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, code_distance, error_model, channels, use_memory):
        """
        Args:
            opponent: Fixed
            code_distance: the code distance
            error_model: 0 - bitflip, 1 - depolarizing
            channels: list for indicating which operators
                      to use in the state representation.
                      [0] - plaquettes only
                      [1] - stars only
                      [0,1] - both :)
            use_memory: include physical qubits in state representation if true

        """
        self.board_size = code_distance
        self.error_model = error_model
        self.channels = channels
        self.memory = use_memory

        # Keep track of the moves
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]

        # Empty State
        self.state = Board(self.board_size)
        self.done = None

        # No error rate
        self.current_error_rate = 0

        # Build the perspective slices
        self._build_perspectives()

        # Action space depends on perspectives and error model
        num_pauli_operators = 1 if error_model == 0 else 3
        num_qubits_to_act_on = 2*code_distance**2
        self.action_space = gym.spaces.Discrete(num_qubits_to_act_on*num_pauli_operators)

        # Observation space depends the errors only.
        shape = (code_distance, code_distance) if error_model == 0 else (2, code_distance, code_distance)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=shape, dtype=np.uint8)

    def reset(self):
        self.generate_errors(self.current_error_rate)

    def generate_errors(self, error_rate):
        self.current_error_rate = error_rate

        # Reset the board state
        self.state.reset()

        # Let the "opponent" do it's initial evil moves :)
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]
        self._set_initial_errors(self.current_error_rate)

        self.done = self.state.is_terminal()
        self.reward = 0
        if self.done:
            self.reward = 1
            if self.state.has_logical_error(self.initial_qubits_flips):
                self.reward = -1

        return self.state.encode(self.channels, self.memory)

    def step(self, action, without_illegal_actions=False):
        '''
        Args:
            action: a number representing which qubit to act on, with which
                    operator. action = 3*qubit_number + {0,1,2}

        Return:
            observation: board encoding,
            reward: reward of the game,
            done: boolean,
            info: state dict
        '''
        # If already terminal, then don't do anything, count as win
        if self.done:
            self.reward = 1
            return self.state.encode(self.channels, self.memory), 1., True, {'state': self.state, 'message':"success"}

        # Check if we flipped twice the same qubit
        pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
        pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)

        # Game is lost if we repeat an action if we are playing and not training
        if not without_illegal_actions:
            if pauli_X_flip and location in self.qubits_flips[0]:
                return self.state.encode(self.channels, self.memory), -1.0, True, {'state': self.state, 'message': "illegal_action"}
            if pauli_Z_flip and location in self.qubits_flips[1]:
                return self.state.encode(self.channels, self.memory), -1.0, True, {'state': self.state, 'message': "illegal_action"}

        if pauli_X_flip:
            self.qubits_flips[0].append(location)
        if pauli_Z_flip:
            self.qubits_flips[1].append(location)

        # Act on the state
        self.state.act(location, pauli_opt)

        # Reward: if nonterminal, then the reward is 0
        if not self.state.is_terminal():
            self.done = False
            self.reward = 0
            return self.state.encode(self.channels, self.memory), 0., False, {'state': self.state, 'message':"continue"}

        # We're in a terminal state. Reward is 1 if won, -1 if lost
        self.done = True
        if self.state.has_logical_error(self.initial_qubits_flips):
            return self.state.encode(self.channels, self.memory), -1.0, True, {'state': self.state, 'message':"logical_error"}
        else:
            return self.state.encode(self.channels, self.memory), 1.0, True, {'state': self.state, 'message':"success"}

    def _set_initial_errors(self, error_rate):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''
        for q in self.state.qubit_pos:
            if np.random.rand() < error_rate:
                if self.error_model == 0:
                    pauli_opt = 0
                elif self.error_model == 1:
                    pauli_opt = np.random.randint(0,3)

                pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
                pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)

                if pauli_X_flip:
                    self.initial_qubits_flips[0].append( q )
                if pauli_Z_flip:
                    self.initial_qubits_flips[1].append( q )

                self.state.act(q, pauli_opt)

        # Now unflip the qubits, they're a secret
        self.state.qubit_values = np.zeros((2, 2*self.board_size*self.board_size))

    def _build_perspectives(self):

        size = self.board_size
        op_pos={}
        qubit_pos, op_pos[0], op_pos[1] = Board.component_positions(size)

        input_dim=len(channels)*size**2
        if self.use_memory:
            input_dim *= 3

        indices = np.arange(input_dim)

        # Loading the slices of the input
        slices={}
        index0, dim=0,0
        for channel in channels:
            dim=size**2
            slices['op_'+str(channel)]=indices[index0:index0+dim].reshape(size, size)
            index0+=dim

            if self.use_memory:
                dim=2*size**2
                slices['qubit_'+str(channel)]=indices[index0:index0+dim]
                index0+=dim

                # The qubit matrix is not squared
                dummy_squared=-np.ones((2*size, 2*size), dtype=np.int)
                for x in range(2*size):
                    for y in range(2*size):
                        if [x,y] in qubit_pos:
                            qubit_index = qubit_pos.index([x,y])
                            dummy_squared[x,y]=slices['qubit_'+str(channel)][qubit_index]

                slices['qubit_'+str(channel)]=dummy_squared

        # For the qubits
        # To make sure the lattice is in the right convention
        # Having first row and first column having star operators
        # We sometimes need to shift it by one row or column
        rolling_axis_after_rotation=[[], [0], [0,1], [1]]

        # Define the size**2 ways of shifting the board
        self.perspectives = {i : {} for i in range(4)}
        for channel in [0,1]:
            for rot_i in range(4):
                for i in range(size):
                    for j in range(size):
                        # Shift the syndrome to central plaquette
                        index = j+size*i
                        plaq = op_pos[channel][index]

                        transformed_slices={}
                        for key in slices:
                            # Translate
                            factor = 2 if "qubit" in key else 1
                            center = int((size-1)/2) # Assume that size is odd
                            slice = np.roll(slices[key], factor*(center - i), axis=0)
                            slice = np.roll(slice, factor*(center - j), axis=1)

                            # Rotate
                            slice = np.rot90(slice, rot_i)

                            # Filter out dummy variable in qubits matrix
                            if "qubit" in key:
                                # Rotating the board causes the qubits to move out
                                # of their conventional location
                                slice = np.roll(slice, 1, rolling_axis_after_rotation[rot_i]).flatten()

                                filtered = np.argwhere(slice!=-1)
                                slice = slice[filtered]

                            transformed_slices[key] = slice.flatten()

                        # Concatenate the indices
                        self.perspectives[rot_i][tuple(plaq)] =np.array([], dtype=np.int)
                        # To use symmetric roles of plaquette and stars, we normalize the order of the input
                        index0, index1 = channel, (channel+1)%2
                        ordered_keys = ["op_"+str(index0), "qubit_"+str(index0), "op_"+str(index1), "qubit_"+str(index1)]
                        for key in ordered_keys:
                            if key in slices.keys():
                                self.perspectives[rot_i][tuple(plaq)] = np.concatenate((self.perspectives[rot_i][tuple(plaq)], transformed_slices[key].flatten()))

    def get_perspective(self, plaq, rotation_number=0):
        '''
        Return the indices of the new lattice shifted such that the syndrome is
        placed at the central plaquette.

        Also allows for returning the reflected indices given by rotation_number
        '''
        return self.perspectives[rotation_number][(plaq[0], plaq[1])]

    def close(self):
        self.state = None

    def render(self, mode="human", close=False):
        fig, ax = plt.subplots()
        a=1/(2*self.board_size)

        for i, p in enumerate(self.state.plaquet_pos):
            if self.state.op_values[0][i]==1:
                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax.add_patch(plaq)

        for i, p in enumerate(self.state.star_pos):
            if self.state.op_values[1][i]==1:
                fc = 'green'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax.add_patch(plaq)

        # Draw lattice
        for x in range(self.board_size):
            for y in range(self.board_size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax.add_patch(lattice)

        for i, p in enumerate(self.state.qubit_pos):
            pos=(a*p[0], a*p[1])
            fc='darkgrey'
            if self.state.qubit_values[0][i] == 1 and self.state.qubit_values[1][i] == 0:
                fc='darkblue'
            elif self.state.qubit_values[0][i] == 0 and self.state.qubit_values[1][i] == 1:
                fc='darkred'
            elif self.state.qubit_values[0][i] == 1 and self.state.qubit_values[1][i] == 1:
                fc='darkmagenta'
            circle = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc)
            ax.add_patch(circle)
            plt.annotate(str(i), pos, fontsize=8, ha="center")

        ax.set_xlim([-.1,1.1])
        ax.set_ylim([-.1,1.1])
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')
        plt.show()

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]

class Board(object):
    '''
    Basic Implementation of a ToricGame Board, actions are int [0,2*board_size**2)
    o : qubit
    P : plaquette operator
    x : star operator

    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |
    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |
    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |

    '''

    @staticmethod
    def component_positions(size):
        qubit_pos   = [[x,y] for x in range(2*size) for y in range((x+1)%2, 2*size, 2)]
        plaquet_pos = [[x,y] for x in range(1,2*size,2) for y in range(1,2*size,2)]
        star_pos    = [[x,y] for x in range(0,2*size,2) for y in range(0,2*size,2)]
        return qubit_pos, plaquet_pos, star_pos

    def __init__(self, board_size):
        self.size = board_size

        # Real-space locations
        self.qubit_pos, self.plaquet_pos, self.star_pos  = self.component_positions(self.size)

        # Mapping between 1-index and 2D position
        '''
        self.qubit_dict, self.star_dict, self.plaquet_dict = {},{},{}
        for i in range(2*self.size*self.size):
            self.qubit_dict[self.qubit_pos[i]] = i
        for i in range(self.size*self.size):
            self.star_dict[i] = self.star_pos[i]
            self.plaquet_dict[i] = self.plaquet_pos[i]
        '''

        # Define here the logical error for efficiency
        self.z1pos = [[0,x] for x in range(1, 2*self.size, 2)]
        self.z2pos = [[y,0] for y in range(1, 2*self.size, 2)]
        self.x1pos = [[1,x] for x in range(0, 2*self.size, 2)]
        self.x2pos = [[y,1] for y in range(0, 2*self.size, 2)]

        self.reset()

    def reset(self):
        #self.board_state = np.zeros( (2, 2*self.size, 2*self.size) )

        self.qubit_values = np.zeros((2, 2*self.size*self.size))
        self.op_values = np.zeros((2, self.size*self.size))

        self.syndrome_pos = [] # Location of syndromes

    def act(self, coord, operator):
        '''
            Args: input action in the form of position [x,y]
            coord: real-space location of the qubit to flip
        '''

        pauli_X_flip = (operator==0 or operator==2)
        pauli_Z_flip = (operator==1 or operator==2)

        qubit_index = self.qubit_pos.index(coord)

        # Flip it!
        if pauli_X_flip:
            self.qubit_values[0][qubit_index] = (self.qubit_values[0][qubit_index] + 1) % 2
        if pauli_Z_flip:
            self.qubit_values[1][qubit_index] = (self.qubit_values[1][qubit_index] + 1) % 2

        # Update the syndrome measurements
        # Only need to incrementally change
        # Find plaquettes that the flipped qubit is a part of
        plaqs=[]
        if pauli_X_flip:
            if coord[0] % 2 == 0:
                plaqs += [ [ (coord[0] + 1) % (2*self.size), coord[1] ], [ (coord[0] - 1) % (2*self.size), coord[1] ] ]
            else:
                plaqs += [ [ coord[0], (coord[1] + 1) % (2*self.size) ], [ coord[0], (coord[1] - 1) % (2*self.size) ] ]

        if pauli_Z_flip:
            if coord[0] % 2 == 0:
                plaqs += [ [ coord[0], (coord[1] + 1) % (2*self.size) ], [ coord[0], (coord[1] - 1) % (2*self.size) ] ]
            else:
                plaqs += [ [ (coord[0] + 1) % (2*self.size), coord[1] ], [ (coord[0] - 1) % (2*self.size), coord[1] ] ]


        # Update syndrome positions
        for plaq in plaqs:
            if plaq in self.syndrome_pos:
                self.syndrome_pos.remove(plaq)
            else:
                self.syndrome_pos.append(plaq)

            # The plaquette or vertex operators are only 0 or 1
            if plaq in self.star_pos:
                op_index = self.star_pos.index(plaq)
                channel = 1
            elif plaq in self.plaquet_pos:
                op_index = self.plaquet_pos.index(plaq)
                channel = 0

            self.op_values[channel][op_index] = (self.op_values[channel][op_index] + 1) % 2


    def is_terminal(self):
        # Not needed I think
        #if len(self.get_legal_action()) == 0:
        #    return True

        # Are all syndromes removed?
        return len(self.syndrome_pos) == 0

    def has_logical_error(self, initialmoves, debug=False):
        if debug:
            print("Initial errors:", [self.qubit_pos.index(q) for q in initialmoves])

        # Check for Z logical error
        zerrors = [0,0]
        for pos in self.z1pos:
            if pos in initialmoves[0]:
                zerrors[0] += 1
            qubit_index = self.qubit_pos.index(pos)
            zerrors[0] += self.qubit_values[0][ qubit_index ]

        for pos in self.z2pos:
            if pos in initialmoves[0]:
                zerrors[1] += 1
            qubit_index = self.qubit_pos.index(pos)
            zerrors[1] += self.qubit_values[0][ qubit_index ]

        # Check for X logical error
        xerrors = [0,0]
        for pos in self.x1pos:
            if pos in initialmoves[1]:
                xerrors[0] += 1
            qubit_index = self.qubit_pos.index(pos)
            xerrors[0] += self.qubit_values[1][ qubit_index ]

        for pos in self.x2pos:
            if pos in initialmoves[1]:
                xerrors[1] += 1
            qubit_index = self.qubit_pos.index(pos)
            xerrors[1] += self.qubit_values[1][ qubit_index ]

        #print("Zerrors", zerrors)

        if (zerrors[0]%2 == 1) or (zerrors[1]%2 == 1) or \
            (xerrors[0]%2 == 1) or (xerrors[1]%2 == 1):
            return True

        return False


    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        return qubit_values, op_values

    def encode(self, channels, use_memory):
        '''Return: np array
            np.array(board_size, board_size): state observation of the board
        '''
        # In case of uncorrelated noise for instance, we don't need information
        # about the star operators

        img=np.array([])
        for channel in channels:
            img = np.concatenate((img, self.op_values[channel]))
            if use_memory:
                img = np.concatenate((img, self.qubit_values[channel]))
        return img

    def image_view(self, number=False, channel=0):
        image = np.empty((2*self.size, 2*self.size), dtype=object)
        for i, plaq in enumerate(self.plaquet_pos):
            if self.op_values[0][i] == 1:
                image[plaq[0], plaq[1]] = "P"+str(i) if number else "P"
            elif self.op_values[0][i] == 0:
                image[plaq[0], plaq[1]] = "x"+str(i) if number else "x"
        for i,plaq in enumerate(self.star_pos):
            if self.op_values[1][i] == 1:
                image[plaq[0], plaq[1]] = "S"+str(i) if number else "S"
            elif self.op_values[1][i] == 0:
                image[plaq[0], plaq[1]] = "+"+str(i) if number else "+"
        for i,pos in enumerate(self.qubit_pos):
            image[pos[0], pos[1]] = str(int(self.qubit_values[channel,i]))+str(i) if number else str(int(self.qubit_values[channel, i]))

        return np.array(image)
