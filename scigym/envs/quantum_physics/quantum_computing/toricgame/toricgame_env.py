import sys, os
import numpy as np
import gym
from gym.utils import seeding

# Just in case a python version less than 3 is used
from __future__ import division

### Environment
class ToricGameEnv(gym.Env):
    '''
    ToricGameEnv environment. Effective single player game.
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, board_size, error_model, error_rate):
        """
        Args:
            opponent: Fixed
            board_size: board_size of the board to use
            error_model: 0 = uncorrelated, 1 = depolarizing
            error_rate: rate of physical qubit errors
        """
        self.board_size = board_size
        self.error_model = error_model
        self.error_rate = error_rate

        # Keep track of the moves
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]

        # Empty State
        self.state = Board(self.board_size)
        self.done = None

        # Set observation and action spaces
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(board_size[0],board_size[0]))
        self.action_space = gym.spaces.Discrete(3*2*board_size**2)

    def step(self, action):
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
            return self.state.encode(), 1., True, {'state': self.state, 'message':"success"}

        # Convert action to qubit and gate
        qubit_number = int(action / self.board_size)
        location = [int(qubit_number / self.board_size), int(qubit_number % self.board_size)]
        pauli_opt = int(action % self.board_size)

        # Determine whether we are flipping X, Y or Z
        pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
        pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)

        if pauli_X_flip and location in self.qubits_flips[0]:
            return self.state.encode(), -1.0, True, {'state': self.state, 'message': "illegal_action"}
        if pauli_Z_flip and location in self.qubits_flips[1]:
            return self.state.encode(), -1.0, True, {'state': self.state, 'message': "illegal_action"}

        # This qubit was not yet flipped
        if pauli_X_flip:
            self.qubits_flips[0].append(location)
        if pauli_Z_flip:
            self.qubits_flips[1].append(location)

        # Execute the flip on the board
        self.state.act(location, pauli_opt)

        # Reward: if nonterminal, then the reward is 0
        if not self.state.is_terminal():
            self.done = False
            self.reward = 0
            return self.state.encode(), 0., False, {'state': self.state, 'message':"continue"}

        # We're in a terminal state. Reward is 1 if won, -1 if lost
        self.done = True
        if self.state.has_logical_error(self.initial_qubits_flips):
            return self.state.encode(), -1.0, True, {'state': self.state, 'message':"logical_error"}
        else:
            return self.state.encode(), 1.0, True, {'state': self.state, 'message':"success"}

    def reset(self):
        self.generate_errors()

    def render(self, mode="human", close=False):
        if( mode != "human" ):
            raise NotImplemented("Please use \"human\" as the rendering mode")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        a=1/(2*self.board_size)

        for p in self.state.plaquet_pos:
            if self.state.board_state[0][p[0],p[1]]==1:
                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax.add_patch(plaq)

        for p in self.state.star_pos:
            if self.state.board_state[0][p[0],p[1]]==1:
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
            if self.state.board_state[0][p[0],p[1]] == 1 and self.state.board_state[1][p[0],p[1]] == 0:
                fc='darkblue'
            elif self.state.board_state[0][p[0],p[1]] == 0 and self.state.board_state[1][p[0],p[1]] == 1:
                fc='darkred'
            elif self.state.board_state[0][p[0],p[1]] == 1 and self.state.board_state[1][p[0],p[1]] == 1:
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
        # Derive a random seed
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]

    def generate_errors(self):
        # Reset the board state
        self.state.reset()

        # Let the opponent do it's initial evil
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]
        self._set_initial_errors(self.error_rate)

        self.done = self.state.is_terminal()
        self.reward = 0
        if self.done:
            self.reward = 1
            if self.state.has_logical_error(self.initial_qubits_flips):
                self.reward = -1

        return self.state.encode()

    def _set_initial_errors(self, error_rate):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''
        # Probabilitic mode
        # Pick random sites according to error rate

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
        for q in self.state.qubit_pos:
            self.state.board_state[0][q[0],q[1]] = 0
            self.state.board_state[1][q[0],q[1]] = 0

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

    def __init__(self, board_size):
        self.size = board_size

        self.qubit_pos   = [[x,y] for x in range(2*self.size) for y in range((x+1)%2, 2*self.size, 2)]
        self.plaquet_pos = [[x,y] for x in range(1,2*self.size,2) for y in range(1,2*self.size,2)]
        self.star_pos    = [[x,y] for x in range(0,2*self.size,2) for y in range(0,2*self.size,2)]

        # Define here the logical error for efficiency
        self.z1pos = [[0,x] for x in range(1, 2*self.size, 2)]
        self.z2pos = [[y,0] for y in range(1, 2*self.size, 2)]
        self.x1pos = [[1,x] for x in range(0, 2*self.size, 2)]
        self.x2pos = [[y,1] for y in range(0, 2*self.size, 2)]

        self.reset()

    def reset(self):
        self.board_state = np.zeros( (2, 2*self.size, 2*self.size) )
        self.syndrome_pos = [] # Location of syndromes

    def act(self, coord, operator):
        '''
            Args: input action in the form of position [x,y]
        '''

        pauli_X_flip = (operator==0 or operator==2)
        pauli_Z_flip = (operator==1 or operator==2)

        # Flip it!
        if pauli_X_flip:
            self.board_state[0][coord[0],coord[1]] = (self.board_state[0][coord[0], coord[1]] + 1) % 2
        if pauli_Z_flip:
            self.board_state[1][coord[0],coord[1]] = (self.board_state[1][coord[0], coord[1]] + 1) % 2

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
            self.board_state[0][plaq[0],plaq[1]] = (self.board_state[0][plaq[0],plaq[1]] + 1) % 2


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
            zerrors[0] += self.board_state[0][ pos[0], pos[1] ]

        for pos in self.z2pos:
            if pos in initialmoves[0]:
                zerrors[1] += 1
            zerrors[1] += self.board_state[0][ pos[0], pos[1] ]

        # Check for X logical error
        xerrors = [0,0]
        for pos in self.x1pos:
            if pos in initialmoves[1]:
                xerrors[0] += 1
            xerrors[0] += self.board_state[1][ pos[0], pos[1] ]

        for pos in self.x2pos:
            if pos in initialmoves[1]:
                xerrors[1] += 1
            xerrors[1] += self.board_state[1][ pos[0], pos[1] ]

        #print("Zerrors", zerrors)

        if (zerrors[0]%2 == 1) or (zerrors[1]%2 == 1) or \
            (xerrors[0]%2 == 1) or (xerrors[1]%2 == 1):
            return True

        return False

    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        out = self.board_state
        return out

    def encode(self):
        '''Return: np array
            np.array(board_size, board_size): state observation of the board
        '''
        #img = [self.board_state[0,x,y] for x in range(2*self.size) for y in range(2*self.size)
        #            if ([x,y] in self.plaquet_pos+self.star_pos)]
        img = np.array(self.board_state[0,:,:])
        return img

    def image_view(self, number=False, channel=0):
        image = np.array(self.board_state[0,:,:], dtype=object)
        #print(image)
        for i, plaq in enumerate(self.plaquet_pos):
            if image[plaq[0]][plaq[1]] == 1:
                image[plaq[0], plaq[1]] = "P"+str(i) if number else "P"
            elif image[plaq[0], plaq[1]] == 0:
                image[plaq[0], plaq[1]] = "x"+str(i) if number else "x"
        for i,plaq in enumerate(self.star_pos):
            if image[plaq[0], plaq[1]] == 1:
                image[plaq[0], plaq[1]] = "S"+str(i) if number else "S"
            elif image[plaq[0], plaq[1]] == 0:
                image[plaq[0], plaq[1]] = "+"+str(i) if number else "+"

        for i,pos in enumerate(self.qubit_pos):
            image[pos[0], pos[1]] = str(int(self.board_state[channel,pos[0], pos[1]]))+str(i) if number else str(int(self.board_state[channel, pos[0], pos[1]]))

        return np.array(image)
