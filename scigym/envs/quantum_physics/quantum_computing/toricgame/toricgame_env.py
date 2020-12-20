from __future__ import division
import numpy as np

import gym
from gym.utils import seeding
import sys, os
import matplotlib.pyplot as plt

### Environment
class ToricGameEnv(gym.Env):
    '''
    ToricGameEnv environment. Effective single player game.
    '''

    def __init__(self, board_size):
        """
        Args:
            opponent: Fixed
            board_size: board_size of the board to use
        """
        self.board_size = board_size

        self.seed()

        # Keep track of the moves
        self.qubits_flips = []
        self.initial_qubits_flips = []

        # Empty State
        self.state = Board(self.board_size)

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]

    def generate_errors(self, error_rate, mode):
        # Reset the board state
        self.state.reset()

        # Let the opponent do it's initial evil
        self.qubits_flips = []
        self.initial_qubits_flips = []
        self._set_initial_errors(error_rate, mode)

        self.done = self.state.is_terminal()
        self.reward = 0
        if self.done:
            self.reward = 1
            if self.state.has_logical_error(self.initial_qubits_flips):
                self.reward = -1

        return self.state.encode()

    def close(self):
        self.state = None

    def render(self, mode="human", close=False):
        fig, ax = plt.subplots(dpi=300)

        scale = 3/self.board_size

        array = self.state.encode()
        for p in self.state.plaquet_pos:
            fc = 'white' if array[p[0],p[1]]==0 else 'darkorange'#[1,0,0,0.8]
            plaq = plt.Rectangle( (-0.7 + scale*p[0]*0.25 - scale*0.25, 0.7 - scale*p[1]*0.25 + scale*0.25), scale*0.5, -0.5*scale, fc=fc, ec='black' )
            ax.add_patch(plaq)

        for i, p in enumerate(self.state.qubit_pos):
            circle = plt.Circle( (-0.7 + scale*0.25*p[0], 0.7 - scale*0.25*p[1]), radius=scale*0.05, ec='k', fc='darkgrey' if array[p[0],p[1]] == 0 else 'darkblue')
            ax.add_patch(circle)
            plt.annotate(str(i), (-0.7 + scale*0.25*p[0], 0.7 - scale*0.25*p[1]), fontsize=8, ha="center")

        #for p in g.toriccode.plaquet_pos:
        #    ax.text( -0.72 + 0.25*p[0], 0.68 - 0.25*p[1], "p")

        #for s in g.toriccode.star_pos:
        #    ax.text( -0.7 + 0.25*s[0], 0.7 - 0.25*s[1], "s")

        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')
        plt.show()

    def step(self, action):
        '''
        Args:
            action: coord of the qubit to flip
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

        # Check if we flipped twice the same qubit
        if action in self.qubits_flips:
            return self.state.encode(), -1.0, True, {'state': self.state, 'message': "illegal_action"}
        else:
            self.qubits_flips.append(action)
            self.state.act(action)

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

    def _set_initial_errors(self, error_rate, mode=0):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''
        # Probabilitic mode
        # Pick random sites according to error rate
        if mode == 0:
            for q in self.state.qubit_pos:
                if np.random.rand() < error_rate:
                    self.initial_qubits_flips.append( q )
                    self.state.act(q)

        # Deterministic mode
        # Select error_rate*n_qubits qubits to flip
        else:
            n_errors = int(error_rate * 2 * self.board_size * self.board_size)
            for q in np.random.permutation(self.state.qubit_pos)[:n_errors]:
                self.initial_qubits_flips.append( list(q) )
                self.state.act(list(q))

        # Now unflip the qubits, they're a secret
        for q in self.state.qubit_pos:
            self.state.board_state[q[0],q[1]] = 0

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

        self.reset()

    def reset(self):
        self.board_state = np.zeros( (2*self.size, 2*self.size) )
        self.syndrome_pos = [] # Location of syndromes
        self.perspectives = [] # Shifted indices to center the syndrome

    # Not used
    def isAdjacentToSyndrome(self, move):
        if self.board_state[(move[0]-1)%(2*self.size),move[1]] == 1:
            return True
        if self.board_state[(move[0]+1)%(2*self.size),move[1]] == 1:
            return True
        if self.board_state[move[0],(move[1]-1)%(2*self.size)] == 1:
            return True
        if self.board_state[move[0],(move[1]+1)%(2*self.size)] == 1:
            return True

        return False

    def act(self, coord):
        '''
            Args: input action in the form of position [x,y]
        '''

        # Flip it!
        self.board_state[coord[0],coord[1]] = (self.board_state[coord[0], coord[1]] + 1) % 2

        # Update the syndrome measurements
        # Only need to incrementally change
        # Find plaquettes that the flipped qubit is a part of
        # Only flips plaquettes operators (no star ones)
        if coord[0] % 2 == 0:
            plaqs = [ [ (coord[0] + 1) % (2*self.size), coord[1] ], [ (coord[0] - 1) % (2*self.size), coord[1] ] ]
        else:
            plaqs = [ [ coord[0], (coord[1] + 1) % (2*self.size) ], [ coord[0], (coord[1] - 1) % (2*self.size) ] ]

        # Update syndrome positions
        for plaq in plaqs:
            if plaq in self.syndrome_pos:
                self.syndrome_pos.remove(plaq)
            else:
                self.syndrome_pos.append(plaq)

            self.board_state[plaq[0],plaq[1]] = (self.board_state[plaq[0],plaq[1]] + 1) % 2


    def is_terminal(self):
        # Not needed I think
        #if len(self.get_legal_action()) == 0:
        #    return True

        # Are all syndromes removed?
        return len(self.syndrome_pos) == 0

    def has_logical_error(self, initialmoves, debug=False):
        if debug:
            print("Initial errors:", [self.qubit_pos.index(q) for q in initialmoves])

        # Check for Z error
        zerrors = [0,0]
        for pos in self.z1pos:
            if pos in initialmoves:
                zerrors[0] += 1
            zerrors[0] += self.board_state[ pos[0], pos[1] ]

        for pos in self.z2pos:
            if pos in initialmoves:
                zerrors[1] += 1
            zerrors[1] += self.board_state[ pos[0], pos[1] ]

        if (zerrors[0]%2 == 1) or (zerrors[1]%2 == 1):
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
        img = np.array(self.board_state)
        return img

    def image_view(self, number=False):
        image = np.array(self.board_state, dtype=object)
        for i, plaq in enumerate(self.plaquet_pos):
            if image[plaq[0], plaq[1]] == 1:
                image[plaq[0], plaq[1]] = "P"+str(i) if number else "P"
            elif image[plaq[0], plaq[1]] == 0:
                image[plaq[0], plaq[1]] = "x"+str(i) if number else "x"
        for i,plaq in enumerate(self.star_pos):
            if image[plaq[0], plaq[1]] == 1:
                image[plaq[0], plaq[1]] = "S"+str(i) if number else "S"
            elif image[plaq[0], plaq[1]] == 0:
                image[plaq[0], plaq[1]] = "+"+str(i) if number else "+"

        for i,pos in enumerate(self.qubit_pos):
            image[pos[0], pos[1]] = str(int(image[pos[0], pos[1]]))+str(i) if number else str(int(image[pos[0], pos[1]]))

        return np.array(image)
