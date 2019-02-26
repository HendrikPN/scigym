import gym
from gym import error, spaces, utils, error
from gym.utils import seeding
import copy
from itertools import product, starmap
import random
import numpy as np
import keras
from keras.models import load_model
import os
import pkg_resources

class SurfaceCodeEnv(gym.Env):
	"""
	A surface code environment for obtaining decoding agents in the fault-tolerant setting.
	In particular:

		- The visible state consists of a syndrome volume + completed action volume
		- The agent can perform single qubit Pauli flips on physical data qubits
		-   - an error volume is introduced if the agent does the identity
			- an error volume is introduced if the agent repeats the same move twice
		- a plus 1 reward is given for every action that results in the code being in the ground state space.
		- The terminal state criterion is satisfied for every hidden state that cannot be decoded by the referee/static decoder

	Note, this environment can:
		- cater for error_model in {"X","DP"}
		- cater for faulty syndromes - i.e. p_meas > 0

	Also, this environment provides all methods as required by an openAi gym class. In particular:
		- reset
		- step


	Attributes
	----------

	:param: d: The code distance
	:param: p_phys: The physical error probability on a single physical data qubit
	:param: p_meas: The measurement error probability on a single syndrome bit
	:param: error_model: A string in ["X, DP"]
	:param: use_Y: A boolean indicating whether the environment accepts Y Pauli flips as actions
	:param: volume_depth: The number of sequential syndrome measurements performed when generating a new syndrome volume.
	:param: static_decoder: A homology class predicting decoder for perfect syndromes.

	"""

	# Indicate that currently no render modes are available
	metadata = {'render.modes': []}

	# ----------------- Required gym.Env methods ---------------------------------------------------------------------

	def __init__(self, p_phys=0.001, p_meas=0.0, error_model="X", use_Y=False, volume_depth=1):

		self.d = 5
		self.p_phys = p_phys
		self.p_meas= p_meas
		self.error_model = error_model
		self.use_Y = use_Y
		self.volume_depth = volume_depth

		self.n_action_layers = 0
		if error_model == "X":
			self.num_actions = self.d**2 + 1
			self.n_action_layers = 1
			static_decoder_path = pkg_resources.resource_filename('scigym', 'envs/surfacecode/referee_decoders/X_decoder')
		elif error_model == "DP":
			static_decoder_path = pkg_resources.resource_filename('scigym', 'envs/surfacecode/referee_decoders/X_decoder')
			if use_Y:
				self.num_actions = 3*self.d**2 + 1
				self.n_action_layers = 3
			else:
				self.num_actions = 2*self.d**2 + 1 
				self.n_action_layers = 2
		else:
			print("specified error model not currently supported!")

		self.static_decoder = load_model(static_decoder_path)

		self.identity_index = self.num_actions -1
		self.identity_indicator = self.generate_identity_indicator(self.d)

		self.qubits = self.generateSurfaceCodeLattice(self.d)
		self.qubit_stabilizers = self.get_stabilizer_list(self.qubits, self.d)  
		self.qubit_neighbours = self.get_qubit_neighbour_list(self.d) 
		self.completed_actions = np.zeros(self.num_actions, int)
		
	
		self.observation_space=gym.spaces.Box(low=0,high=1, shape=(self.volume_depth+self.n_action_layers, 
			2*self.d+1, 2*self.d+1), dtype=np.uint8)
		
		self.action_space = gym.spaces.Discrete(self.num_actions)

		self.hidden_state = np.zeros((self.d, self.d), int)
		self.current_true_syndrome = np.zeros((self.d+1, self.d+1), int)
		self.summed_syndrome_volume = None         
		self.board_state = np.zeros((self.volume_depth + self.n_action_layers, 2 * self.d + 1, 2 * self.d + 1),int)

		self.completed_actions = np.zeros(self.num_actions, int)
		self.acted_on_qubits = set()
		self.legal_actions = set()
		self.done = False
		self.lifetime = 0

		self.multi_cycle = True

	def reset(self):
		"""
		Resetting of the environment introduces a new non-trivial syndrome volume.

		:return: self.board_state: The new reset visible state of the environment = syndrome volume + blank action history volume
		"""

		self.done = False
		self.lifetime = 0
		
		# Create the initial error - we wait until there is a non-trivial syndrome. BUT, the lifetime is still updated!
		self.initialize_state()

		# Update the legal moves available to us
		self.reset_legal_moves()

		return self.board_state

	def step(self, action):
		"""
		Given an action, this method executes the logic of the environment.

		:param: action: The action, given as an integer, supplied by some agent.
		:return: self.board_state: The new reset visible state of the environment = syndrome volume + action history volume
		:return: reward: The reward for the action
		:return: self.done: The boolean terminal state indicator
		:return: info: A dictionary via which additional diagnostic information can be provided. Empty here.
		"""

		new_error_flag = False
		done_identity = False
		if action == self.identity_index or int(self.completed_actions[action]) == 1:
			done_identity = True

		# 1) Apply the action to the hidden state
		action_lattice = self.index_to_move(self.d, action, self.error_model, self.use_Y)
		self.hidden_state = self.obtain_new_error_configuration(self.hidden_state, action_lattice)

		# 2) Calculate the reward
		self.current_true_syndrome = self.generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
		current_true_syndrome_vector = np.reshape(self.current_true_syndrome,(self.d+1)**2) 
		num_anyons = np.sum(self.current_true_syndrome)

		correct_label = self.generate_one_hot_labels_surface_code(self.hidden_state, self.error_model)
		decoder_label = self.static_decoder.predict(np.array([current_true_syndrome_vector]), batch_size=1, verbose=0)

		reward = 0

		if np.argmax(correct_label) == 0 and num_anyons == 0:
			reward = 1.0
		elif np.argmax(decoder_label[0]) != np.argmax(correct_label):
			self.done = True


		# 3) If necessary, apply multiple errors and obtain an error volume - ensure that a non-trivial volume is generated
		if done_identity:

			trivial_volume = True
			while trivial_volume: 
				self.summed_syndrome_volume = np.zeros((self.d + 1, self.d + 1), int)
				faulty_syndromes = []
				for j in range(self.volume_depth):
					error = self.generate_error(self.d, self.p_phys, self.error_model)
					if int(np.sum(error)!=0):
						self.hidden_state = self.obtain_new_error_configuration(self.hidden_state, error)
						self.current_true_syndrome = self.generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
					current_faulty_syndrome = self.generate_faulty_syndrome(self.current_true_syndrome, self.p_meas)
					faulty_syndromes.append(current_faulty_syndrome)
					self.summed_syndrome_volume += current_faulty_syndrome
					self.lifetime += 1

				if int(np.sum(self.summed_syndrome_volume)) != 0:
					trivial_volume = False

			for j in range(self.volume_depth):
				self.board_state[j, :, :] = self.padding_syndrome(faulty_syndromes[j])


			# reset the legal moves
			self.reset_legal_moves()

			# update the part of the state which shows the actions you have just taken
			self.board_state[self.volume_depth:,:,:] = np.zeros((self.n_action_layers, 2 * self.d + 1, 2 * self.d + 1),int)


		else:
			# Update the completed actions and legal moves
			self.completed_actions[action] = int(not(self.completed_actions[action]))
			if not action == self.identity_index:

				acted_qubit = action%(self.d**2)
				
				if acted_qubit not in self.acted_on_qubits:
					self.acted_on_qubits.add(acted_qubit)
					for neighbour in self.qubit_neighbours[acted_qubit]:
							for j in range(self.n_action_layers):
								self.legal_actions.add(neighbour + j*self.d**2)

				
			# update the board state to reflect the action thats been taken
			for k in range(self.n_action_layers):
					self.board_state[self.volume_depth + k, :, :] = self.padding_actions(self.completed_actions[k * self.d ** 2:(k + 1) * self.d ** 2])


		return self.board_state, reward, self.done, {}
	

	def render(self):
		raise NotImplementedError

	# ----------------- helper methods ---------------------------------------------------------------------

	def initialize_state(self):
		"""
		Generate an initial non-trivial syndrome volume
		"""

		self.done = False
		self.hidden_state = np.zeros((self.d, self.d), int)
		self.current_true_syndrome = np.zeros((self.d+1, self.d+1), int) 
		self.board_state = np.zeros((self.volume_depth + self.n_action_layers, 2 * self.d + 1, 2 * self.d + 1),int)
		
		trivial_volume = True
		while trivial_volume:
			self.summed_syndrome_volume = np.zeros((self.d + 1, self.d + 1), int)
			faulty_syndromes = []
			for j in range(self.volume_depth):
				error = self.generate_error(self.d, self.p_phys, self.error_model)
				if int(np.sum(error)) != 0:
					self.hidden_state = self.obtain_new_error_configuration(self.hidden_state, error)
					self.current_true_syndrome = self.generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
				current_faulty_syndrome = self.generate_faulty_syndrome(self.current_true_syndrome, self.p_meas)
				faulty_syndromes.append(current_faulty_syndrome)
				self.summed_syndrome_volume += current_faulty_syndrome
				self.lifetime += 1

			if int(np.sum(self.summed_syndrome_volume)) != 0:
				trivial_volume = False

		# update the board state to reflect the measured syndromes
		for j in range(self.volume_depth):
			self.board_state[j, :, :] = self.padding_syndrome(faulty_syndromes[j])


	def reset_legal_moves(self):
		"""
		Reset the legal moves
		"""

		self.completed_actions = np.zeros(self.num_actions, int)
		self.acted_on_qubits = set()
		self.legal_actions = set()

		legal_qubits = set()
		for qubit_number in range(self.d**2):

			# first we deal with qubits that are adjacent to violated stabilizers
			if self.is_adjacent_to_syndrome(qubit_number):
				legal_qubits.add(qubit_number)

		# now we have to make a list out of it and account for different types of actions
		self.legal_actions.add(self.identity_index)
		for j in range(self.n_action_layers):
			for legal_qubit in legal_qubits:
				self.legal_actions.add(legal_qubit + j*self.d**2)



	def is_adjacent_to_syndrome(self, qubit_number):
		"""
		Determine whether a qubit is adjacent to a violated stabilizer
		"""

		for stabilizer in self.qubit_stabilizers[qubit_number]:
			if self.summed_syndrome_volume[stabilizer] != 0:
				return True

		return False

	def padding_syndrome(self, syndrome_in):
		"""
		Pad a syndrome into the required embedding
		"""

		syndrome_out = np.zeros((2*self.d+1, 2*self.d+1),int)
		
		for x in range( 2*self.d+1 ):
			for y in range( 2*self.d+1 ):

				#label the boundaries and corners
				if x==0 or x== 2*self.d:
					if y%2 == 1:
						syndrome_out[x,y] = 1

				if y==0 or y== 2*self.d:
					if x%2 == 1:
						syndrome_out[x,y] = 1

				if x%2 == 0 and y%2 == 0: 
					# copy in the syndrome
					syndrome_out[ x, y ] = syndrome_in[ int(x/2), int(y/2) ]
				elif x%2 == 1 and y%2 == 1:
					if (x+y)%4 == 0:
						#label the stabilizers
						syndrome_out[x,y] = 1
		return syndrome_out
		
	def padding_actions(self,actions_in):
		"""
		Pad an action history for a single type of Pauli flip into the required embedding.
		"""
		actions_out = np.zeros( ( 2*self.d+1, 2*self.d+1 ),int )

		for action_index, action_taken in enumerate( actions_in ):
			if action_taken:
				row = int( action_index / self.d )
				col = int( action_index % self.d )

				actions_out[ int( 2*row+1 ), int( 2*col+1 ) ] = 1

		return actions_out

	def indicate_identity(self, board_state):
		"""
		Pad the action history to indicate that an identity has been performed.
		"""

		for k in range(self.n_action_layers):
			board_state[self.volume_depth + k,:, :] = board_state[self.volume_depth + k,:, :] + self.identity_indicator

		return board_state

	def get_qubit_stabilizer_list(self, qubits, qubit):
		""""
		Given a qubit specification [qubit_row, qubit_column], this function returns the list of non-trivial stabilizer locations adjacent to that qubit
		"""

		qubit_stabilizers = []
		row = qubit[0]
		column = qubit[1]
		for j in range(4):
			if qubits[row,column,j,:][2] != 0:     # i.e. if there is a non-trivial stabilizer at that site
				qubit_stabilizers.append(tuple(qubits[row,column,j,:][:2]))
		return qubit_stabilizers  

	def get_stabilizer_list(self, qubits, d):
		""""
		Given a lattice, this function outputs a list of non-trivial stabilizers adjacent to each qubit in the lattice, indexed row-wise starting from top left
		"""
		stabilizer_list = []
		for qubit_row in range(self.d):
			for qubit_column in range(self.d):
				stabilizer_list.append(self.get_qubit_stabilizer_list(qubits,[qubit_row,qubit_column]))
		return stabilizer_list

	def get_qubit_neighbour_list(self, d):
		""""
		Given a lattice, this function provides a list of the neighbouring qubits for each physical qubit.
		"""

		count = 0
		qubit_dict = {}
		qubit_neighbours = []
		for row in range(d):
			for col in range(d):
				qubit_dict[str(tuple([row,col]))] = count
				cells = starmap(lambda a,b: (row+a, col+b), product((0,-1,+1), (0,-1,+1)))
				qubit_neighbours.append(list(cells)[1:])
				count +=1
			
		neighbour_list = []
		for qubit in range(d**2):
			neighbours = []
			for neighbour in qubit_neighbours[qubit]:
				if str(neighbour) in qubit_dict.keys():
					neighbours.append(qubit_dict[str(neighbour)])
			neighbour_list.append(neighbours)

		return neighbour_list

	def generate_identity_indicator(self, d):
		""""
		A simple helper function to generate the array that will be added to the action history to indicate that an identity has been performed.
		"""

		identity_indicator = np.ones((2 *d + 1, 2 *d + 1),int)
		for j in range(d):
			row = 2*j + 1
			for k in range(d):
				col = 2*k + 1
				identity_indicator[row,col] = 0
		return identity_indicator

	# ------------- methods to be integrated -------------------------------


	def generateSurfaceCodeLattice(self, d):
		""""
		This function generates a distance d square surface code lattice. in particular, the function returns 
		an array which, for each physical qubit, details the code-stabilizers supported on that qubit. To be more
		precise:
		
		 - qubits[i,j,:,:] is a 4x3 array describing all the stabilizers supported on physical qubit(i,j)
			   -> for the surface code geometry each qubit can support up to 4 stabilizers
		 - qubits[i,j,k,:] is a 3-vector describing the k'th stabilizer supported on physical qubit(i,j)
			   -> qubits[i,j,k,:] = [x_lattice_address, y_lattice_address, I or X or Y or Z]
		
		:param: d: The lattice width and height (or, equivalently, for the surface code, the code distance)
		:return: qubits: np.array listing and describing the code-stabilizers supported on each qubit
		"""
		
		if np.mod(d,2) != 1:
			raise Exception("for the surface code d must be odd!")
			
		qubits = [ [ [
					   [ x, y, ((x+y)%2)*2+1],
					   [ x, y+1, ((x+y+1)%2)*2+1],
					   [ x+1, y, ((x+1+y)%2)*2+1],
					   [ x+1, y+1, ((x+1+y+1)%2)*2+1]
					] for y in range(d)] for x in range(d)]
		qubits = np.array(qubits)
		
		for x in range(d):
			for y in range(d):
				for k in range(4):
					if (qubits[x,y,k,0] == 0 and qubits[x,y,k,1]%2 == 0):
						qubits[x,y,k,2] = 0
					if (qubits[x,y,k,0] == d and qubits[x,y,k,1]%2 == 1):
						qubits[x,y,k,2] = 0
						
					if (qubits[x,y,k,1] == 0 and qubits[x,y,k,0]%2 == 1):
						qubits[x,y,k,2] = 0
					if (qubits[x,y,k,1] == d and qubits[x,y,k,0]%2 == 0):
						qubits[x,y,k,2] = 0
		return qubits


	def multiplyPaulis(self, a,b):
		""""
		A simple helper function for multiplying Pauli Matrices. Returns ab.
		:param: a: an int in [0,1,2,3] representing [I,X,Y,Z]
		:param: b: an int in [0,1,2,3] representing [I,X,Y,Z]
		"""
		
		out = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
		return out[int(a)][int(b)]


	def generate_error(self, d,p_phys,error_model):
		""""
		This function generates an error configuration, via a single application of the specified error channel, on a square dxd lattice.
		
		:param: d: The code distance/lattice width and height (for surface/toric codes)
		:param: p_phys: The physical error rate.
		:param: error_model: A string in ["X", "DP", "IIDXZ"] indicating the desired error model.
		:return: error: The error configuration
		"""
		
		if error_model == "X":
			return self.generate_X_error(d,p_phys)
		elif error_model == "DP":
			return self.generate_DP_error(d,p_phys)
		elif error_model == "IIDXZ":
			return self.generate_IIDXZ_error(d,p_phys)
			
		return error

	def generate_DP_error(self,d,p_phys):
		""""
		This function generates an error configuration, via a single application of the depolarizing noise channel, on a square dxd lattice.
		
		:param: d: The code distance/lattice width and height (for surface/toric codes)
		:param: p_phys: The physical error rate.
		:return: error: The error configuration
		"""

		error = np.zeros((d,d),int) 
		for i in range(d): 
			for j in range(d):
				p = 0
				if np.random.rand() < p_phys:
					p = np.random.randint(1,4)
					error[i,j] = p
					
		return error

	def generate_X_error(self, d,p_phys):
		""""
		This function generates an error configuration, via a single application of the bitflip noise channel, on a square dxd lattice.
		
		:param: d: The code distance/lattice width and height (for surface/toric codes)
		:param: p_phys: The physical error rate.
		:return: error: The error configuration
		"""
		
		
		error = np.zeros((d,d),int) 
		for i in range(d): 
			for j in range(d):
				p = 0
				if np.random.rand() < p_phys:
					error[i,j] = 1
		
		return error
					
	def generate_IIDXZ_error(self, d,p_phys):
		""""
		This function generates an error configuration, via a single application of the IIDXZ noise channel, on a square dxd lattice.
		
		:param: d: The code distance/lattice width and height (for surface/toric codes)
		:param: p_phys: The physical error rate.
		:return: error: The error configuration
		"""
		
		error = np.zeros((d,d),int)
		for i in range(d):
			for j in range(d):
				X_err = False
				Z_err = False
				p = 0
				if np.random.rand() < p_phys:
					X_err = True
					p = 1
				if np.random.rand() < p_phys:
					Z_err = True
					p = 3
				if X_err and Z_err:
					p = 2

				error[i,j] = p
		
		return error

	def generate_surface_code_syndrome_NoFT_efficient(self, error,qubits):
		""""
		This function generates the syndrome (violated stabilizers) corresponding to the input error configuration, 
		for the surface code.
		
		:param: error: An error configuration on a square lattice
		:param: qubits: The qubit configuration
		:return: syndrome: The syndrome corresponding to input error
		"""
		
		d = np.shape(error)[0]
		syndrome = np.zeros((d+1,d+1),int)

		for i in range(d): 
			for j in range(d):
				if error[i,j] != 0:
					for k in range(qubits.shape[2]):
						if qubits[i,j,k,2] != error[i,j] and qubits[i,j,k,2] != 0:
							a = qubits[i,j,k,0]
							b = qubits[i,j,k,1]
							syndrome[a,b] = 1 - syndrome[a,b]
							
		return syndrome

	def generate_faulty_syndrome(self, true_syndrome, p_measurement_error):
		""""
		This function takes in a true syndrome, and generates a faulty syndrome according to some
		given probability of measurement errors.
		
		:param: true_syndrome: The original perfect measurement syndrome
		:return: p_measurement_error: The probability of measurement error per stabilizer
		:return: faulty_syndrome: The faulty syndrome
		"""
		
		faulty_syndrome = np.zeros(np.shape(true_syndrome),int)

		# First we take care of the "bulk stabilizers"
		for row in range(1, true_syndrome.shape[0]-1):
			for col in range(1,true_syndrome.shape[1]-1):
				if np.random.rand() < p_measurement_error:
					faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
				else:
					faulty_syndrome[row,col] = true_syndrome[row,col]

		# Now we take care of the boundary stabilizers
		row = 0
		for col in [2*x +1 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
			if np.random.rand() < p_measurement_error:
					faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
			else:
				faulty_syndrome[row,col] = true_syndrome[row,col]
		row = true_syndrome.shape[0] - 1
		for col in [2*x + 2 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
			if np.random.rand() < p_measurement_error:
					faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
			else:
				faulty_syndrome[row,col] = true_syndrome[row,col]

		col = 0
		for row in [2*x + 2 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
			if np.random.rand() < p_measurement_error:
					faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
			else:
				faulty_syndrome[row,col] = true_syndrome[row,col]
		col = true_syndrome.shape[0] - 1
		for row in [2*x +1 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
			if np.random.rand() < p_measurement_error:
					faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
			else:
				faulty_syndrome[row,col] = true_syndrome[row,col]
	  
		return faulty_syndrome


	def obtain_new_error_configuration(self, old_configuration,new_gates):
		""""
		This function generates a new error configuration out of an old configuration and a new configuration,
		 which might arise either from errors or corrections.
		
		:param: old_configuration: An error configuration on a square lattice
		:param: new_gates: An error configuration on a square lattice
		:return: new_configuration: The resulting error configuration
		"""
		
		new_configuration = np.zeros(np.shape(old_configuration))
		for row in range(new_configuration.shape[0]):
			for col in range(new_configuration.shape[1]):
				new_configuration[row,col] = self.multiplyPaulis(new_gates[row,col], old_configuration[row,col])
				
		return new_configuration

	def index_to_move(self, d,move_index,error_model,use_Y=True):
		""""
		Given an integer index corresponding to a Pauli flip on a physical data qubit, this
		function generates the lattice representation of the move.
		
		:param: d: The code distance
		:param: move_index: The integer representation of the Pauli Flip
		:param: error_model: A string in ["X", "DP", "IIDXZ"] indicating the desired error model.
		:param: use_Y: a boolean indicating whether or not Y flips are allowed
		:return: new_move: A lattice representation of the desired move.
		"""

		new_move = np.zeros((d,d))
		
		if error_model == "X":
			if move_index < (d**2):

				move_type = 1
				new_index = move_index
				row = new_index/d
				col = new_index%d

				new_move[int(row),int(col)] = move_type

		elif error_model == "DP":
			if use_Y:
				if move_index < (d**2)*3:
					move_type = int(move_index/d**2) + 1
					new_index = move_index - (move_type - 1)*d**2

					row = new_index/d
					col = new_index%d

					new_move[int(row),int(col)] = move_type
			else:
				if move_index < (d**2)*2:
					move_type = int(move_index/d**2) + 1

					new_index = move_index - (move_type - 1)*d**2

					row = new_index/d
					col = new_index%d

					if move_type == 2:
						move_type = 3

					new_move[int(row),int(col)] = move_type

		else:
			print("Error model you have specified is not currently supported")

		return new_move

	def generate_one_hot_labels_surface_code(self, error,err_model):
		""""

		This function generates the homology class label, in a one-hot encoding, for a given perfect syndrome, to use as the target label
		for a feed forward neural network homology class predicting decoder.

		:param: error: An error configuration on a square lattice
		:param: err_model: A string in ["IIDXZ","DP","X"]
		:return: training_label: The one-encoded training label
		"""
		
		d = error.shape[0]
		
		X = 0
		Z = 0
			
		for x in range(d):
			if error[x,0] == 1 or error[x,0] == 2:
				X = 1 - X
		for y in range(d):
			if error[0,y] == 3 or error[0,y] == 2:
				Z = 1 - Z
				
		if err_model in ["IIDXZ","DP"]:
			training_label = np.zeros(4,int)                       
		else:
			training_label = np.zeros(2,int)

		training_label[X + 2*Z] = 1
		
		return training_label

