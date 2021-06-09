import numpy as np
from scipy import linalg
from functools import reduce

from scigym.envs.quantum_physics.quantum_information.entangled_ions.operations.qudit_qm import QuditQM

class LaserGates(QuditQM):
    def __init__(self, dim, num_ions, phases):
        """
        This class generates the required gate set for an ion trap quantum
        computer as described in https://arxiv.org/pdf/1907.08569.pdf.
        Specifically, it implements single-qudit gates of the form Eq. (2) given
        angles theta (here 'pulse_angles') and phi (here 'pulse_phases').
        Moreover, it generates Molmer-Sorensen gates of the form Eq. (6) given
        angles theta (here 'ms_phases').

        Args:
            dim (int): The local (odd) dimension of an ion.
            num_ions (int): The number of ions.
            phases (dict): The phases defining the laser gate set.
                           We require the following structure:
                            {'pulse_angles': list,
                            'pulse_phases': list,
                            'ms_phases': list}
        """
        # add methods (such as `ket` or `bra`) from parent
        super().__init__(dim, num_ions)
        #list: List of angles theta in Eq. (2)
        self.pulse_angles = phases['pulse_angles']
        #list: List of angles phi in Eq. (2)
        self.pulse_phases = phases['pulse_phases']
        #list: List of angles theta in Eq. (6)
        self.ms_phases = phases['ms_phases']
        #`list` of `np.ndarray`: List of unitaries.
        self.gates = []
        # fills the `gates` with unitaries as specified by the angles.
        self._generate_gates()

    def pulse(self, j, k, theta, phi):
        """
        Single-ion laser pulse of the form Eq. (2) in 
        https://arxiv.org/pdf/1907.08569.pdf.

        .. math::
            G(j,k;\theta,\phi) = \exp(i\theta(e^{i\phi}|j\rangle\langle k| +
            e^{-i\phi}|k\rangle\langle j|))
        
        NOTE: According to the paper only nearest neighbor transitions 
        k = j+1 are practically implementable.

        Args:
            j (int): Basis vector |j> for transition |j><k|.
            k (int): Basis vector |k>for transition |j><k|.
            theta (float): Pulse angle (depending on the Rabi frequency).
            phi (float): Phase controlled by microwave or optical radiation.

        Returns:
            pulse (np.ndarray): The matrix representing the laser pulse's action
                                on an ion.
        """
        # |j><k|
        jk = self.ket(j)@self.bra(k)
        # |k><j|
        kj = self.ket(k)@self.bra(j)
        # definition Eq. (2)
        pulse = linalg.expm(-1.j*theta*(np.exp(1.j*phi)*jk+np.exp(-1.j*phi)*kj))

        return pulse

    def pad_gate(self, gate, ion):
        """
        Pads a single-ion gate with identities on other ions.

        Args:
            gate (np.ndarray): Single-ion gate to be padded.
            ion (int): The index of the ion on which the gate acts.

        Returns:
            multi_gate (np.ndarray): Tensor product of `gate` with identities on
                                     all other ions.
        """
        # define list of empty single-ion identities
        multi_gate = [np.eye(self.dim) for i in range(self.num_ions)]
        # add gate at the position of desired ion
        multi_gate[ion] = gate
        # apply tensor product between single-ion gates
        multi_gate = reduce(np.kron, multi_gate)

        return multi_gate
    
    def molmer_sorensen(self, theta):
        """
        Molmer-Sorensen (MS) gate from Eq. (6) of 
        https://arxiv.org/pdf/1907.08569.pdf.
        This is a global laser beam that can entangle all ions.

        .. math::
            U_{\mathrm{MS}}(\theta_0) = \exp\left(i\theta_0\left(\sum_{i=1}^N 
            S_{x,i}\right)^2\right)
            S_{x} = (S_{+} + S_{-})/2
        
        where S_{+} and S_{-} are creation and annihilation operators.

        Args:
            theta (float): The MS phase (depending on the Rabi frequency).

        Returns:
            ms (np.ndarray): MS gate unitary acting on all ions.
        """
        # define generalized X gate 
        sx = (self.creation() + self.annihilation())/2.
        # define empty ms gate exponent
        ms = np.zeros((self.dim**self.num_ions, self.dim**self.num_ions), 
                       dtype=complex
                     )
        # calculate matrix exponent, i.e. sum of Sx for all ions
        for n in range(self.num_ions):
            # define list of empty single-ion identities
            sx_n = [np.eye(self.dim) for i in range(self.num_ions)]
            # add Sx at position of the desired ion
            sx_n[n] = sx
            # apply tensor product between single-ion gates and add up
            ms += reduce(np.kron, sx_n)

        # definition Eq. (6)
        ms = linalg.expm(1.j*theta*(np.linalg.matrix_power(ms,2)))

        return ms

    # ---------------------- helper methods ------------------------------------
    
    def _generate_gates(self):
        """
        Adds single-ion and MS unitaries to the list of n-ion gates according to
        the provided list of phases.
        """
        # add ms gate for each angle theta
        for theta in self.ms_phases:
            self.gates.append(self.molmer_sorensen(theta))
        
        # adds all single-ion gates
        # for all ions
        for ion in range(self.num_ions):
            # for all transitions k <--> k+1
            for k in range(self.dim-1):
                # for all angles theta
                for theta in self.pulse_angles:
                    # for all angles phi
                    for phi in self.pulse_phases:
                        # create gate
                        gate = self.pulse(k, k+1, theta, phi)
                        # add n-ion gate
                        self.gates.append(self.pad_gate(gate, ion))
