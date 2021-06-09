import numpy as np
from functools import reduce
from itertools import product
from copy import deepcopy

class QuditQM(object):
    def __init__(self, dim, num_ions):
        """
        Here we define some useful methods for n d-level quantum systems in
        accordance with https://arxiv.org/pdf/1907.08569.pdf.
        Specifically, this includes basis states |k> and <k|, creation S_{+} and
        annihilation S_{-} operators. Further, we introduce some useful
        mathematical tools such as partial trace and Schmidt rank vector.

        Args:
            dim (int): The local (odd) dimension of an ion.
            num_ions (int): The number of ions.
        """
        if dim % 2 == 0:
            raise NotImplementedError('Our methods are only defined for odd ' +
                                      f'dimensions. Your dimension: {dim}.')
        self.dim = dim
        self.num_ions = num_ions

    def ket(self, i):
        """
        Basis vector |i> = (0,...,0,1,0,...,0).

        Args:
            i (int): Index of basis vector.

        Returns:
            vec (np.ndarray): Vector form of |i> as (d,1)-dimensional array.
        """
        vec = np.zeros((self.dim, 1), dtype=complex)
        vec[i] = 1.
        
        return vec

    def bra(self, i):
        """
        Basis vector <i| = (0,...,0,1,0,...,0)^T.

        Args:
            i (int): Index of basis vector.
        
        Returns:
            vec (np.ndarray): Vector form of <i| as (1,d)-dimensional array.
        """
        vec = self.ket(i).conj().T

        return vec

    def to_density(self, ket):
        """
        Creates density matrix from complex vector |psi> as |psi><psi|.

        Args:
            ket (np.ndarray): Complex (D,1)-dimensional array. 
        
        Returns:
            rho (np.ndarray): Complex (D,D)-dimensional density matrix.
        """
        rho = ket@ket.conj().T

        return rho

    def creation(self):
        """
        Creation operator S_{+} as defined in Eq. (5) for odd dimensions.

        .. math::
            S_{+} = \sum_{l=-s}^{s-1}\sqrt{s(s+1)-l(l+1)}
            |l+s+1\rangle\langle{l+s}|

        Returns:
            creation (np.ndarray): Creation operator of given dimension.
        """
        # define empty creation operator
        creation = np.zeros((self.dim, self.dim), dtype=complex)

        # add summand |l+s+1><l+s| as defined in Eq. (5)
        s = int((self.dim - 1)/2)
        for l in range(-s, s):
            m = np.sqrt(s*(s+1) - l*(l+1))*(self.ket(l+s+1)@self.bra(l+s))
            creation += m
        
        return creation

    def annihilation(self):
        """
        Annihilation operator S_{-} as defined in Eq. (5) for odd dimensions.

        .. math::
            S_{-} = \sum_{l=-s}^{s-1}\sqrt{s(s+1)-l(l+1)}
            |l+s\rangle\langle{l+s+1}|

        Returns:
            annihilation (np.ndarray): Annihilation operator of given dimension.
        """
        # define empty annihilation operator
        annihilation = np.zeros((self.dim, self.dim), dtype=complex)
        s = int((self.dim - 1)/2)

        # add summand |l+s><l+s+1| as defined in Eq. (5)
        for l in range(-s, s):
            m = np.sqrt(s*(s+1) - l*(l+1))*(self.ket(l+s)@self.bra(l+s+1))
            annihilation += m
        
        return annihilation
    
    def partial_tr(self, rho, *args):
        """
        Partial trace for a density matrix of multiple qudits (ions).
        Given a set of indices of qudits, this methods traces the respective
        qudits.

        Args:
            rho (np.ndarray): Density matrix which is to be traced.
            *args (int): Variable list of indices of qudits/ions which are to be 
                         traced.

        Returns:
            reduced_rho (np.ndarray): Reduced density matrix.
        """
        # define empty reduced density matrix
        dim_reduced = self.dim**(self.num_ions-len(args))
        reduced_rho = np.zeros((dim_reduced, dim_reduced), dtype=complex)
        # define list of identities to be replaced by trace operator/contractor
        basis = [np.eye(self.dim, dtype=complex) for i in range(self.num_ions)]
        # replace id with all-zero ket vectors for ions which are to be traced
        for ion in args:
            basis[ion] = np.zeros((self.dim, 1), dtype=complex)
        # perform contraction <i_1,..,i_k|\rho|i_1,...,i_k> for all bases
        for d_vec in product(range(self.dim), repeat=len(args)):
            # fix basis state |i_1,...,i_k>
            contractor = deepcopy(basis)
            for n, ion in enumerate(args):
                contractor[ion][d_vec[n]] = 1.
            contractor = reduce(np.kron, contractor)
            # add contraction
            reduced_rho += contractor.conj().T@rho@contractor

        return reduced_rho

    def srv(self, ket):
        """
        Calculates the SRV of an arbitrary pure state.

        Args:
            ket (np.ndarray): The input state.

        Returns:
            srv (list): List of integeres specifying the Schmidt ranks of all
                        single-qudit marginals.
        """
        # get density matrix from pure state
        rho = self.to_density(ket)
        # set of all ions
        ions = set(range(self.num_ions))
        # define empty list of SRs
        srv = []
        # calculate SR for all single-qudit marginals
        for i in range(self.num_ions):
            # perform partial trace over the respective ion
            reduced_rho = self.partial_tr(rho, i)
            # calculate rank of reduce density matrix
            rank = np.linalg.matrix_rank(reduced_rho)
            # add to SRV
            srv.append(rank)

        return srv
