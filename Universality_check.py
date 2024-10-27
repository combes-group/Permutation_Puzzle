import numpy as np
import math

## Helper functions for checking the center of the group ##
def valid_gateset(S):
    """
    :param S: a list numpy arrays
    :return: Bool. True if everything in S is a numpy array representing a matrix in SU(d)
    """
    dim = S[0].shape[0]
    for gate in S:
        ## make sure all gates are the same dimension and square
        if gate.shape[0] != dim or gate.shape[1] != dim:
            return False
        
        ## make sure all gates are unitary
        if not np.allclose(np.conj(np.transpose(gate))@gate,np.eye(dim)):
            return False
        
        ## check if determinant is 1
        if not np.isclose(np.abs(np.linalg.det(gate)),1):
            return False
        
    return True

def SO_basis(dim):
    """
    Constructs the generalized Gell-Mann matrices that form a basis of SO(dim^2-1)
    Adapted from the function `gell_mann_basis` in the QInfer package:
    https://github.com/QInfer/python-qinfer/blob/master/src/qinfer/tomography/bases.py

    Also see https://mathworld.wolfram.com/GeneralizedGell-MannMatrix.html
    
    :param dim: Hilbert Space dimension
    :return: a list of dim * dim numpy arrays representing the generalized Gell-Mann matrices
    """
    basis = []
    # diagonal matrices
    for idx_basis in range(1, dim):
        a = np.diag(np.concatenate([
            np.ones((idx_basis, )),
            [-idx_basis],
            np.zeros((dim - idx_basis - 1, ))
        ])) / np.sqrt(idx_basis + idx_basis**2)
        basis.append(np.matrix(a))
        
    y_offset = dim * (dim - 1) // 2
    for idx_i in range(1, dim):
        for idx_j in range(idx_i):
            a = np.zeros((dim, dim), dtype=complex)
            b = np.zeros((dim, dim), dtype=complex)
            a[[idx_i, idx_j], [idx_j, idx_i]] = 1 / np.sqrt(2)
            b[[idx_i, idx_j], [idx_j, idx_i]] = [1j / np.sqrt(2), -1j / np.sqrt(2)]
            basis.append(np.matrix(a))
            basis.append(np.matrix(b))
    
    return basis


def Ad(U, G=None):
    """
    Implements Equation 4 of Sawicki and Karnas https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062303

    :param U: numpy array representing a unitary gate U in SU(d)
    :param G: Optional basis of SO(d^2-1) generated from `SO_basis` to avoid recomputing the full basis set
    :return: numpy array with the adjoint representation of U in SO(d^2-1)
    """
    d = U.shape[0]
    if G == None:
        G = SO_basis(d)
    Ad_U = np.matrix(np.zeros((d**2-1,d**2-1),dtype=complex))
    for i in range(d**2-1):
        for j  in range(d**2-1):
            # Note that the factor of -1/2 in Eq. 4 is due to their definition of
            # inner product, and that factor is "already incorporated" into
            # our definition of the generalized Gell-Mann matrices
            Ad_U[i,j] =  np.trace(G[i]*U*G[j]*U.conj().T)
    return Ad_U

## Helper functions for checking the cardinality of the group ##

def ball_check(gate, N):
    """
    Implements Equation 17 of Sawicki and Karnas 
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062303

    :param gate: a unitary gate in SU(d) represented as a numpy array
    :param N: int, a maximum power of the gate U**n to check
    
    Fact 5.6. of  https://arxiv.org/abs/1609.05780 provides an explicit
    upper bound on N, which scales exponentially with the Hilbert space dimension
    For example, for SU(6) the upper bound on N is of order 10^7 which is impractical.
    Instead we allow this to be input as a free parameter. 
    Thus if N is set too low, it is possible that a universal gate set will 
    incorrectly decided as not universal.
    
    :return: Bool. True if there exists a power of the gate 1<n<N such
             that gate^n is in the ball B_{alpha_m} and gate^n is not
             equivalant to the identity.
    """
    dim = gate.shape[0]
    eig,_ = np.linalg.eig(gate) 
    for n in range(N):
        for m in range(dim):
            theta = 2*np.pi*m/dim
            eig_sum = 0
            for i in range(dim):
                phi = n*np.angle(eig[i])
                eig_sum += np.sin((phi-theta)/2)**2
            if eig_sum < 1/8:
                # Assuming gate is unitary, then it is proportional to the
                # identity iff all eigenvalues are identical
                # furthermal if all eigenvalues are identical then they must
                # all be a multiple of the root of unity given by the Hilbert dimension 
                # however it is sufficient to simply check they are identical
                if not np.allclose(eig**n,eig[0]**n*np.ones(dim)): 
                    return True
    return False


def test_close(A, B, tol=1e-9):
    """
    Checks if two square matrices A and B are close up to a relative phase:
    
    Tr[A^\dag B] - dim

    :param A: numpy array
    :param B: numpy array
    :param tol: scalar
    :return: scalar
    """ 
    dim = B.shape[0]
    return math.isclose(np.abs(np.trace(np.conj(A.T)@B)),dim,rel_tol=tol)


def add_unique(new_elems, group_elems):
    """
    Checks if the set of new elements are in group elems,
    and adds any that are not present to group_elems
    
    param: new_elems list of numpy arrays
    param: group_elems list of numpy arrays
    returns: Number of elements added
    """
    def elem_in_group(elem):
        for group_elem in group_elems:
            if test_close(elem,group_elem):
                return True
        return False

    added = 0
    for new_elem in new_elems:
        if not(elem_in_group(new_elem)): 
            group_elems.append(new_elem)
            added += 1
    return added

### Functions that Implement Algorithmic check of Universality ##

def check_center(S, tol=1e-8):
    """
    Implements Equation 10 of Sawicki and Karnas 
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062303

    :param S: a list of gates in SU(d) represented as numpy arrays
    :param tol: tolerance for numerical matrix rank calculation
    :return: Bool. True if the center of the subgroup described by S 
             contains only multiples of the identity

    If tolerance is too low, might mistakenly believe Ms matrix is higher rank
    than it is in which case we might mistakenly return True
    """
    assert(valid_gateset(S))
    dim = S[0].shape[0]**2-1
    I = np.matrix(np.eye(dim))
    Ms = np.matrix(np.zeros((dim**2*len(S),dim**2),dtype=complex))
    for i,gate in enumerate(S):
        Ms[i*dim**2:(i+1)*dim**2,:] = np.kron(I,Ad(gate)) - np.kron(Ad(gate.conj().T),I)
        
    dim_ker = dim**2 - np.linalg.matrix_rank(Ms,tol)
    return dim_ker == 1



def check_finite(S, N, lmax, verbose=True):
    """
    Implements steps 2 and 3 of the algorithm from sec IV of 
    Sawicki and Karnas
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062303 
    
    :param S: gate set in SU(d) given as a list of numpy arrays
    :param N: a maximum integer power of the gate U**n to check
    :param lmax: integer maximum word length of gates from gateset to enumerate
    
    returns: False if the subgroup spaned by S is infinite otherwise
             when the group is finite, returns the size of the group
    """
    assert(valid_gateset(S))
    dim = S[0].shape[0]
    G_s = [np.matrix(np.eye(dim))]
    new_index = 0
    next_index = 1
    
    # check words up to length lmax starting at l = 0
    for l in range(lmax):
        new_gates = []
        for gate in G_s[new_index:]: 
            if ball_check(gate, N):
                if(verbose):
                    # if it is infinite also output the gate that is part of 
                    # the ball and not the center
                    print('Infinite',gate) 
                return False
            for U in S:
                new_gates.append(gate@U) 
      
        num_added = add_unique(np.array(new_gates), G_s)
        if(verbose):
            print('l = ', l, 'current size = ', len(G_s))
        new_index = next_index
        next_index += num_added 
        if num_added == 0:
            return len(G_s)

    print('Reached lmax')
    return len(G_s)

def check_universal(S, N=10, lmax=100, verbose=False):  
    """
    Implements the algorithm from Sec IV of of Sawicki and Karnas 
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062303 
    Determines if a gate set S is universal.

    :param S: List of numpy matricies 
    :param N: Integer maximum power to of matricies to check
    :param lmax: Integer maximum word length of gates from gateset to enumerate
    For small N  it is possible that a universal gate set will 
    incorrectly be said to be nonuniversal
    For small lmax it is possible that a gate set that generates an infinite group, 
    might incorrectly be listed as nonuniversal
    
    :return: boolean 
    """ 
    if not check_center(S):
        return False
    finite = check_finite(S, N, lmax, verbose=verbose)
    return not(finite)
