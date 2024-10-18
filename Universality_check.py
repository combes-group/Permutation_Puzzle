import numpy as np
import math


def SO_basis(dim):
    """
    Constructs the generalized Gell-Mann matrices that form a basis of SO(dim^2-1)
    See https://github.com/QInfer/python-qinfer/blob/master/src/qinfer/tomography/bases.py
    adapted from def gell_mann_basis
    :param dim: Hilbert Space dimension
    :return: a list of dim * dim numpy arrays that are the Gell-Mann matrices
    """
    basis = []
    # diagonal matrices
    for idx_basis in range(1, dim):
        a= np.diag(np.concatenate([
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


def Ad(U,G = None):
    """
    Implements Equation 4 of Sawicki and Karnas https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062303

    :param U: U gate in SU(d)
    :param G: Basis of SO(d^2-1)
    
    :return: numpy arrays that is The adjoint representation of U in SO(d^2-1)
    """
    d = U.shape[0]
    if G == None:
        G = SO_basis(d)
    Ad_U = np.matrix(np.zeros((d**2-1,d**2-1),dtype=complex))
    for i in range(d**2-1):
        for j  in range(d**2-1):
            Ad_U[i,j] =  np.trace(G[i]*U*G[j]*np.conj(U.T))
    return Ad_U


def check_center(S,G=None,tol=10**-8):
    """
    Implements Equation 10 of Sawicki and Karnas 
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062303

    :param S: a list of gates in SU(d)
    :param G: G generators of the lie algebra of SU(d)
    :param tol: tolerance for numerical matrix rank calculation
    If tolerance is to low it might mistakenly believe matrix is higher rank than it is.
    
    :return: Bool. True if the center of the subgroup described by S 
             contains only multiples of the identity
    """
    dim = S[0].shape[0]**2-1
    if G == None:
        G = SO_basis(S[0].shape[0])
    I = np.matrix(np.eye(dim))
    Ms = np.matrix(np.zeros((dim**2*len(S),dim**2),dtype=complex))
    for i,gate in enumerate(S):
        Ms[i*dim**2:(i+1)*dim**2,:] = np.kron(I,Ad(gate,G)) - np.kron(Ad(np.conj(gate.T),G),I)
        
    dim_ker = dim**2 - np.linalg.matrix_rank(Ms,tol)
    return dim_ker == 1


def ball_check(gate,N):
    """
    Implements Equation 17 of Sawicki and Karnas 
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062303

    :param S: a list of gates in SU(d)
    :param N: int, a maximum power of the gate U**n to check
    Sawicki provides a bound on N which is shown for SU(2) in equation 19
    This bound scales poorly with hilbert space dimension, for SU(6) it is 36398100 
    which is impratical. Instead we allow this to be input as a free parameter. 
    For N less than the bound it is possible that a universal gate set will 
    incorrectly be said to be nonuniversal
    
    :return: Bool. True if there exists a power of the gate 1<n<N such
             that gate^n is in B and gate^n is not the identity.
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
                # make sure all the eigenvalues arent the same
                # This is needed to determine if the matrix is a multiple of I
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
    ## Brute force checks if the set of new elements are in group elems,
    ## and adds any that are not present to group_elems
    ## returns the number of elements added
    added = 0
    for new_elem in new_elems:
        flag = False
        for group_elem in group_elems:
            if test_close(new_elem,group_elem):
                flag = True
                break
        if not(flag): 
            group_elems.append(new_elem)
            added += 1
    return added

def check_finite(S,N,lmax,verbose=True):
    ## input: S gate set in SU(d)
    ##        N a maximum power of the gate U**n to check
    ##        lmax longest length of words to check before the program terminates
    ## output:True if the subgroup spaned by S is infinite.
    dim = S[0].shape[0]
    G_s = [np.matrix(np.eye(dim))]
    new_index = 0
    next_index = 1
    
    # check words up to length lmax starting at l = 0
    for l in range(lmax):
        new_gates = []
        for gate in G_s[new_index:]: 
            if ball_check(gate,N):
                if(verbose):
                    # if it is infinite also output the gate that is part of 
                    # the ball and not the center
                    print('Infinite',gate) 
                return (False,None)
            for U in S:
                new_gates.append(gate@U) 
      
        num_added = add_unique(np.array(new_gates),G_s)
        if(verbose):
            print('l = ',l, 'current size = ', len(G_s))
        new_index = next_index
        next_index += num_added 
        if num_added == 0:
            return (True,len(G_s))
    print('Reached lmax')
    return (True,len(G_s))

def check_universal(S,N=10,lmax=100):  
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
    finite_bool,number = check_finite(S,N,lmax,verbose=False)
    return not(finite_bool)