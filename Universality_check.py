import numpy as np
import math


def Algebra_generators(dim):
    ## code taken from Qinfer
    basis = []
    # diagonal matricies
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

def Ad(U,G = -1):
    ## input: U gate in SU(d)
    ##        G generators of the lie algebra of SU(d)
    ## ouput: The adjoint representation of U in SO(d^2-1)
    d = U.shape[0]
    if G == -1:
        G = Algebra_generators(d)
    Ad_U = np.matrix(np.zeros((d**2-1,d**2-1),dtype=complex))
    for i in range(d**2-1):
        for j  in range(d**2-1):
            Ad_U[i,j] =  np.trace(G[i]*U*G[j]*np.conj(U.T))
    return Ad_U

def check_Center(S,G=-1):
     ## input: S a list of gates in SU(d)
     ##        G generators of the lie algebra of SU(d)
     ## ouput: True if the center of the subgroup described by S contains only multiples of the identity
    dim = S[0].shape[0]**2-1
    if G == -1:
        G = Algebra_generators(S[0].shape[0])
    I = np.matrix(np.eye(dim))
    Ms = np.matrix(np.zeros((dim**2*len(S),dim**2),dtype=complex))
    for i,gate in enumerate(S):
        Ms[i*dim**2:(i+1)*dim**2,:] = np.kron(I,Ad(gate,G)) - np.kron(Ad(np.conj(gate.T),G),I)
        
    dim_ker = dim**2 - np.linalg.matrix_rank(Ms)
    return dim_ker == 1

def ball_check(gate,N):
    ## input: gate a gate in SU(d)
    ##        N a maximum power of the gate U**n to check
    ## output:True if there exists a power of the gate 1<n<N such that gate^n is in B and gate^n is not the identity.
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
                if not np.allclose(eig**n,eig[0]**n*np.ones(dim)): # make sure all the eigenvalues arent the same
                    return True
    return False

def test_close(A,B):
    ## checks if two matricies A and B are close up to a relative phase
    dim = B.shape[0]
    return math.isclose(np.abs(np.trace(np.conj(A.T)@B)),dim,rel_tol=1e-9)

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

def check_Finite(S,N,lmax):
    ## input: S gate set in SU(d)
    ##        N a maximum power of the gate U**n to check
    ##        lmax longest length of words to check before the program terminates
    ## output:True if the subgroup spaned by S is infinite.
    dim = S[0].shape[0]
    G_s = [np.matrix(np.eye(dim))]
    new_index = 0
    next_index = 1
    for l in range(lmax):# check words up to length lmax starting at l = 0
        new_gates = []
        for gate in G_s[new_index:]: 
            if ball_check(gate,N):
                print('Infinite',gate) # if it is infinite also output the gate that is part of the ball and not the center
                return [False,-1]
            for U in S:
                new_gates.append(gate@U) 
      
        num_added = add_unique(np.array(new_gates),G_s)
        print('l = ',l, 'current size = ', len(G_s))
        new_index = next_index
        next_index += num_added 
        if num_added == 0:
            return [True,len(G_s)]
    print('Reached lmax')
    return [True,len(G_s)]

def check_universal(S,N=10,lmax=100):  
    if not check_Center(S):
        return False
    ball,number = check_Finite_eigs(S,N,lmax)
    return ball