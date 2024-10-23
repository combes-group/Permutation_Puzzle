import numpy as np
import math

def S_gate(dim):
    """
    Constructs the Qudit S gate.
    Equation 8 of https://arxiv.org/abs/1911.08162

    :param dim: Hilbert Space dimension
    :return: numpy array
    """
    omega = np.exp(1j*2*np.pi/dim)
    S_gate = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        if dim % 2 == 0:
            S_gate[i,i] = omega**(i*(i)/(2))
        else:
            S_gate[i,i] = omega**(i*(i+1)/(2))
    return S_gate

def H_gate(dim):
    """
    Constructs the Qudit H gate.
    Equation 7 of https://arxiv.org/abs/1911.08162

    :param dim: Hilbert Space dimension
    :return: numpy array
    """
    omega = np.exp(1j*2*np.pi/dim)
    H = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        for j in range(dim):
            H[i,j] = 1/math.sqrt(dim) * omega**(j*i)
    return H

def Z_gate(dim):
    """
    Constructs the Qudit Z gate.
    Section 2 of https://arxiv.org/abs/quant-ph/9802007
    
    :param dim: Hilbert Space dimension
    :return: numpy array
    """
    omega = np.exp(1j*2*np.pi/dim)
    return np.matrix(np.diag([omega**n for n in range(dim)]),dtype=complex)

def X_gate(dim):
    """
    Constructs the Qudit X gate.
    Section 2 of https://arxiv.org/abs/quant-ph/9802007
    
    :param dim: Hilbert Space dimension
    :return: numpy array
    """
    
    X = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        X[i,i-1] = 1
    return X

def Y_gate(dim):
    """
    Constructs the Qudit Y gate.
    
    :param dim: Hilbert Space dimension
    :return: numpy array
    """
    
    return 1j*X_gate(dim)@Z_gate(dim)
