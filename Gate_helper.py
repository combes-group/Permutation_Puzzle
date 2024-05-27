import numpy as np
import math

def S_gate(dim):
    """
    Constructs the Qudit S gate.
    [TODO: what is the arxiv reference and equation number?]

    :param dim: Hilbert Space dimension
    :return: numpy array
    """
    if dim == 2:
        return np.matrix(np.diag([1,1j]))
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
    [TODO: what is the arxiv reference and equation number?]

    :param dim: Hilbert Space dimension
    :return: numpy array
    """
    if dim == 2:
        return 1/np.sqrt(2)*np.matrix([[1,1],[1,-1]])
    omega = np.exp(1j*2*np.pi/dim)
    H = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        for j in range(dim):
            H[i,j] = 1/math.sqrt(dim) * omega**(j*i)
    return H

def Z_gate(dim):
    """
    Constructs the Qudit Z gate.
    [TODO: what is the arxiv reference and equation number?]

    :param dim: Hilbert Space dimension
    :return: numpy array
    """
    omega = np.exp(1j*2*np.pi/dim)
    return np.matrix(np.diag([omega**n for n in range(dim)]))