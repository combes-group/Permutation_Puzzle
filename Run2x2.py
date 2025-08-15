import  os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import pandas as pd
import Strategy_helper
import twobytwohelper

from multiprocessing import Pool
pool = Pool(processes=100)
from tqdm import tqdm

def is_permutation_matrix(matrix):
    matrix = np.abs(matrix)
    rows, cols = matrix.shape
    if rows != cols:
        return False
    
    # Check if all elements are 0 or 1
    if not np.all(np.logical_or(matrix == 0, matrix == 1)):

        return False
    
    # Check if each row and column sums to 1
    if not (np.all(np.abs(matrix.sum(axis=1)) == 1) and np.all(np.abs(matrix.sum(axis=0)) == 1)):
        return False
    
    return True

## 2-1-1 puzzle
def basis(Nmax,n):
    vec = np.zeros((Nmax,1))
    vec[n] = 1
    return np.array(vec)

def main():
    Nmax = 12
    U = 1/2*(basis(Nmax,0)@basis(Nmax,0).T + basis(Nmax,1)@basis(Nmax,1).T) \
        + basis(Nmax,2)@basis(Nmax,6).T + basis(Nmax,3)@basis(Nmax,7).T\
        + basis(Nmax,4)@basis(Nmax,8).T + basis(Nmax,5)@basis(Nmax,9).T\
        + basis(Nmax,10)@basis(Nmax,11).T
    U = U + U.T
    
    D = 1/2*(basis(Nmax,10)@basis(Nmax,10).T + basis(Nmax,11)@basis(Nmax,11).T) \
        + basis(Nmax,2)@basis(Nmax,4).T + basis(Nmax,3)@basis(Nmax,5).T\
        + basis(Nmax,6)@basis(Nmax,8).T + basis(Nmax,7)@basis(Nmax,9).T\
        + basis(Nmax,0)@basis(Nmax,1).T
    D = D + D.T
    
    R = 1/2*(basis(Nmax,6)@basis(Nmax,6).T + basis(Nmax,7)@basis(Nmax,7).T) \
        + basis(Nmax,2)@basis(Nmax,0).T + basis(Nmax,3)@basis(Nmax,1).T\
        + basis(Nmax,8)@basis(Nmax,11).T + basis(Nmax,10)@basis(Nmax,9).T\
        + basis(Nmax,4)@basis(Nmax,5).T
    R = R + R.T
    
    L = 1/2*(basis(Nmax,4)@basis(Nmax,4).T + basis(Nmax,5)@basis(Nmax,5).T) \
        + basis(Nmax,9)@basis(Nmax,0).T + basis(Nmax,8)@basis(Nmax,1).T\
        + basis(Nmax,3)@basis(Nmax,11).T + basis(Nmax,10)@basis(Nmax,2).T\
        + basis(Nmax,6)@basis(Nmax,7).T
    L = L + L.T
    
    
    # define moves
    CS = [U,D,L,R]
    QS = []
    MS = []
    for move in CS:
        QS.append((np.eye(Nmax) + 1j*move)/np.sqrt(2))
        QS.append((np.eye(Nmax) - 1j*move)/np.sqrt(2))
        MS.append((np.eye(Nmax) + 1j*move)/np.sqrt(2))
        MS.append((np.eye(Nmax) - 1j*move)/np.sqrt(2))
        MS.append(move)
    
    for gate in CS:
        assert(is_permutation_matrix(gate))
    
    ## Generate a bunch of scrambles
    Num_scrambles = 200
    min_length = 200
    max_length = 500
    Scrambles = []
    for i in range(Num_scrambles):
        l = random.randint(min_length,max_length)
        move_list = random.choices(QS,k=l)
        state = basis(Nmax,0) # choose the 0 state as solved
        for move in move_list:
            state = move @ state
        Scrambles.append(state)
        
    dataC = []
    for answer, scramble in tqdm(pool.imap_unordered(twobytwohelper.C_2x1x1, Scrambles),
                        total=Num_scrambles):
        dataC.append(answer)
        
    dbfile = open("Data/Other_puzzle/Classical_2x1x1_0", 'wb')
         
    pickle.dump(dataC, dbfile) 
    dbfile.close()
    
    dataQ = []
    for answer, scramble in tqdm(pool.imap_unordered(twobytwohelper.Q_2x1x1, Scrambles),
                        total=Num_scrambles):
        dataQ.append(answer)
    
    dbfile = open("Data/Other_puzzle/Quantum_2x1x1_0", 'wb')
         
    pickle.dump(dataQ, dbfile) 
    dbfile.close()
    
    ## Generate a bunch of scrambles
    Num_scrambles = 200
    min_length = 200
    max_length = 500
    Scrambles = []
    for i in range(Num_scrambles):
        l = random.randint(min_length,max_length)
        move_list = random.choices(QS,k=l)
        state = basis(Nmax,0) # choose the 0 state as solved
        for move in move_list:
            state = move @ state
        Scrambles.append(state)
        
    dataC = []
    for answer, scramble in tqdm(pool.imap_unordered(twobytwohelper.C_2x1x1, Scrambles),
                        total=Num_scrambles):
        dataC.append(answer)
        
    dbfile = open("Data/Other_puzzle/Classical_2x1x1_1", 'wb')
         
    pickle.dump(dataC, dbfile) 
    dbfile.close()
    
    dataQ = []
    for answer, scramble in tqdm(pool.imap_unordered(twobytwohelper.Q_2x1x1, Scrambles),
                        total=Num_scrambles):
        dataQ.append(answer)
    
    dbfile = open("Data/Other_puzzle/Quantum_2x1x1_1", 'wb')
         
    pickle.dump(dataQ, dbfile) 
    dbfile.close()
    
    ## Generate a bunch of scrambles
    Num_scrambles = 200
    min_length = 200
    max_length = 500
    Scrambles = []
    for i in range(Num_scrambles):
        l = random.randint(min_length,max_length)
        move_list = random.choices(QS,k=l)
        state = basis(Nmax,0) # choose the 0 state as solved
        for move in move_list:
            state = move @ state
        Scrambles.append(state)
        
    dataC = []
    for answer, scramble in tqdm(pool.imap_unordered(twobytwohelper.C_2x1x1, Scrambles),
                        total=Num_scrambles):
        dataC.append(answer)
        
    dbfile = open("Data/Other_puzzle/Classical_2x1x1_2", 'wb')
         
    pickle.dump(dataC, dbfile) 
    dbfile.close()
    
    dataQ = []
    for answer, scramble in tqdm(pool.imap_unordered(twobytwohelper.Q_2x1x1, Scrambles),
                        total=Num_scrambles):
        dataQ.append(answer)
    
    dbfile = open("Data/Other_puzzle/Quantum_2x1x1_2", 'wb')
         
    pickle.dump(dataQ, dbfile) 
    dbfile.close()
    return