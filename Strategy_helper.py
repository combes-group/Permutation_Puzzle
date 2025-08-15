import math
import numpy as np
import pandas as pd
## Helper functions for computing probabilities ##

def GeoMean(state,m):
    """
    For a given state that is m moves away from the scrambled state,
    computes the expected number of moves if the scramble is solved through this state.
    :param state: numpy array of representing a state of the puzzle
    :param m: number of moves between state and scramble
    
    :returns: expected move count
    """
    P = np.abs(state[0])**2
    return m/P

def conditionalGeo(n,M,P):
    """
    :param n: integer move count
    :param M: integer depth of optimal state
    :param P: probability of sucess with optimal state 
    
    :returns: G(n|M,P) probability of solving in  exactly n moves given M,P 
    """
    if (n/M == n//M):
        return (1-P)**(n//M-1)*P
    return 0

def Prob(n,Mvals,Pvals,Z):
    """
    :param n: integer move count
    :param Mvals: numpy array of values of M to be considered
    :param Pvals: numpy array 
    :param Z: numpy array containing the probilites of a scramble to have optimal 
              parameters M,P evaluated on the grid of Pvals and Mvals 
    
    :returns: probability of solving in  exactly n moves G(n)
    """
    psum = 0
    for i,m in enumerate(Mvals):
        for j,p in enumerate(Pvals):
            psum += conditionalGeo(n,m,p) * Z[j,i]
    return psum

## helper functions for enumerating reachable states ##

def canonical_phase(state):
    """
    phase choosen so that the first nonzero element is positive
    :param state: numpy array
    
    :returns: numpy array
    """
    for coeff in state:
        if not math.isclose(np.abs(coeff),0):
            phase = np.angle(coeff)
            break
    return state / np.exp(1j*phase)

def hash_func(state):
    """
    :param state: numpy array
    
    :returns: string hash of a given state used for comparison
    """
    string = ''
    temp = np.round(canonical_phase(state),8)
    for coeff in temp:
        if np.real(coeff) == 0 :
            string += '[0.]'
        else:
            string +=   str(np.real(coeff))
        if np.imag(coeff) == 0:
            string += '[0.]'
        else:
            string += str(np.imag(coeff))
        
    return string

## Finding the optimal solution to a given scramble ##

def optimal_search(scramble,move_set,Cost=[1,1],max_depth = 100):
    """
    :param scramble: initial state of a scramble
    :param move_set: list of allowed actions to apply to the scramble
    :param Cost: a list of costs, Cost[0] is the cost of moving and Cost[1] is the cost of measureing
    :param max_depth: a finite depth to terminate the search for an optimal state after
    
    :returns best_depth: the depth parameter, M, of the optimal state
    :returns best_prob: the probabilty parameter, P, of the optimal state
    """
    move_cost = Cost[0]
    meas_cost = Cost[1]
    group = {hash_func(scramble):scramble}
    new_elems = group
    depth = 0
    best  = GeoMean(scramble,depth+meas_cost)
    best_depth = depth+meas_cost
    best_prob = 1/best*best_depth
    while((depth+meas_cost)<min(best,max_depth)):
        current_group = new_elems.copy()
        new_elems = {}
        for g in move_set:
            # iterate through the group
            for key in current_group: 
                state = g@current_group[key]
                hashed = hash_func(state)
                if not (hashed in group):
                    new_elems.update({hashed:state}) 
                    group.update({hashed:state})
                    if(GeoMean(state,depth+move_cost+meas_cost)< best):
                        best = GeoMean(state,depth+move_cost+meas_cost)
                        best_depth = depth+move_cost+meas_cost
                        best_prob = 1/best*best_depth
        depth += move_cost
    return [best_depth,best_prob]

def basis(Nmax,n):
    vec = np.zeros((Nmax,1))
    vec[n] = 1
    return np.array(vec)
def SWAP(Nmax,n):
    assert(n<Nmax-1)
    # -1 from fermionic exchange
    mat = -1*np.eye(Nmax)
    mat[n,n] = 0
    mat[n+1,n+1] = 0
    mat[n,n+1] = 1
    mat[n+1,n] = 1
    return mat

def Q_opt_search(scramble):
    Costs = [1,1]
    maxdepth = 15
    Nmax = 6
    CS = [SWAP(Nmax,n) for n in range(Nmax-1)]
    QS = []
    MS = []
    for move in CS:
        QS.append((np.eye(Nmax) + 1j*move)/np.sqrt(2))
        QS.append((np.eye(Nmax) - 1j*move)/np.sqrt(2))
    return [optimal_search(scramble,QS,Costs,maxdepth),scramble]

def C_opt_search(scramble):
    Costs = [1,1]
    maxdepth = 15
    Nmax = 6
    CS = [SWAP(Nmax,n) for n in range(Nmax-1)]
    return [optimal_search(scramble,CS,Costs,maxdepth),scramble]

def Sample_scrambles_opt(Scrambles,move_set,Cost,max_depth=100,Verbose=True):
    """
    :param scrambles: list of scrambled states
    :param move_set: list of allowed actions to apply to the scramble
    :param Cost: a list of costs, Cost[0] is the cost of moving and Cost[1] is the cost of measureing
    :param max_depth: a finite depth to terminate the search for an optimal state after
    
    :returns M_samples: np array of integer depths, M, from the optimal state from each scramble
    :returns P_samples: np array of probabilites, P, from the optimal state of each scramble
    """
    P_samples = []
    M_samples = []
    for i,scramble in enumerate(Scrambles):
        [M,P] = optimal_search(scramble,move_set,Cost,max_depth)
        P_samples.append(P)
        M_samples.append(M)
        if(Verbose and (i/len(Scrambles)%.1< (i-1)/len(Scrambles)%.1)):
            print("Progress:", i/len(Scrambles))
    return [np.array(M_samples),np.array(P_samples)]


def scramble_probabilities(data,depth,Costs=[1,1],Mvals = None,Pvals=None):
    """
    :param data: 2D numpy array of optimal values of M,P from many scrambles
    :param depth: integer largest M 
    :param Cost: a list of costs, Cost[0] is the cost of moving and Cost[1] is the cost of measureing
    :param Mvals: grid of M values to evaluate probabilties on
    :param Pvals: grid of P values to evaluate probabilties on
    
    :returns Z: 2D numpy array with dimensions [len(Mvals),len(Pvals)]
    :returns Mvals: np array of grid used to calculate probabilties
    :returns Pvals: np array of grid used to calculate probabilties
    """
    df = pd.DataFrame({'M': data[0],'P':data[1]})
    Num_scrambles = len(data[0])
    # Discretizing the sample space
    if((Mvals is None)):
        Mvals = np.linspace(Costs[1],depth*Costs[0]+Costs[1],depth*Costs[0]+1)
    if((Pvals is None)):
        #choosing number of bins according to Terrel-Scott Rule
        N = int((2*Num_scrambles)**(1/3))
        Pvals = np.linspace(0,1,N)
    
    df = df.sort_values(by=['M','P'])
    counts = np.zeros([len(Pvals),depth+1])
    
    # Marginalizing over scramble parameters 
    for j,M in enumerate(Mvals):
        marg = df.loc[df['M']==M].get('P')
        for i,p in enumerate(Pvals):
            assert(p<=1)
            for data_point in marg:
                if (data_point > p and data_point<=Pvals[i+1]):
                    counts[i,j]+= 1
                
    # Normalizing counts to probabilities
    Z = counts/Num_scrambles
    return (Z,Mvals,Pvals)

def scramble_expected_dist(data):
    """
    :param data: 2D numpy array of optimal values of M,P from many scrambles
    
    :returns Z: 2D numpy array with dimensions len(E_vals)
    :returns E_vals: array of bins used
    """
    expected = data[0]/data[1]
    Num_scrambles = len(data[0])
    std = np.std(expected)
    N = int(2*(Num_scrambles)**(1/3))
    E_vals = np.linspace(1,11,N-1)
    E_vals = np.insert(E_vals,0,0)
    counts = np.zeros(N)
    for i in range(N-1):
        for datapoint in expected:
            if (datapoint>E_vals[i] and datapoint<= E_vals[i+1]):
                counts[i] += 1
    return (counts/Num_scrambles,E_vals)