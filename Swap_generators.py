import numpy as np

zero  = np.array([[1], [0], [0], [0], [0], [0]],dtype=complex)
one   = np.array([[0], [1], [0], [0], [0], [0]],dtype=complex)
two   = np.array([[0], [0], [1], [0], [0], [0]],dtype=complex)
three = np.array([[0], [0], [0], [1], [0], [0]],dtype=complex)
four  = np.array([[0], [0], [0], [0], [1], [0]],dtype=complex)
five  = np.array([[0], [0], [0], [0], [0], [1]],dtype=complex)


def swaps(exchange_stat_1,exchange_stat_2,diagonals=False):
    """
    :param exchange_stat_1: phase from swaping the first species of particle
    :param exchange_stat_2: phase from swaping the second species of particle
    :diagonals: boolean, True if you want diagonal swaps
    
    returns: list of numpy matricies
    """ 
    
    if np.abs(exchange_stat_1)**2 != 1 or np.abs(exchange_stat_2)**2 != 1:
          raise Exception("exchange statistics must have magnitude 1")
    
    SN = exchange_stat_1*zero@zero.T + exchange_stat_2*one@one.T + five@two.T \
         +four@three.T+three@four.T+ two@five.T

    SS =  exchange_stat_2*zero@zero.T +exchange_stat_1*one@one.T + five@three.T \
         +four@two.T+two@four.T+ three@five.T

    SE =  exchange_stat_1*two@two.T + exchange_stat_2*three@three.T + five@one.T \
         +four@zero.T+zero@four.T+ one@five.T

    SW = exchange_stat_2*two@two.T + exchange_stat_1*three@three.T + five@zero.T \
         +four@one.T+one@four.T+ zero@five.T

    D1 = zero@three.T + three@zero.T +one@two.T + two@one.T \
         +exchange_stat_1*four@four.T + exchange_stat_2*five@five.T

    D2 = zero@two.T + two@zero.T +one@three.T + three@one.T \
         +exchange_stat_2*four@four.T + exchange_stat_1*five@five.T
    
    if diagonals:
        return[SN,SS,SE,SW,D1,D2]
    return [SN,SS,SE,SW]

def sqrt_swaps(exchange_stat_1,exchange_stat_2,diagonals=False):
    """
    :param exchange_stat_1: phase from swaping the first species of particle
    :param exchange_stat_2: phase from swaping the second species of particle
    :diagonals: boolean, True if you want diagonal swaps
    
    returns: list of numpy matricies
    """ 
    Id = np.eye(6)
    Perms = swaps(exchange_stat_1,exchange_stat_2,diagonals)
    return [(Id + 1j*P)/np.sqrt(2) for P in Perms]

def general_swaps(exchange_stat_1,exchange_stat_2,theta,diagonals=False):
    """
    :param exchange_stat_1: phase from swaping the first species of particle
    :param exchange_stat_2: phase from swaping the second species of particle
    :diagonals: boolean, True if you want diagonal swaps
    
    returns: list of numpy matricies
    """ 
    Id = np.eye(6)
    Perms = swaps(exchange_stat_1,exchange_stat_2,diagonals)
    return [np.cos(theta/2)*Id + np.sin(theta/2)*1j*P for P in Perms]