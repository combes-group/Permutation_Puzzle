import numpy as np
import Gate_helper
import Universality_check as Uc

## Dimension 2 ##
#Clifford Group
S = Gate_helper.S_gate(2)
H = Gate_helper.H_gate(2)
S_c = [S,H]
## expected size of qubit clifford group is 2^3(2^2-1)= 24

#Pauli Group
X = Gate_helper.X_gate(2)
Y = Gate_helper.Y_gate(2)
Z = Gate_helper.Z_gate(2)
S_p = [X,Y,Z]

#Clifford + T 
T_gate = np.matrix(np.diag([1,np.exp(1j*np.pi/4)]))
S_t = [S,H,T_gate]

## Size of qubit Clifford group is 24 ##
assert (Uc.check_finite(S_c,100,10,verbose=False)==24)

## Size of qubit Pauli group is 4 ##
assert (Uc.check_finite(S_p,1000,10,verbose=False)==4)

## Qubit Clifford + T is infinite ##
assert (not Uc.check_finite(S_t,100,10,verbose=False))

## Qubit Clifford is not universal ##
assert (not Uc.check_universal(S_c))

## Qubit Pauli is not universal ##
assert (not Uc.check_universal(S_p))

## Qubit Clifford +T is universal ##
assert (Uc.check_center(S_t))
assert (Uc.check_universal(S_t))


## Dimension 3 ##
dim = 3
S_c = [Gate_helper.S_gate(dim),Gate_helper.H_gate(dim),Gate_helper.Z_gate(dim)]
S_p = [Gate_helper.X_gate(dim),Gate_helper.Y_gate(dim),Gate_helper.Z_gate(dim)]
S_t = [Gate_helper.S_gate(dim),Gate_helper.H_gate(dim),Gate_helper.fractional_Z_gate(dim,1/4)]

## check size of qutrit cliffords is 216
assert (Uc.check_finite(S_c,100,20,verbose=False)==dim**3*(dim**2-1))

## Size of qutrit Pauli is dim^2 ##
assert (Uc.check_finite(S_p,1000,10,verbose=False)==dim**2)

## Qutrit Clifford is not universal ##
assert (not Uc.check_universal(S_c))

## Qutrit Pauli is not universal ##
assert (not Uc.check_universal(S_p))

## Qutrit Clifford + Z/4 is  universal ##
assert ( Uc.check_universal(S_t))


## Dimension 4 ##
dim = 4
S_c = [Gate_helper.S_gate(dim),Gate_helper.H_gate(dim),Gate_helper.Z_gate(dim)]
S_p = [Gate_helper.X_gate(dim),Gate_helper.Y_gate(dim),Gate_helper.Z_gate(dim)]
S_t = [Gate_helper.S_gate(dim),Gate_helper.H_gate(dim),Gate_helper.fractional_Z_gate(dim,1/4)]

## check size of qudit cliffords is 768
assert (Uc.check_finite(S_c,100,20,verbose=False)==768)

## Size of qudit Pauli is dim^2 ##
assert (Uc.check_finite(S_p,1000,10,verbose=False)==dim**2)

## Qudit Clifford is not universal ##
assert (not Uc.check_universal(S_c))

## Qudit Pauli is not universal ##
assert (not Uc.check_universal(S_p))
 
## Qudit Clifford + Z/4 is  universal ##
assert ( Uc.check_universal(S_t))

print("All tests passed")
