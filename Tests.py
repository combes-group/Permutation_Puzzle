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

## Size of qubit clifford is 24 ##
assert (Uc.check_finite(S_c,100,10,verbose=False)[1]==24)

## Size of qubit Pauli is 4 ##
assert (Uc.check_finite(S_p,1000,10,verbose=False)[1]==4)

## Qubit clifford + T is infinite ##
assert (not Uc.check_finite(S_t,100,10,verbose=False)[0])

## Qubit Clifford is not universal ##
assert (not Uc.check_universal(S_c))

## Qubit Pauli is not universal ##
assert (not Uc.check_universal(S_p))

## Qubit Clifford +T is universal ##
assert (Uc.check_universal(S_t))


## Dimension 3 ##
dim = 3
S_c = [Gate_helper.S_gate(dim),Gate_helper.H_gate(dim),Gate_helper.Z_gate(dim)]
S_p = [Gate_helper.X_gate(dim),Gate_helper.Y_gate(dim),Gate_helper.Z_gate(dim)]

## check size of qutrit cliffords is 216
assert (Uc.check_finite(S_c,100,20,verbose=False)[1]==dim**3*(dim**2-1))

## Size of qutrit Pauli is dim^2 ##
assert (Uc.check_finite(S_p,1000,10,verbose=False)[1]==dim**2)

## Qutrit Clifford is not universal ##
assert (not Uc.check_universal(S_c))

## Qutrit Pauli is not universal ##
assert (not Uc.check_universal(S_p))
 

## Dimension 4 ##
dim = 4
S_c = [Gate_helper.S_gate(dim),Gate_helper.H_gate(dim),Gate_helper.Z_gate(dim)]
S_p = [Gate_helper.X_gate(dim),Gate_helper.Y_gate(dim),Gate_helper.Z_gate(dim)]

## check size of qutrit cliffords is 768
assert (Uc.check_finite(S_c,100,20,verbose=False)[1]==768)

## Size of qutrit Pauli is dim^2 ##
assert (Uc.check_finite(S_p,1000,10,verbose=False)[1]==dim**2)

## Qutrit Clifford is not universal ##
assert (not Uc.check_universal(S_c))

## Qutrit Pauli is not universal ##
assert (not Uc.check_universal(S_p))
 

print("All tests passed")