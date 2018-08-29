import numpy as np

chi_i = 2

def boltzmann(i,j,k,l,beta):
    return np.exp(beta*((2*i-1)*(2*j-1)+(2*j-1)*(2*k-1)+(2*k-1)*(2*l-1)+(2*l-1)*(2*i-1)))

def initialize_A(beta):
    A = np.array([[[[boltzmann(i,j,k,l,beta) for l in range(chi_i)] for k in range(chi_i)] for j in range(chi_i)] for i in range(chi_i)])
    A = A/np.linalg.norm(A)
    return A

Tc = (2/np.log(1+np.sqrt(2)))
relTemp = 1
Tval  = relTemp*Tc
betaval = 1./Tval

Ainit = initialize_A(betaval)




Atemp = np.einsum(Ainit,[12,13,3,1],Ainit,[3,14,15,2],Ainit,[11,1,4,18],Ainit,[4,2,16,17]).reshape(4,4,4,4)
