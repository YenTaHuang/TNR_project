
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator


class SimpleLinOp:
    def __init__(self, n, f):
        self.dtype = 'float'
        self.shape = (n,n)
    def matvec(self,x):
        return f(x)

n = 100000000

def func0(x):
    result = np.zeros((n))
    result[0] = 2*x[0]
    result[1] = 3*x[1]
    return result
print("starts defining A...")

A = LinearOperator((n,n),func0)


print("starts solving eigs...")

w=scipy.sparse.linalg.eigs(A, k=10, which='LM', maxiter=None, tol=1e-2, return_eigenvectors=False)

print(w)
