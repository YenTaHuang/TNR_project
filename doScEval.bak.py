import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator

def ct(A):
    return np.conjugate(np.transpose(A))

def eigCut(rho, chimax = 100000, dtol = 1e-10):
    w, v = np.linalg.eigh(0.5*(rho+ct(rho)))
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]
    chitemp = min(sum(w>dtol),chimax)
    return w[:chitemp],v[:,:chitemp]

##### Define action of scaling superoperator on a local operator




def doScEval(Ax,qx,sx,yx,vx,wx,chiK):
    # function doScEval(Ax,qx,sx,yx,vx,wx,chiK,numeval)
    # ------------------------
    # by Glen Evenbly, v1.0 (2018).
    #
    # Compute scaling dimensions from the tensors produced by the
    # (scale-invariant) version of the TNR algorithm, using the logarithmic
    # transformation. Input 'chiK' is a control parameter that can be adjusted
    # for accuracy, and 'numeval' dictates the number of scaling dimensions to
    # evaluate.

    ##### Compress vertical indices
    chitemp = Ax.shape[3]
    genv = np.einsum(Ax,[1,2,5,21],Ax,[3,4,5,22],Ax,[1,2,6,23],Ax,[3,4,6,24],order='C',optimize=True).reshape(chitemp**2,chitemp**2)
    dtemp, gtemp = eigCut(genv, chimax = chiK, dtol = 0)
    gx = gtemp.reshape(chitemp,chitemp,gtemp.shape[1])

    ##### Compress horizontal indices
    chitemp = Ax.shape[2]
    renv = np.einsum(Ax,[1,2,22,5],Ax,[4,3,21,5],Ax,[1,2,24,6],Ax,[4,3,23,6],order='C',optimize=True).reshape(chitemp**2,chitemp**2)
    dtemp, rtemp = eigCut(renv, chimax = chiK, dtol = 0)
    rx = rtemp.reshape(chitemp,chitemp,rtemp.shape[1])

    ##### Compute gates
    print("compute gates")
    Aqs = np.einsum(Ax,[22,21,1,2],qx,[1,2,3],sx,[3,23],order='C',optimize=True)
    rT = np.einsum(rx,[10,12,22],vx,[5,6,12],Aqs,[4,5,24],Aqs,[4,6,11],rx,[3,2,21],yx,[2,1,11],yx,[3,1,13],vx,[9,8,10],Aqs,[7,8,13],Aqs,[7,9,23],order='C',optimize=True)
    gT = np.einsum(gx,[12,11,22],wx,[6,5,12],yx,[4,5,23],yx,[4,6,10],
        gx,[2,3,21],Aqs,[2,1,10],Aqs,[3,1,13],wx,[8,9,11],yx,[7,8,13],yx,[7,9,24])


    chir = rT.shape[0]
    chig = gT.shape[1]
    n = (chir*chig)**2
    print("rT.shape: ",rT.shape)
    print("gT.shape: ",gT.shape)
    print("n: ",n)

    def logScaleSuper(psi):
        return np.einsum(psi.reshape(chir,chig,chir,chig),[1,3,5,7],rT,[1,21,8,2],gT,[3,22,2,4],rT,[5,23,4,6],gT,[7,24,6,8],order='C',optimize=True).reshape((chir*chig)**2)

    Atemp = LinearOperator(((chir*chig)**2,(chir*chig)**2), matvec = logScaleSuper)


    print("calculating eigs")
    w = scipy.sparse.linalg.eigsh(Atemp, k=4, which='LM', maxiter=5, tol=1e-2, return_eigenvectors=False)

    





    return -np.log2(np.abs(w/w[0]))
