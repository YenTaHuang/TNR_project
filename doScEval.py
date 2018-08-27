import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator

from doScInvTNR import eigCut

##### Define action of scaling superoperator on a local operator

def doScEval(Ax,qx,sx,yx,vx,wx,chiK,N_level):
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
    genv = np.einsum(Ax,[1,2,5,21],Ax,[3,4,5,22],Ax,[1,2,6,23],Ax,[3,4,6,24],order='C').reshape(chitemp**2,chitemp**2)
    dtemp, gtemp = eigCut(genv, chimax = chiK, dtol = 0)
    gx = gtemp.reshape(chitemp,chitemp,gtemp.shape[1])

    ##### Compress horizontal indices
    chitemp = Ax.shape[2]

    renv = np.einsum(Ax,[1,2,22,5],Ax,[4,3,21,5],Ax,[1,2,24,6],Ax,[4,3,23,6],order='C').reshape(chitemp**2,chitemp**2)
    dtemp, rtemp = eigCut(renv, chimax = chiK, dtol = 0)
    rx = rtemp.reshape(chitemp,chitemp,rtemp.shape[1])

    ##### Compute gates
    print("compute gates")
    Aqs = np.einsum(Ax,[22,21,1,2],qx,[1,2,3],sx,[3,23])
    rT = np.einsum(rx,[10,12,22],vx,[5,6,12],Aqs,[4,5,24],Aqs,[4,6,11],rx,[3,2,21],yx,[2,1,11],yx,[3,1,13],vx,[9,8,10],Aqs,[7,8,13],Aqs,[7,9,23])
    gT = np.einsum(gx,[12,11,22],wx,[6,5,12],yx,[4,5,23],yx,[4,6,10],gx,[2,3,21],Aqs,[2,1,10],Aqs,[3,1,13],wx,[8,9,11],yx,[7,8,13],yx,[7,9,24])

    chir = rT.shape[0]
    chig = gT.shape[0]
    n = (chir*chig)**2
    print("rT.shape: ",rT.shape)
    print("gT.shape: ",gT.shape)
    print("n: ",n)


    def logScaleSuper(v):
        v_temp = v.reshape(chir,chig,chir,chig)
        temp1 = np.einsum(rT,[1,2,3,7],gT,[4,5,7,6])
        temp2 = np.einsum(v_temp,[1,2,14,15],temp1,[1,11,16,2,12,13])
        return np.einsum(temp1,[2,23,1,3,24,4],temp2,[21,22,1,2,3,4]).reshape(n)
    print("defining Atemp...")
    Atemp = LinearOperator((n,n), matvec = logScaleSuper, dtype='float64')
    
    print("calculating eigs")
    w = scipy.sparse.linalg.eigs(Atemp, k=N_level, which='LM', maxiter=200, tol=1e-5, return_eigenvectors=False)
    w=w[::-1]
    spec = -np.log2(np.abs(w/w[0]))
    
    return spec
