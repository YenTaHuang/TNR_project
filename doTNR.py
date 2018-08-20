#!/usr/bin/env python3

# translated from Glen Evenbly's Julia code

import numpy as np
import time




def conj(x):
    return np.conjugate(x)

# define conjugate transpose
def ct(A):
    return np.conjugate(np.transpose(A))

def eigCut(rho, chimax = 100000, dtol = 1e-10):
    w, v = np.linalg.eigh(rho)
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]
    chitemp = min(sum(w>dtol),chimax)
    return w[:chitemp],v[:,:chitemp]

def TensorUpdateSVD(win,leftnum):
    ws = win.shape
    U,S,Vh = np.linalg.svd(ct(win.reshape(int(np.prod(ws[0:leftnum])),int(np.prod(ws[leftnum:len(ws)])))),full_matrices=False)
    return ct(U@Vh).reshape(ws)




#############################################################################
def doTNR(A, allchi, dtol = 1e-10, disiter = 2000, miniter = 100, dispon = True, convtol = 0.01):
    # Implementation of TNR using implicit disentangling. Input 'A' is a four
    # index tensor that defines the (square-lattice) partition function while
    # 'allchi = [chiM,chiS,chiU,chiH,chiV]' are the bond dimensions.
    #
    # Optional input parameters:
    # dtol: eigenvalue threshold for automatic truncate indication of indices  [float | [1e-10]]
    # disiter: maximum number of iterations in disentangler optimization [integer | [2000]]
    # miniter: minimum number of iterations in disentangler optimization [integer | [100]]
    # dispon: display information during optimization [false | [true]]
    # convtol: threshold for relative error change to stop disentangler optimization [float | [1e-2]]


    chiHI = A.shape[0]
    chiVI = A.shape[1]
    chiM = min(allchi[0],chiHI*chiVI)
    chiS = allchi[1]
    chiU = min(allchi[2],chiVI)
    chiH = min(allchi[3],chiHI**2)
    chiV = min(allchi[4],chiU**2)
    ##### determine 'q' isometry
    print("shape of A: ",A.shape)
    qenv = np.einsum(A,[-4,-3,11,12],A,[7,8,11,9],A,[5,12,1,2],A,[5,9,3,4],A,[-2,-1,13,14],A,[7,8,13,10],A,[6,14,1,2],A,[6,10,3,4],optimize=True).reshape((chiHI*chiVI,chiHI*chiVI))
    dtemp, qtemp = eigCut(qenv, chimax = chiM, dtol = dtol);
    q = qtemp.reshape((chiHI,chiVI,qtemp.shape[1]))
    chiM = q.shape[2]
    chiS = min(chiS,chiM)

    SP1exact = np.trace(qenv)
    SP1err = np.abs((SP1exact - np.trace(ct(qtemp)@qenv@qtemp))/SP1exact) + 1e-16
    #print("SP1err : ",SP1err)

    qA = np.einsum(q,[1,2,-1],A,[1,2,-2,-3],order='C',optimize=True)
    C = np.einsum(qA,[1,3,-4],qA,[2,3,-3],qA,[1,4,-2],qA,[2,4,-1],order='C',optimize=True)
    ###### iteration to determine 's' matrix, 'y' isometry, 'u' disentangler
    u = np.kron(np.eye(chiVI,chiU),np.eye(chiVI,chiU)).reshape(chiVI,chiVI,chiU,chiU)
    y = q[:,:u.shape[2],:chiS]
    s = np.eye(q.shape[2],chiS)

    Cdub = np.einsum(C,[-4,-3,1,2],C,[-2,-1,1,2],order='C',optimize=True)
    sCenvD = np.moveaxis(Cdub,[0,1,2,3],[0,2,1,3])
    SP2exact = np.einsum(C,[1,2,3,4],C,[1,2,3,4],order='C',optimize=True)
    SP2err = 1

    for k in range(disiter):
        sCenvS = np.einsum(Cdub,[-4,-2,7,8],q,[1,3,7],q,[4,6,8],u,[3,6,2,5],y,[1,2,-3],y,[4,5,-1],order='C',optimize=True)
        senvS = np.einsum(sCenvS,[-2,-1,1,2],s,[1,2],order='C',optimize=True)
        senvD = np.einsum(sCenvD,[-2,-1,1,2],s@ct(s),[1,2],order='C',optimize=True)

        if k%100 == 0:
            SP2errnew = np.abs(1 - ((np.trace(senvS@ct(s)))**2)/(np.trace(ct(s)@senvD@s)*SP2exact)) + 1e-16;
            if k > 50:
                errdelta = np.abs(SP2errnew-SP2err)/np.abs(SP2errnew)
                if (errdelta < convtol) or (np.abs(SP2errnew) < 1e-10):
                    SP2err = SP2errnew
                    if dispon:
                        print("Iteration: ",k," of ",disiter,", Trunc. Error: ",SP1err,", ",SP2err,"\n")
                    break
            SP2err = SP2errnew
            if dispon:
                print("Iteration: ",k," of ",disiter,", Trunc. Error: ",SP1err,", ",SP2err,"\n")

        #     stemp = senvD\senvS;
        stemp = np.linalg.pinv(senvD/np.trace(senvD),rcond=dtol)@senvS
        stemp = stemp/np.linalg.norm(stemp[:])

        Serrold = np.abs(1 - (np.trace(senvS@ct(s))**2)/(np.trace(ct(s)@senvD@s)*SP2exact)) + 1e-16
        for p in range(10):
            snew = (1 - 0.1*p)*stemp + 0.1*p*s;
            Serrnew = np.abs(1 - (np.einsum(sCenvS,[1,2,3,4],snew,[1,2],snew,[3,4],order='C',optimize=True)**2)/(np.einsum(sCenvD,[1,2,3,4],snew@ct(snew),[1,2],snew@ct(snew),[3,4],order='C',optimize=True)*SP2exact))+ 1e-16

            if Serrnew <= Serrold:
                s = snew/np.linalg.norm(snew[:]);
                break

        if k > 50:
            yenv = np.einsum(C,[10,6,3,4],q,[-3,11,10],q,[5,8,6],u,[11,8,-2,9],y,[5,9,7],s,[1,-1],s,[2,7],C,[1,2,3,4],order='C',optimize=True)
            y = TensorUpdateSVD(yenv,2);
            uenv = np.einsum(C,[6,9,3,4],q,[5,-4,6],q,[8,-3,9],y,[5,-2,7],y,[8,-1,10],s,[1,7],s,[2,10],C,[1,2,3,4],order='C',optimize=True)
            uenv = uenv + np.moveaxis(uenv,[0,1,2,3],[1,0,3,2])
            u = TensorUpdateSVD(uenv,2)

    Cmod = np.einsum(C,[1,2,3,4],s,[1,-4],s,[2,-3],s,[3,-2],s,[4,-1],order='C',optimize=True)
    Cnorm = np.einsum(Cmod,[1,2,3,4],Cmod,[1,2,3,4],optimize=True)/np.einsum(C,[1,2,3,4],C,[1,2,3,4],order='C',optimize=True)
    s = s/(Cnorm**(1/8))
    #print("SP2err : ",SP2err)

    ###### determine 'v' isometry
    venv = np.einsum(y,[1,3,17],y,[1,4,24],y,[2,3,18],y,[2,4,29],s,[5,17],qA,[7,11,5],qA,[7,12,6],s,[6,19],s,[8,18],qA,[10,11,8],qA,[10,12,9],s,[9,20],y,[13,15,19],y,[13,16,25],y,[14,15,20],y,[14,16,30],s,[21,24],qA,[23,-4,21],qA,[23,-3,22],s,[22,25],s,[26,29],qA,[28,-2,26],qA,[28,-1,27],s,[27,30],order='C',optimize=True)
    venv = 0.5*(venv + np.moveaxis(venv,[0,1,2,3],[1,0,3,2])).reshape(chiHI**2,chiHI**2);
    dtemp, vtemp = eigCut(venv, chimax = chiH, dtol = dtol)
    v = vtemp.reshape(chiHI,chiHI,vtemp.shape[1])

    SP3exact = np.trace(venv)
    SP3err = np.abs((SP3exact - np.trace(ct(vtemp)@venv@vtemp))/SP3exact) + 1e-16
    #print("SP3err : ",SP3err)

    ###### determine 'w' isometry

    #print("step 4 contraction starts: ...")
    wenv = np.einsum(y,[25,-4,26],y,[25,-3,27],y,[28,-2,29],y,[28,-1,30],s,[1,26],qA,[3,7,1],qA,[3,8,2],s,[2,13],s,[4,29],qA,[6,7,4],qA,[6,8,5],s,[5,14],y,[9,11,13],y,[9,12,23],y,[10,11,14],y,[10,12,24],s,[15,27],qA,[17,21,15],qA,[17,22,16],s,[16,23],s,[18,30],qA,[20,21,18],qA,[20,22,19],s,[19,24],order='C',optimize=True)

    #print("step 4 contraction ends!")
    wenv = 0.5*(wenv + np.moveaxis(wenv,[0,1,2,3],[1,0,3,2])).reshape(chiU**2,chiU**2)
    dtemp, wtemp = eigCut(wenv, chimax = chiV, dtol = dtol)
    w = wtemp.reshape(chiU,chiU,wtemp.shape[1])

    SP4exact = np.trace(wenv)
    SP4err = np.abs((SP4exact - np.trace(ct(wtemp)@wenv@wtemp))/SP4exact) + 1e-16;
    #print("SP4err : ",SP4err)

    ###### generate new 'A' tensor
    Atemp = np.einsum(v,[10,9,-4],s,[7,19],qA,[6,9,7],qA,[6,10,8],s,[8,14],w,[17,18,-3],y,[16,17,19],y,[16,18,20],v,[4,5,-2],s,[1,20],qA,[3,4,1],qA,[3,5,2],s,[2,15],w,[13,12,-1],y,[11,12,14],y,[11,13,15],order='C',optimize=True)
    Anorm = np.linalg.norm(Atemp[:])
    Aout = Atemp/Anorm

    SPerrs = [SP1err,SP2err,SP3err,SP4err]
    return Aout, q, s, u, y, v, w, Anorm, SPerrs
