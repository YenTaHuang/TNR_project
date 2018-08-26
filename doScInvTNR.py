import numpy as np
import time

def TensorUpdateSVD(win,leftnum):
    ws = win.shape
    U,S,Vh = np.linalg.svd(ct(win.reshape(int(np.prod(ws[0:leftnum])),int(np.prod(ws[leftnum:len(ws)])))),full_matrices=False)
    return ct(U@Vh).reshape(ws)

def eigCut(rho, chimax = 100000, dtol = 1e-10):
    w, v = np.linalg.eigh(rho)
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]
    chitemp = min(sum(w>dtol),chimax)
    return w[:chitemp],v[:,:chitemp]


def ct(A):
    return A.conj().T

###################################################################

def  doScInvTNR(A,allchi,Cold,qold,sold,uold,yold,vold,wold, dtol = 1e-10, disiter = 2000, miniter = 100, dispon = True, convtol = 0.01, mixratio = 10, midsteps = 30, sctype = 0):
    # function [Aout,C,q,s,u,y,v,w,Anorm,SPerrs] = doScInvTNR(A,allchi,OPTS,Cold,qold,sold,uold,yold,vold,wold)
    # ------------------------
    # modified from Glen Evenbly, 2018.
    #
    # Implementation of TNR using implicit disentangling. Input 'A' is a four
    # index tensor that defines the (square-lattice) partition function while
    # 'allchi = [chiM,chiS,chiU,chiH,chiV]' are the bond dimensions. Can
    # achieve a manifestly scale-invariant RG flow by reusing the old tensors
    # from the previous RG step as the starting point for the current RG step
    #
    # Optional arguments:
    # dtol: eigenvalue threshold for automatic truncate indication of indices  [float | [1e-10]]
    # disiter: maximum number of iterations in disentangler optimization [integer | [2000]]
    # miniter: minimum number of iterations in disentangler optimization [integer | [100]]
    # dispon: display information during optimization [0 | [1]]
    # convtol: threshold for relative error change to stop disentangler optimization [float | [1e-2]]
    # mixratio: ratio of previous level tensor to be added to environment [float,[10]]
    # midsteps: max number of iterations for convergence of 'q' ,'v' and 'w' isometries [integer,[30]]
    # sctype: sctype = 0 for no scale invariance, sctype = 1 to fix output
    #   gauge, sctype = 2 to fix gauge and reuse previous tensors as starting point

    chiHI = A.shape[0]
    chiVI = A.shape[1]
    chiM = min(allchi[0],chiHI*chiVI)
    chiU = min(allchi[2],chiVI)
    chiH = min(allchi[3],chiHI**2)
    chiV = min(allchi[4],chiU**2)

    ##### determine 'q' isometry
    qenv = np.einsum(A,[21,22,11,12],A,[7,8,11,9],A,[5,12,1,2],A,[5,9,3,4],A,[23,24,13,14],A,[7,8,13,10],A,[6,14,1,2],A,[6,10,3,4],order='C').reshape((chiHI*chiVI,chiHI*chiVI))
    SP1exact = np.trace(qenv)

    if sctype ==2:
        nmC = 1
        q = qold
        Cold = Cold/np.linalg.norm(Cold)
        for k in range(midsteps):
            qA = np.einsum(q,[1,2,23],A,[1,2,22,21],order='C')
            C = np.einsum(qA,[1,3,21],qA,[2,3,22],qA,[1,4,23],qA,[2,4,24],order='C')
            nmC = np.linalg.norm(C)
            C = C/nmC
            Cmix = C + mixratio*Cold
            qenv = np.einsum(A,[21,22,7,6],qA,[1,7,2],qA,[1,4,3],qA,[6,4,5],Cmix,[23,2,3,5],order='C')
            q = TensorUpdateSVD(qenv,2)
        SP1err = np.abs((SP1exact-nmC**2)/SP1exact) + 1e-16
    else:
        dtemp, qtemp = eigCut(qenv,chimax = chiM, dtol = dtol)
        q = qtemp.reshape(chiHI,chiVI,qtemp.shape[1])
        SP1err = abs((SP1exact - np.trace(ct(qtemp) @ qenv @ qtemp))/SP1exact) + 1e-16

    chiM = q.shape[2]
    chiS = min(allchi[1],chiM)
    qA = np.einsum(q,[1,2,23],A,[1,2,22,21],order='C')
    C = np.einsum(qA,[1,3,21],qA,[2,3,22],qA,[1,4,23],qA,[2,4,24],order='C')

    ###### iteration to determine 's' matrix, 'y' isometry, 'u' disentangler
    if sctype == 2:
        u = uold
        y = yold
        s = sold
    else:
        u = np.kron(np.eye(chiVI,chiU),np.eye(chiVI,chiU)).reshape(chiVI,chiVI,chiU,chiU)
        y = q[:,:u.shape[2],:chiS];
        s = np.eye(q.shape[2],chiS);

    Cdub = np.einsum(C,[21,22,1,2],C,[23,24,1,2],order='C')
    sCenvD = np.moveaxis(Cdub,[0,1,2,3],[0,2,1,3])
    SP2exact = np.einsum(C,[1,2,3,4],C,[1,2,3,4],order='C')
    SP2err = 1

    for k in range(disiter):
        sCenvS = np.einsum(Cdub,[21,23,7,8],q,[1,3,7],q,[4,6,8],u,[3,6,2,5],y,[1,2,22],y,[4,5,24],order='C')
        senvS = np.einsum(sCenvS,[21,22,1,2],s,[1,2],order='C')
        senvD = np.einsum(sCenvD,[21,22,1,2],s@ct(s),[1,2],order='C')

        if k%100 == 0:
            SP2errnew = np.abs(1 - ((np.trace(senvS@ct(s)))**2)/(np.trace(ct(s)@senvD@s)*SP2exact)) + 1e-16;
            if k > 50:
                errdelta = np.abs(SP2errnew-SP2err)/np.abs(SP2errnew)
                if (errdelta < convtol) or (np.abs(SP2errnew) < 1e-10):
                    SP2err = SP2errnew
                    if dispon:
                        print("Iteration: ",k," of ",disiter,", Trunc. Error: %.6g,%.6g" %(SP1err,SP2err))
                    break
            SP2err = SP2errnew
            if dispon:
                print("Iteration: ",k," of ",disiter,", Trunc. Error: %.6g,%.6g" %(SP1err,SP2err))

        #     stemp = senvD\senvS;
        stemp = np.linalg.pinv(senvD/np.trace(senvD),rcond=dtol)@senvS
        stemp = stemp/np.linalg.norm(stemp[:])
        Serrold = np.abs(1 - (np.trace(senvS@ct(s))**2)/(np.trace(ct(s)@senvD@s)*SP2exact)) + 1e-16
        for p in range(10):
            snew = (1 - 0.1*p)*stemp + 0.1*p*s;
            Serrnew = np.abs(1 - (np.einsum(sCenvS,[1,2,3,4],snew,[1,2],snew,[3,4],order='C')**2)/(np.einsum(sCenvD,[1,2,3,4],snew@ct(snew),[1,2],snew@ct(snew),[3,4],order='C')*SP2exact))+ 1e-16

            if Serrnew <= Serrold:
                s = snew/np.linalg.norm(snew[:]);
                break

        if k > 50:
            yenv = np.einsum(C,[10,6,3,4],q,[21,11,10],q,[5,8,6],u,[11,8,22,9],y,[5,9,7],s,[1,23],s,[2,7],C,[1,2,3,4],order='C')
            y = TensorUpdateSVD(yenv,2);
            uenv = np.einsum(C,[6,9,3,4],q,[5,21,6],q,[8,22,9],y,[5,23,7],y,[8,24,10],s,[1,7],s,[2,10],C,[1,2,3,4],order='C')
            uenv = uenv + np.moveaxis(uenv,[0,1,2,3],[1,0,3,2])
            u = TensorUpdateSVD(uenv,2)

    Cmod = np.einsum(C,[1,2,3,4],s,[1,21],s,[2,22],s,[3,23],s,[4,24],order='C')
    Cnorm = np.einsum(Cmod,[1,2,3,4],Cmod,[1,2,3,4])/np.einsum(C,[1,2,3,4],C,[1,2,3,4],order='C')
    s = s/(Cnorm**(1/8))

    ###### determine 'v' and 'w' isometries
    venv = np.einsum(y,[1,3,17],y,[1,4,24],y,[2,3,18],y,[2,4,29],s,[5,17],qA,[7,11,5],qA,[7,12,6],s,[6,19],s,[8,18],qA,[10,11,8],qA,[10,12,9],s,[9,20],y,[13,15,19],y,[13,16,25],y,[14,15,20],y,[14,16,30],s,[21,24],qA,[23,31,21],qA,[23,32,22],s,[22,25],s,[26,29],qA,[28,33,26],qA,[28,34,27],s,[27,30],order='C')
    SP3exact = np.einsum(venv,[1,2,1,2],order='C')

    if sctype > 0:
        Aold = A/np.linalg.norm(A)

        if sctype == 2:
            v = vold
            w = wold
        else:
            w = TensorUpdateSVD(np.random.rand(u.shape[2],u.shape[2],A.shape[1]),2)
            for k in range(50):
                wenv = np.einsum(y,[9,31,10],y,[9,32,11],s,[1,10],qA,[5,7,1],qA,[5,8,2],s,[2,13],w,[16,15,18],y,[12,15,13],y,[12,16,14],s,[3,11],qA,[6,7,3],qA,[6,8,4],s,[4,14],Aold,[17,18,17,33],order='C')
                w = TensorUpdateSVD(wenv,2)

            v = TensorUpdateSVD(np.random.rand(A.shape[2],A.shape[2],A.shape[2]),2)
            for k in range(50):
                venv = np.einsum(y,[1,3,10],y,[1,4,15],v,[9,8,18],s,[5,10],qA,[7,8,5],qA,[7,9,6],s,[6,11],y,[2,3,11],y,[2,4,16],s,[12,15],qA,[14,31,12],qA,[14,32,13],s,[13,16],Aold,[33,17,18,17],order='C')
                v = TensorUpdateSVD(venv,2)

        for k in range(2*midsteps):
            Atemp = np.einsum(v,[10,9,21],s,[7,19],qA,[6,9,7],qA,[6,10,8],s,[8,14],w,[17,18,22],y,[16,17,19],y,[16,18,20],v,[4,5,23],s,[1,20],qA,[3,4,1],qA,[3,5,2],s,[2,15],w,[13,12,24],y,[11,12,14],y,[11,13,15],order='C')
            Amix = Atemp/np.linalg.norm(Atemp) + mixratio*Aold

            if k%2 == 1:
                venv = np.einsum(w,[2,3,11],y,[1,2,9],y,[1,3,20],v,[8,7,10],s,[4,9],qA,[6,7,4],qA,[6,8,5],s,[5,15],w,[14,13,16],y,[12,13,15],y,[12,14,21],s,[17,20],qA,[19,22,17],qA,[19,23,18],s,[18,21],Amix,[10,11,24,16],order='C')
                v = TensorUpdateSVD(venv,2)
            else:
                wenv = np.einsum(y,[19,23,20],y,[19,24,21],v,[5,4,10],s,[1,20],qA,[3,4,1],
                    qA,[3,5,2],s,[2,9],w,[8,7,11],y,[6,7,9],y,[6,8,17],v,[15,16,18],s,[12,21],
                    qA,[14,15,12],qA,[14,16,13],s,[13,17],Amix,[10,25,18,11],order='C')
                w = TensorUpdateSVD(wenv,2)
    else:
        venv = 0.5*((venv+venv.transpose(1,0,3,2)).reshape(chiHI**2,chiHI**2))
        dtemp, vtemp = eigCut(venv, chimax = chiH, dtol = dtol)
        v = vtemp.reshape(chiHI,chiHI,vtemp.shape[1])
        wenv = np.einsum(y,[25,31,26],y,[25,32,27],y,[28,33,29],y,[28,34,30],s,[1,26],
            qA,[3,7,1],qA,[3,8,2],s,[2,13],s,[4,29],qA,[6,7,4],qA,[6,8,5],s,[5,14],
            y,[9,11,13],y,[9,12,23],y,[10,11,14],y,[10,12,24],s,[15,27],qA,[17,21,15],qA,[17,22,16],s,[16,23],s,[18,30],qA,[20,21,18],qA,[20,22,19],s,[19,24],order='C')
        wenv = 0.5*(wenv + wenv.transpose(1,0,3,2)).reshape(chiU**2,chiU**2)
        dtemp, wtemp = eigCut(wenv, chimax = chiV, dtol = dtol)
        w = wtemp.reshape(chiU,chiU,wtemp.shape[1])

    Atemp = np.einsum(v,[10,9,21],s,[7,19],qA,[6,9,7],qA,[6,10,8],s,[8,14],w,[17,18,22],y,[16,17,19],y,[16,18,20],v,[4,5,23],s,[1,20],qA,[3,4,1],qA,[3,5,2],s,[2,15],w,[13,12,24],y,[11,12,14],y,[11,13,15],order='C')
    Atemp = 0.5*(Atemp + Atemp.transpose(2,3,0,1))
    SP3err = np.abs((SP3exact - np.sum(Atemp**2))/SP3exact) + 1e-16

    # generate new 'A' tensor
    Anorm = np.linalg.norm(Atemp)
    Aout = Atemp/Anorm
    SPerrs = [SP1err,SP2err,SP3err]

    return Aout, C, q, s, u, y, v, w, Anorm, SPerrs