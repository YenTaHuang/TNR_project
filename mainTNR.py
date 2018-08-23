#!/usr/bin/env python3



###### Your paths go here
from doTNR import *

# mainTNR
# ------------------------
# by Glen Evenbly
#
# Script file for initializing a partition function (here the classical
# Ising model) before passing to a Tensor Network Renormalization (TNR)
# coarse-graining routine. The magnetization, free energy and internal
# energy are then computed and compared against the exact results. Produces
# a binary MERA defined from disentanglers 'upC' and isometries 'wC' for
# the ground state of the transverse field quantum Ising model.

chiM = 8
chiS = 4
chiU = 4
chiH = 6
chiV = 6

allchi = [chiM,chiS,chiU,chiH,chiV]

relTemp = 1.6 # temp relative to the crit temp, in [0,inf]
numlevels = 20 # number of coarse-grainings

O_dtol = 1e-10
O_disiter = 2000
O_miniter = 200
O_dispon = False
O_convtol = 0.01
O_disweight = 0

def mainTNR(relTemp,allchi,numlevels, dtol = 1e-10, disiter = 2000, miniter = 100, dispon = True, convtol = 0.01, disweight = 0):
    ##### Set bond dimensions and options





    ###### Define partition function (classical Ising)
    Tc = (2/np.log(1+np.sqrt(2)))
    Tval  = relTemp*Tc
    betaval = 1./Tval
    Jtemp = np.zeros((2,2,2))
    Jtemp[0,0,0]=1
    Jtemp[1,1,1]=1
    Etemp = np.array([[np.exp(betaval), np.exp(-betaval)], [np.exp(-betaval),np.exp(betaval)]])

    Ainit = np.einsum(Jtemp,[21,8,1],Jtemp,[22,2,3],Jtemp,[23,4,5],Jtemp,[24,6,7],Etemp,[1,2],Etemp,[3,4],Etemp,[5,6],Etemp,[7,8])
    Xloc = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
    Ainit = np.einsum(Ainit,[1,2,3,4],Xloc,[1,21],Xloc,[2,22],Xloc,[3,23],Xloc,[4,24])
    Atemp = np.einsum(Ainit,[12,13,3,1],Ainit,[3,14,15,2],Ainit,[11,1,4,18],Ainit,[4,2,16,17]).reshape(4,4,4,4)
    Anorm = []
    Anorm.append(np.linalg.norm(Atemp[:]))
    A = []
    A.append(Atemp/Anorm[0])

    ###### Do iterations of TNR
    SPerrs = []
    qC = []
    sC = []
    uC = []
    yC = []
    vC = []
    wC = []

    for k in range(numlevels):
        time1 = time.time()
        A_new, qC_new, sC_new, uC_new, yC_new, vC_new, wC_new, Anorm_new, SPerrs_new = doTNR(A[-1], [chiM,chiS,chiU,chiH,chiV], dtol = O_dtol, disiter = O_disiter, miniter = O_miniter, dispon = O_dispon, convtol = O_convtol)
        A.append(A_new)
        qC.append(qC_new)
        sC.append(sC_new)
        uC.append(uC_new)
        yC.append(yC_new)
        vC.append(vC_new)
        wC.append(wC_new)
        Anorm.append(Anorm_new)
        SPerrs.append(SPerrs_new)
        time2 = time.time()

        print("RGstep: ",k," , Truncation Errors:",SPerrs_new)
        print("time spent for this RGstep: ",time2-time1)
        print("shape of A:",A_new.shape)
        # print("shape of q:",qC_new.shape)
        # print("shape of s:",sC_new.shape)
        # print("shape of u:",uC_new.shape)
        # print("shape of y:",yC_new.shape)
        # print("shape of v:",vC_new.shape)
        # print("shape of w:",wC_new.shape)


    gaugeX = np.moveaxis(np.eye(4).reshape(2,2,2,2),[0,1,2,3],[1,0,2,3]).reshape(4,4)
    upC = []
    upC.append(np.einsum(uC[0],[21,1,23,24],gaugeX,[1,22]))

    for k in range(1,numlevels):
        U, S, Vh = np.linalg.svd(np.einsum(wC[k-1],[1,2,21],wC[k-1],[2,1,22]))
        gaugeX = U@Vh
        upC.append(np.einsum(uC[k],[21,1,23,24],gaugeX,[1,22]))

    ##### Change gauge on disentanglers 'u'
    # gaugeX = reshape(permutedims(reshape(eye(4),2,2,2,2),[2,1,3,4]),4,4);
    # upC = Array{Any}(numlevels); upC[1] = ncon(Any[uC[1],gaugeX],Any[[-1,1,-3,-4],[1,-2]]);
    # for k = 2:numlevels
    #     F = svdfact(ncon(Any[wC[k-1],wC[k-1]],Any[[1,2,-1],[2,1,-2]]));
    #     gaugeX = F[:U]*F[:Vt];
    #     upC[k] = ncon(Any[uC[k],gaugeX],Any[[-1,1,-3,-4],[1,-2]]);
    # end


    sXcg = ((1/4)*(np.kron(np.eye(8),sX) + np.kron(np.eye(4),np.kron(sX,np.eye(2))) + np.kron(np.eye(2),np.kron(sX,np.eye(4))) + np.kron(sX,np.eye(8)))).reshape(4,4,4,4)
    for k in range(numlevels):
        sXcg = np.einsum(sXcg,[3,4,1,2],upC[k],[1,2,6,9],upC[k],[3,4,7,10],wC[k],[5,6,21],wC[k],[9,8,22],wC[k],[5,7,23],wC[k],[10,8,24])
    w,v = np.linalg.eigh(sXcg.reshape(sXcg.shape[0]**2,sXcg.shape[2]**2))
    ExpectX = np.max(w)

    print("Magnetization: ",ExpectX)

    print("##############################################")
    # sXcg = reshape((1/4)*(kron(eye(8),sX) + kron(eye(4),kron(sX,eye(2))) +
    #     kron(eye(2),kron(sX,eye(4))) + kron(sX,eye(8))),4,4,4,4);
    # for k = 1:numlevels
    #     sXcg = ncon(Any[sXcg,upC[k],upC[k],wC[k],wC[k],wC[k],wC[k]],
    #         Any[[3,4,1,2],[1,2,6,9],[3,4,7,10],[5,6,-1],[9,8,-2],[5,7,-3],[10,8,-4]]);
    # end
    # F = eigfact(reshape(sXcg,size(sXcg,1)^2,size(sXcg,3)^2));
    # ExpectX = maximum(F[:values]);  # magnetization

    return ExpectX

if __name__ == "__main__":
    mainTNR(relTemp,allchi,numlevels,O_dtol,O_disiter,O_miniter,O_dispon,O_convtol,O_disweight)
