# mainScInvTNR
# ------------------------
# by Glen Evenbly
#
# Script file for initializing a partition function (here the classical
# Ising model at critical temperature) before passing to a version of the
# Tensor Network Renormalization (TNR) coarse-graining routine tailored to
# capture the scale-invariance of the critical point. The free energy and
# internal energy are then computed and compared against the exact results.
# Produces a binary MERA defined from disentanglers 'upC' and isometries
# 'wC' for the ground state of the transverse field quantum Ising model.

from doScInvTNR import *

import numpy as np
import matplotlib.pyplot as plt
import time

time_start = time.time()


##### Set bond dimensions and options
chiM = 12
chiS = 6
chiU = 8
chiH = 10
chiV = 10

numlevels = 10 # number of coarse-grainings

O_dtol = 1e-10
O_disiter = 2000
O_miniter = 100
O_dispon = True
O_convtol = 0.01
O_midsteps = 20
O_mixratio = 10

###### Define partition function (classical Ising)
Tc = (2/np.log(1+np.sqrt(2)))
relTemp = 1
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

# from initialize_XY import Atemp
# from initialize_Ising import Atemp

print("Initial Atemp.shape: ",Atemp.shape)
print("Vertical reflection: ",np.linalg.norm(Atemp-np.moveaxis(Atemp,[0,1,2,3],[0,3,2,1])))
print("Horizontal reflection: ",np.linalg.norm(Atemp-np.moveaxis(Atemp,[0,1,2,3],[2,1,0,3])))
print("Vertical + Horizontal reflection: ",np.linalg.norm(Atemp-np.moveaxis(Atemp,[0,1,2,3],[2,3,0,1])))




Anorm = [0]*(numlevels+1)
Anorm[0]=np.linalg.norm(Atemp)
A = [np.nan]*(numlevels+1)
A[0]=Atemp/Anorm[0]

Adiff = [1]*numlevels

SPerrs = [0]*numlevels
qC = [np.nan]*numlevels
sC = [np.nan]*numlevels
uC = [np.nan]*numlevels
yC = [np.nan]*numlevels
vC = [np.nan]*numlevels
wC = [np.nan]*numlevels
C = [np.nan]*numlevels

###### Do iterations of TNR
for k in range(numlevels):
    time1 = time.time()
    if k < 1:
        sctype = 0
        A[k+1], C[k], qC[k], sC[k], uC[k], yC[k], vC[k], wC[k], Anorm[k+1], SPerrs[k] = \
        doScInvTNR(A[k],[chiM,chiS,chiU,chiH,chiV],[0],[0],[0],[0],[0],[0],[0],
                   dtol = O_dtol, disiter = O_disiter, miniter = O_miniter, dispon = O_dispon, convtol = O_convtol, mixratio = O_mixratio, midsteps = O_midsteps, sctype = sctype)
    elif k == 1:
        sctype = 1
        A[k+1], C[k], qC[k], sC[k], uC[k], yC[k], vC[k], wC[k], Anorm[k+1], SPerrs[k] = \
        doScInvTNR(A[k],[chiM,chiS,chiU,chiH,chiV],[0],[0],[0],[0],[0],[0],[0],
                   dtol = O_dtol, disiter = O_disiter, miniter = O_miniter, dispon = O_dispon, convtol = O_convtol, mixratio = O_mixratio, midsteps = O_midsteps, sctype = sctype)
        Adiff[k] = np.linalg.norm((A[k+1] - A[k]))
    else:
        sctype = 2
        A[k+1], C[k], qC[k], sC[k], uC[k], yC[k], vC[k], wC[k], Anorm[k+1], SPerrs[k] = \
        doScInvTNR(A[k],[chiM,chiS,chiU,chiH,chiV],C[k-1],qC[k-1],sC[k-1],uC[k-1],yC[k-1],vC[k-1],wC[k-1],
                   dtol = O_dtol, disiter = O_disiter, miniter = O_miniter, dispon = O_dispon, convtol = O_convtol, mixratio = O_mixratio, midsteps = O_midsteps, sctype = sctype)
        Adiff[k] = np.linalg.norm((A[k+1] - A[k]))
    time2 = time.time()
    print("RGstep: %d ,A_differ: %.6g , Truncation Errors: %.6g, %.6g, %.6g" %(k, Adiff[k],*tuple(SPerrs[k])))
    print("time spent for this RGstep: %.6g"%(time2-time1,))
    print("shape of A:",A[k+1].shape)
    print("shape of q:",qC[k].shape)
    print("shape of s:",sC[k].shape)
    print("shape of u:",uC[k].shape)
    print("shape of y:",yC[k].shape)
    print("shape of v:",vC[k].shape)
    print("shape of w:",wC[k].shape)
    print("Vertical reflection: ",np.linalg.norm(A[k+1]-np.moveaxis(A[k+1],[0,1,2,3],[0,3,2,1])))
    print("Horizontal reflection: ",np.linalg.norm(A[k+1]-np.moveaxis(A[k+1],[0,1,2,3],[2,1,0,3])))
    print("Vertical + Horizontal reflection: ",np.linalg.norm(A[k+1]-np.moveaxis(A[k+1],[0,1,2,3],[2,3,0,1])))

from doScEval import doScEval
print("Adiff: ",Adiff)
sclev = np.argmin(Adiff)
print("sclev: ",sclev)
chiK = 20
N_level = 20
time1 = time.time()
scDims=doScEval(A[sclev],qC[sclev],sC[sclev],yC[sclev],vC[sclev],wC[sclev],chiK,N_level)
time2 = time.time()

spec = np.sort(scDims)
spec = spec - spec[0]

print("Scaling dimension: ",spec)
print("time spent for diagonalization: ",time2-time1)

time_end = time.time()

print("Total time for the whole calculation: ",time_end-time_start)

path1 = "./scaling_dim.txt"
file1 = open(path1,'w')
file1.write(str(spec.tolist()).replace("e","*^").replace("[","{").replace("]","}"))
file1.close()

plt.scatter(np.arange(N_level),spec)

plt.savefig('scaling_dim.pdf')
plt.show()
