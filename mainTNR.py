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

##### Set bond dimensions and options
chiM = 6
chiS = 4
chiU = 4
chiH = 6
chiV = 6

allchi = [chiM,chiS,chiU,chiH,chiV]

relTemp = 0.995 # temp relative to the crit temp, in [0,inf]
numlevels = 12 # number of coarse-grainings

O_dtol = 1e-10
O_disiter = 2000
O_miniter = 200
O_dispon = True
O_convtol = 0.01
O_disweight = 0

print('relTemp=',relTemp)

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
Anorm = [0]*(numlevels+1)
Anorm[0]=np.linalg.norm(Atemp)
A = [np.nan]*(numlevels+1)
A[0]=Atemp/Anorm[0]

SPerrs = [0]*numlevels
qC = [np.nan]*numlevels
sC = [np.nan]*numlevels
uC = [np.nan]*numlevels
yC = [np.nan]*numlevels
vC = [np.nan]*numlevels
wC = [np.nan]*numlevels

###### Do iterations of TNR
for k in range(numlevels):
    A[k+1], qC[k], sC[k], uC[k], yC[k], vC[k], wC[k], Anorm[k+1], SPerrs[k] = doTNR(A[k], [chiM,chiS,chiU,chiH,chiV], dtol = O_dtol, disiter = O_disiter, miniter = O_miniter, dispon = O_dispon, convtol = O_convtol)

    print("RGstep: %d , Truncation Errors: %.6g, %.6g, %.6g, %.6g"%(k,*SPerrs[k]))

##### Change gauge on disentanglers 'u'
gaugeX = np.eye(4).reshape(2,2,2,2).transpose(1,0,2,3).reshape(4,4)
upC = [np.nan]*numlevels
upC[0]=np.einsum(uC[0],[21,1,23,24],gaugeX,[1,22])

for k in range(1,numlevels):
    U, S, Vh = np.linalg.svd(np.einsum(wC[k-1],[1,2,21],wC[k-1],[2,1,22]))
    gaugeX = U@Vh
    upC[k]=np.einsum(uC[k],[21,1,23,24],gaugeX,[1,22])

##### Magnetization
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

