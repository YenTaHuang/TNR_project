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
chiM = 8
chiS = 4
chiU = 4
chiH = 6
chiV = 6

relTemp = 0.5 # temp relative to the crit temp, in [0,inf]
numlevels = 12 # number of coarse-grainings

O_dtol = 1e-10
O_disiter = 2000
O_miniter = 200
O_dispon = True
O_convtol = 0.01
O_disweight = 0

###### Your paths go here
from doTNR import *

###### Define partition function (classical Ising)
Tc = (2/np.log(1+np.sqrt(2)))
Tval  = relTemp*Tc
betaval = 1./Tval
Jtemp = np.zeros((2,2,2))
Jtemp[0,0,0]=1
Jtemp[1,1,1]=1
Etemp = np.array([[np.exp(betaval), np.exp(-betaval)], [np.exp(-betaval),np.exp(betaval)]])

Ainit = np.einsum(Jtemp,[-4,8,1],Jtemp,[-3,2,3],Jtemp,[-2,4,5],Jtemp,[-1,6,7],Etemp,[1,2],Etemp,[3,4],Etemp,[5,6],Etemp,[7,8])
Xloc = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
Ainit = np.einsum(Ainit,[1,2,3,4],Xloc,[1,-4],Xloc,[2,-3],Xloc,[3,-2],Xloc,[4,-1])
Atemp = np.einsum(Ainit,[-7,-6,3,1],Ainit,[3,-5,-4,2],Ainit,[-8,1,4,-1],Ainit,[4,2,-3,-2]).reshape(4,4,4,4)
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
    print("shape of A:",A_new.shape)
    print("RGstep: ",k," , Truncation Errors:",SPerrs_new)
    print("time spent for this RGstep: ",time2-time1)
