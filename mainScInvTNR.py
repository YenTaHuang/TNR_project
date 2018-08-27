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
from doScEval import *
import matplotlib.pyplot as plt

import time

time_start = time.time()

##### Set bond dimensions and options
chiM = 12
chiS = 6
chiU = 8
chiH = 10
chiV = 10

# chiM = 5
# chiS = 2
# chiU = 3
# chiH = 4
# chiV = 4

numlevels = 8 # number of coarse-grainings

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
Anorm = []
Anorm.append(np.linalg.norm(Atemp))
A = []
A.append(Atemp/Anorm[0])

Adiff = [None]

SPerrs = [None]
qC = [None]
sC = [None]
uC = [None]
yC = [None]
vC = [None]
wC = [None]
C = [None]

###### Do iterations of TNR
for k in range(numlevels):
    time1 = time.time()
    if k < 1:
        sctype = 0
        A_new, C_new, qC_new, sC_new, uC_new, yC_new, vC_new, wC_new, Anorm_new, SPerrs_new =doScInvTNR(A[k],[chiM,chiS,chiU,chiH,chiV],[0],[0],[0],[0],[0],[0],[0],dtol = O_dtol, disiter = O_disiter, miniter = O_miniter, dispon = O_dispon, convtol = O_convtol, mixratio = O_mixratio, midsteps = O_midsteps, sctype = sctype)
        Adiff_new = None
        A.append(A_new)
        C.append(C_new)
        qC.append(qC_new)
        sC.append(sC_new)
        uC.append(uC_new)
        yC.append(yC_new)
        vC.append(vC_new)
        wC.append(wC_new)
        Anorm.append(Anorm_new)
        SPerrs.append(SPerrs_new)
        Adiff.append(Adiff_new)

    elif k == 1:
        sctype = 1
        A_new, C_new, qC_new, sC_new, uC_new, yC_new, vC_new, wC_new, Anorm_new, SPerrs_new =doScInvTNR(A[k],[chiM,chiS,chiU,chiH,chiV],[0],[0],[0],[0],[0],[0],[0],dtol = O_dtol, disiter = O_disiter, miniter = O_miniter, dispon = O_dispon, convtol = O_convtol, mixratio = O_mixratio, midsteps = O_midsteps, sctype = sctype)
        A.append(A_new)
        C.append(C_new)
        qC.append(qC_new)
        sC.append(sC_new)
        uC.append(uC_new)
        yC.append(yC_new)
        vC.append(vC_new)
        wC.append(wC_new)
        Anorm.append(Anorm_new)
        SPerrs.append(SPerrs_new)
        Adiff.append(np.linalg.norm((A[-2] - A[-1]).reshape(chiH*chiV,chiH*chiV)))
    else:
        sctype = 2
        A_new, C_new, qC_new, sC_new, uC_new, yC_new, vC_new, wC_new, Anorm_new, SPerrs_new =doScInvTNR(A[k],[chiM,chiS,chiU,chiH,chiV],C[-1],qC[-1],sC[-1],uC[-1],yC[-1],vC[-1],wC[-1],dtol = O_dtol, disiter = O_disiter, miniter = O_miniter, dispon = O_dispon, convtol = O_convtol, mixratio = O_mixratio, midsteps = O_midsteps, sctype = sctype)

        A.append(A_new)
        C.append(C_new)
        qC.append(qC_new)
        sC.append(sC_new)
        uC.append(uC_new)
        yC.append(yC_new)
        vC.append(vC_new)
        wC.append(wC_new)
        Anorm.append(Anorm_new)
        SPerrs.append(SPerrs_new)
        
        Adiff_new = np.linalg.norm((A[-2] - A[-1]).reshape(chiH*chiV,chiH*chiV))
        Adiff.append(Adiff_new)
    time2 = time.time()
    print('\x1b[6;30;42m'+"#############################################"+ '\x1b[0m')
    print("RGstep: ",k," ,A_differ: ",Adiff_new," , Truncation Errors:",SPerrs_new)
    print("time spent for this RGstep: ",time2-time1)
    print("shape of A:",A_new.shape)

###############################################

print("Adiff: ",Adiff)
sclev = Adiff.index(min(x for x in Adiff if x != None))
print("sclev: ",sclev)


chiK = 20
N_level = 20
time1 = time.time()
w = doScEval(A[sclev],qC[sclev],sC[sclev],yC[sclev],vC[sclev],wC[sclev],chiK,N_level)
time2 = time.time()

spec = np.sort(w)
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
