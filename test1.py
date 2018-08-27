from doScEval import *

chiK = 20

w,v = doScEval(A[sclev],qC[sclev],sC[sclev],yC[sclev],vC[sclev],wC[sclev],chiK)

print("Scaling dimension: ",np.sort(w))
