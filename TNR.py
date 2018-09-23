import numpy as np
from scipy.linalg import expm,sqrtm

def doTNR(Aout,chi):
    chiw,chiy,chiu,chiv=chi['w'],chi['y'],chi['u'],chi['v']
    assert chiu<=chiw, 'chiu>chiw'
    assert chiv<=chiu*chiy, 'chiv>chiu*chiy'
    assert chiy<=chiv**2, 'chiy>chiv**2'
    assert chiw<=chiu**2,'chiw>chiu**2'
    #doTNR
    B,U,vl,vr=optim_Uvlvr(Aout,chi)
    yl,yr,D=optim_ylyrD(B,chi)
    w,Anew,Anorm=optim_w(vl,vr,yl,yr,D,chi)
    #Anew=fix_gauge(Aout,Anew,chi)

    return B,U,vl,vr,yl,yr,D,w,Anew,Anorm

######################################################

def optim_Uvlvr(A,chi,iter=1001,verbose=True,dtol=1e-10):
    #optimize U, vl, vr
    chiu,chiv,chiw,chiy=chi['u'],chi['v'],chi['w'],chi['y']
    chiy=min(chiy,A.shape[0])
    chiw=min(chiw,A.shape[1])
    # chiu=min(chiu,A.shape[1])

    #initialize
    # Uenv=np.random.rand(chiw,chiw,chiu,chiu)
    # Uenv=np.random.rand(chiw,chiw,chiu,chiu) if type(Uold)==type(None) else Uold.conj()
    # Aquad=Aquad_(A)
    # Uenv=np.einsum(Aquad,[4,0,1,5,4,2,3,5])
    Uenv=KP(np.eye(chiw,chiu),np.eye(chiw,chiu)).reshape(chiw,chiw,chiu,chiu)+0.001*np.random.rand(chiw,chiw,chiu,chiu)
    U=TensorUpdateSVD(Uenv,2)
    vr=np.eye(chiy*chiu).reshape(chiy,chiu,chiy*chiu)
    norm_old=0

    for i in range(iter):
        AAU=AAU_(A,U)
        vlenvhalf=vlenvhalf_(AAU,vr)
        vlenv2=vlenv2_(vlenvhalf)
        _,vl=compress(vlenv2,chiv)
        vrenvhalf=vrenvhalf_(AAU,vl)
        vrenv2=vrenv2_(vrenvhalf)
        _,vr=compress(vrenv2,chiv)
        Bhalf=Bhalf_(A,U,vl,vr)
        Uenv2=Uenv2_(A,vl,vr,Bhalf)
        U=TensorUpdateSVD(Uenv2,2)
        if i%100==0:
            norm_U=np.abs(Uenv2.flatten()@U.flatten())
            norm_new=norm_U
            if verbose:
                print('iter=%d, U.Uenv=%.6g'%(i,norm_U))
            # if np.abs(norm_new-norm_old)<dtol*norm_new:
            #     break
            # norm_old=norm_new

    Bhalf=Bhalf_(A,U,vl,vr)
    B=B_(Bhalf)

    # vlenv=np.random.rand(chiy,chiu,chiv)
    # vl=TensorUpdateSVD(vlenv,2)
    # vrenv=np.random.rand(chiy,chiu,chiv)
    # vr=TensorUpdateSVD(vrenv,2)
    # norm_old=0

    # #iterations
    # for i in range(iter):
    #     Bhalf=Bhalf_(A,U,vl,vr)
    #     B=B_(Bhalf)

    #     Uenv=Uenv_(A,U,vl,vr,Bhalf,B)
    #     # print('update U')
    #     U=TensorUpdateSVD(Uenv,2)
    #     vlenv=vlenv_(A,U,vl,vr,Bhalf,B)
    #     # print('update vl')
    #     vl=TensorUpdateSVD(vlenv,2)
    #     vrenv=vrenv_(A,U,vl,vr,Bhalf,B)
    #     # print('update vr')
    #     vr=TensorUpdateSVD(vrenv,2)

    #     if i%100==0:
    #         norm_U=np.abs(Uenv.flatten()@U.flatten())
    #         norm_vl=np.abs(vlenv.flatten()@vl.flatten())
    #         norm_vr=np.abs(vrenv.flatten()@vr.flatten())
    #         norm_new=norm_vr
    #         if verbose:
    #             print('iter=%d, U.Uenv=%.6g, vl.vlenv=%.6g, vr.vrenv=%.6g'%(i,norm_U,norm_vl,norm_vr))
    #         if np.abs(norm_new-norm_old)<dtol*norm_new:
    #             break
    #         norm_old=norm_new

    #evaluation
    if verbose:
        Adouble=Adouble_(A)
        Pu_half=Pu_half_(U,vl,vr)
        _,val,_=np.linalg.svd(rs2(Adouble,2))
        _,val2,_=np.linalg.svd(rs2(Adouble,2)@rs2(Pu_half,4))
        err1=(np.sum(val)-np.sum(val2))/np.sum(val)
        print('error in U, vl, vr optimization: %.6g'%(err1,)) 

        # Pu=Pu_(U,vl,vr).reshape(chiy*chiw*chiy*chiw,chiy*chiw*chiy*chiw)
        # Aquad=np.einsum(A,[0,8,9,1],A,[9,10,3,2],A,[4,5,11,8],A,[11,6,7,10],optimize=('greedy', 2**100)).reshape(chiy*chiw*chiy*chiw,chiy*chiw*chiy*chiw)

        # err1=diff(Aquad,Pu@Aquad@Pu)/diff(Aquad,0)
        # print('error in U, vl, vr optimization: %.6g'%(err1,)) 
    return B,U,vl,vr

def Aquad_(A):
    return np.einsum(A,[0,1,11,8],A,[11,2,3,9],A,[4,8,10,5],A,[10,9,7,6])

def AAU_(A,U):
    return np.einsum(A,[2,6,8,0],A,[8,7,5,1],U,[6,7,3,4],optimize=('greedy', 2**100))
def vlenvhalf_(AAU,vr):
    return np.einsum(AAU,[0,1,2,3,5,6],vr,[6,5,4],optimize=('greedy', 2**100))
def vlenv2_(vlenvhalf):
    return np.einsum(vlenvhalf,[4,5,0,1,6],vlenvhalf.conj(),[4,5,2,3,6],optimize=('greedy', 2**100))
def vrenvhalf_(AAU,vl):
    return np.einsum(AAU,[0,1,6,5,3,2],vl,[6,5,4],optimize=('greedy', 2**100))
def vrenv2_(vrenvhalf):
    return np.einsum(vrenvhalf,[4,5,0,1,6],vrenvhalf.conj(),[4,5,2,3,6],optimize=('greedy', 2**100))
def Uenv2_(A,vl,vr,Bhalf):
    return np.einsum(A,[4,0,10,6],A,[10,1,5,7],vl,[4,2,8],vr,[5,3,9],Bhalf.conj(),[6,7,8,9],optimize=('greedy', 2**100))


def Bhalf_(A,U,vl,vr):
    return np.einsum(A,[4,5,6,0],A,[6,7,10,1],U,[5,7,8,9],vl,[4,8,2],vr,[10,9,3],optimize=('greedy', 2**100))

def B_(Bhalf):
    return np.einsum(Bhalf,[4,5,2,3],Bhalf.conj(),[4,5,0,1],optimize=('greedy', 2**100))

def Uenv_(A,U,vl,vr,Bhalf,B):
    return np.einsum(A,[4,5,6,0],A,[6,7,10,1],vl,[4,8,2],vr,[10,9,3],Bhalf.conj(),[0,1,11,12],B.conj(),[11,12,2,3],optimize=('greedy', 2**100))

def vlenv_(A,U,vl,vr,Bhalf,B):
    vlenv=np.einsum(A,[4,5,6,0],A,[6,7,10,1],U,[5,7,8,9],vr,[10,9,3],Bhalf.conj(),[0,1,11,12],B.conj(),[11,12,2,3],optimize=('greedy', 2**100))
    return vlenv.transpose(1,2,0)

def vrenv_(A,U,vl,vr,Bhalf,B):
    vrenv=np.einsum(A,[4,5,6,0],A,[6,7,10,1],U,[5,7,8,9],vl,[4,8,2],Bhalf.conj(),[0,1,11,12],B.conj(),[11,12,2,3],optimize=('greedy', 2**100))
    return vrenv.transpose(2,1,0)

# def Pu_(U,vl,vr):
#     return np.einsum(U,[1,2,6,7],vl,[0,6,4],vr,[3,7,5],U.conj(),[11,12,8,9],vl.conj(),[10,8,4],vr.conj(),[13,9,5],optimize=('greedy', 2**100))

def Adouble_(A):
    return np.einsum(A,[2,3,6,0],A,[6,4,5,1],optimize=('greedy', 2**100))

def Pu_half_(U,vl,vr):
    return np.einsum(U,[1,2,6,7],vl,[0,6,4],vr,[3,7,5],optimize=('greedy', 2**100))
############################################
def optim_ylyrD(B,chi,verbose=True):
    #optimize yl, yr, D
    chiw,chiy,chiu,chiv=chi['w'],chi['y'],chi['u'],chi['v']
    yl,D,yr=h_cut(B,chiy)
    yl=yl.conj()
    yr=yr.conj()

    #evaluation
    if verbose:
        #err2=diff(B,np.einsum(yl.conj()*D,[0,2,4],yr.conj(),[1,3,4]))/diff(B,0)
        _,val,_=np.linalg.svd(rs2(B.transpose(0,2,1,3),2))
        err2=(np.sum(val)-np.sum(D))/np.sum(val)
        print('error in yl, yr, D optimization: %.6g'%(err2,))
    return yl,yr,D

##############################################
def optim_w(vl,vr,yl,yr,D,chi,iter=101,verbose=True,dtol=1e-8):
    #optimize w

    #initialize
    chiw,chiy,chiu,chiv=chi['w'],chi['y'],chi['u'],chi['v']
    ring=ring_(vl,vr,yl,yr,D)
    wenv=wenv_(ring)
    # _,w=eigCut(wenv.reshape(chiu*chiu,chiu*chiu),chiw)
    # w=w.conj().reshape(chiu,chiu,chiw)
    _,w=compress(wenv,chiw)
    # norm_old=0

    # #iterations
    # for i in range(iter):
    #     wenv2=wenv2_(ring,w)
    #     if i%100==0:
    #         norm_new=np.abs(np.einsum(wenv2,[0,1,2,3],w,[0,1,4],w.conj(),[2,3,4]))
    #         if verbose:
    #             print('iter=%d, w.w*.wenv2=%.6g'%(i,norm_new))
    #         if np.abs(norm_new-norm_old)<dtol*norm_new:
    #             break
    #         norm_old=norm_new
    #     _,w=eigCut(wenv2.reshape(chiu*chiu,chiu*chiu),chiw)
    #     w=w.conj().reshape(chiu,chiu,chiw)

    Anew=np.einsum(ring,[4,5,0,2,6,7],w,[6,7,1],w.conj(),[4,5,3],optimize=('greedy', 2**100))

    #evaluation
    if verbose:
        err3=diff(ring,np.einsum(Anew,[2,7,3,6],w,[0,1,6],w.conj(),[4,5,7],optimize=('greedy', 2**100)))/diff(ring,0)
        print('error in w optimization: %.6g'%(err3,))

    Anorm=np.linalg.norm(Anew)
    Anew/=Anorm

    return w,Anew,Anorm

def ring_(vl,vr,yl,yr,D):
    return np.einsum(vl,[7,1,8],vr,[7,0,6],vl.conj(),[10,5,9],vr.conj(),[10,4,11],yl.conj()*np.sqrt(D),[8,9,3],yr.conj()*np.sqrt(D),[6,11,2],optimize=('greedy', 2**100))


def wenv_(ring):
    return np.einsum(ring,[4,5,6,7,0,1],ring.conj(),[4,5,6,7,2,3],optimize=('greedy', 2**100))

# def wenv2_(ring,w):
#     return np.einsum(ring,[4,5,6,7,0,1],ring.conj(),[8,9,6,7,2,3],w,[8,9,10],w.conj(),[4,5,10],optimize=('greedy', 2**100))

####################################################
def fix_gauge(Aold,Anew,chi,iter=10001,verbose=True,dtol=1e-8):
    #fix gauge
    chiw,chiy,chiu,chiv=chi['w'],chi['y'],chi['u'],chi['v']
    chiy=min(chiy,Aold.shape[0])
    chiw=min(chiw,Aold.shape[1])
    chiu=min(chiu,Aold.shape[1])

    #initialize
    u=np.eye(chiw)
    v=np.eye(chiy)
    norm_old=0

    #iterations
    for i in range(iter):
        uenv=uenv_(Aold,Anew,u,v)
        u=TensorUpdateSVD(uenv,1)
        venv=venv_(Aold,Anew,u,v)
        v=TensorUpdateSVD(venv,1)

        if i%1000==0:
            norm_u=np.abs(uenv.flatten()@u.flatten())
            norm_v=np.abs(venv.flatten()@v.flatten())
            norm_new=norm_v
            if verbose:
                print('iter=%d, u.uenv=%.6g, v.venv=%.6g'%(i,norm_u,norm_v))
            if np.abs(norm_new-norm_old)<dtol*norm_new:
                break
            norm_old=norm_new

    Anew=np.einsum(Anew,[4,5,6,7],u,[5,1],u.conj(),[7,3],v,[6,2],v.conj(),[4,0])

    #evaluation
    if verbose:
        Adiff=diff(Anew,Aold)
    #     Adiff=diff(Aval(Aout),Aval(Anew))/diff(Aval(Aout),0)
        print('Adiff %.6g'%(Adiff,)) 
    return Anew,u,v

def uenv_(Aold,Anew,u,v):
    return np.einsum(Aold.conj(),[5,1,3,4],Anew,[6,0,2,7],u.conj(),[7,4],v,[2,3],v.conj(),[6,5],optimize=('greedy', 2**100))
def venv_(Aold,Anew,u,v):
    return np.einsum(Aold.conj(),[5,1,3,4],Anew,[6,0,2,7],u,[0,1],u.conj(),[7,4],v.conj(),[6,5],optimize=('greedy', 2**100))

# def Aval(A):
#     A2=np.einsum(A,[3,1,3,0])
#     val,_=np.linalg.eigh(A2)
#     return val[::-1]

####################################################
#scaling op

def gl_(vl):
    return np.einsum(vl,[1,4,3],vl.conj(),[0,4,2])
def gr_(vr):
    return np.einsum(vr,[1,4,3],vr.conj(),[0,4,2])
def gu_(A,U,vl,vr):
    Adouble=Adouble_(A)
    Pu_half=Pu_half_(U,vl,vr)
    return np.einsum(Adouble,[0,1,4,5,6,7],Pu_half,[4,5,6,7,2,3])
def gnw_(vl,vr,yr,D,w):
    return np.einsum(vl.conj(),[4,6,1],vr.conj(),[4,5,7],yr.conj()*np.sqrt(D),[0,7,2],w,[5,6,3])
def gne_(vl,vr,yl,D,w):
    return np.einsum(vl.conj(),[4,6,7],vr.conj(),[4,5,1],yl.conj()*np.sqrt(D),[0,7,2],w,[5,6,3])

def rg_(M,gl,gr,gu,gnw,gne):
    return np.einsum(M,[8,9,10,11,12,13,14,15],gl,[11,13,19,21],gr,[10,12,18,20],gu,[14,15,22,23],gu.conj(),[8,9,16,17],gnw,[20,22,4,6],gne,[21,23,5,7],gnw.conj(),[18,16,2,0],gne.conj(),[19,17,3,1])

def gauge_(gnw,gne,u,v):
    if type(u)==type(None):
        return gnw,gne
    else:
        gnw_gauged=np.einsum(gnw@u,[0,1,4,3],v.conj(),[4,2])
        gne_gauged=np.einsum(gne@u,[0,1,4,3],v,[4,2])
        return gnw_gauged,gne_gauged

def Mscaled_(M,A,U,vl,vr,yl,yr,D,w,u,v):
    gl=gl_(vl)
    gr=gr_(vr)
    gu=gu_(A,U,vl,vr)
    gnw=gnw_(vl,vr,yr,D,w)
    gne=gne_(vl,vr,yl,D,w)
    gnw,gne=gauge_(gnw,gne,u,v)
    Mscaled=rg_(M,gl,gr,gu,gnw,gne)
    return Mscaled

####################################################
#utils
sx=np.array([[0,1],[1,0]])
sy=np.array([[0,-1j],[1j,0]])
sz=np.array([[1,0],[0,-1]])
s0=np.array([[1,0],[0,1]])

def Asplit(Ainit,chi,verbose=True):
    '''
    input: Ainit, chi, op: operator acting on a leg
    output:up-down symmetric Aout
    '''
    chiw,chiy,chiu,chiv=chi['w'],chi['y'],chi['u'],chi['v']
    
    #vertical cut
    w1,vtemp,tr1=compress(Ainit,chiw,return_tr=verbose)
    v=vtemp.conj()*np.sqrt(w1)

    #horizontal cut
    vl2,w2,vr2,tr2= h_cut(Ainit,chiy,return_tr=verbose)

    assert diff(vl2,vl2.transpose(1,0,2).conj())<1e-10, 'vl2 err'
    assert diff(vr2,vr2.transpose(1,0,2).conj())<1e-10, 'vr2 err'

    Aout=np.einsum(vr2,[4,5,0],v,[5,7,1],vl2*w2,[6,7,2],v.conj(),[4,6,3])
    Aout/=np.linalg.norm(Aout)

    assert diff(Aout,Aout.transpose(0,3,2,1).conj())<1e-10, 'Anorm err'

    if verbose:
        err1=(tr1-np.sum(w1))/tr1
        err2=(tr2-np.sum(w2))/tr2
        print('vertical truncation err:',err1)
        print('horizontal truncation err:',err2)
        print('Aout shape:',Aout.shape)
    return Aout

def h_cut(A,chimax,dtol=1e-10,return_tr=False):
    #horizontal cut
    As=A.shape
    Ah=A.transpose(0,2,1,3).reshape(As[0]*As[2],As[1]*As[3])
    u,w,vt=np.linalg.svd(Ah,full_matrices=False)
    tr=np.sum(w)
    mask=(w[:chimax]>dtol*tr)
    if any(1-mask):
        print('h_cut: w to be truncated:',w)
    w=w[:chimax][mask]
    u=u[:,:chimax][:,mask]
    vt=vt[:chimax,:][mask,:]
    chinew=len(w)
    vl=u.reshape(As[0],As[2],chinew)
    vr=vt.T.reshape(As[1],As[3],chinew)

    #symmetrize between leg 0 and 1
    uTu=(vl.transpose(2,1,0).reshape(chinew,As[0]*As[2])@u).conj()
    uu=sqrtm(uTu)

    assert diff(uu*w,uu*w[:,np.newaxis])<1e-10*diff(uu*w,0), 'uu does not commute with w'

    vl=vl@uu
    vr=vr@uu.conj()
    if return_tr:
        return vl,w,vr,tr
    else:
        return vl,w,vr

def TensorUpdateSVD(env,leftnum,dtol=1e-10):
    '''
    input: env, dims[N0,N1,...,N_leftnum-1,M_0,M-1,...]
    returns: 
    '''
    envs = env.shape
    U,S,Vh = np.linalg.svd(rs2(env,leftnum),full_matrices=False)
    tr=np.sum(S)
    mask=(S>dtol*tr)
    # if any(1-mask):
    #     print('TensorSVD: S to be truncated:',S)
    out=(U[:,mask]@Vh[mask,:]).conj().reshape(envs)
    # print('nonzero eigenvalues=',np.sum(S>1e-6))
    return out

def rs2(tensor,leftnum):
    ts=tensor.shape
    return tensor.reshape(np.prod(ts[:leftnum]),np.prod(ts[leftnum:]))

def eigCut(rho,chimax):
    w, v = np.linalg.eigh(rho)
    w = w[::-1]
    v = v[:,::-1]
    return w[:chimax],v[:,:chimax]

def compress(env,chimax,dtol=1e-10,return_tr=False):
    """
    compress two legs into one leg, env dimension: (N,M,N,M), 
    output: w,v,tr; w: eigenvalues; projector v, dims: (N,M,K), K<=chimax
    env.v.v.cj() ~ trace(env); env ~ v.conj().w.v^T
    """
    env2=rs2(env,2)
    w,v=np.linalg.eigh(env2)
    w = w[::-1]
    v = v[:,::-1]
    tr=np.sum(w)
    mask=(w[:chimax]>dtol*tr)
    # if any(1-mask):
    #     print('compress: w to be truncated:',w)
    w=w[:chimax][mask]
    v=v[:,:chimax][:,mask]
    v=v.conj().reshape(env.shape[0],env.shape[1],v.shape[1])
    if return_tr:
        return w,v,tr
    else:
        return w,v

def diff(A1,A2):
    return np.linalg.norm(A1-A2)

def KP(*arg):
    if len(arg)<=2:
        return np.kron(*arg)
    else:
        return np.kron(arg[0],KP(*arg[1:]))

def chop(x,tol=1e-10):
    return x.real*(abs(x.real)>tol)+1j*x.imag*(abs(x.imag)>tol)

###############################################################