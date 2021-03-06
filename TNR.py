import numpy as np
from scipy.linalg import expm,sqrtm
import warnings

def doTNR(Aold,chi):
    chiw,chiy,chiu,chiv=chi['w'],chi['y'],chi['u'],chi['v']
    assert chiu<=chiw, 'chiu>chiw'
    assert chiv<=chiu*chiy, 'chiv>chiu*chiy'
    assert chiy<=chiv**2, 'chiy>chiv**2'
    assert chiw<=chiu**2,'chiw>chiu**2'
    A=Aold.copy()
    #doTNR
    B,U,vl,vr,err1=optim_Uvlvr(Aold,chi)
    yl,yr,D,err2=optim_ylyrD(B,chi)
    w,Anew,Anorm,err3=optim_w(vl,vr,yl,yr,D,chi)
    Anew,u,v,Adiff=fix_gauge(Aold,Anew,chi)

    tensors={'A':A,'B':B,'U':U,'vl':vl,'vr':vr,'yl':yl,'yr':yr,'D':D,'w':w,'u':u,'v':v,'Anorm':Anorm}

    return tensors,Anew

######################################################

def optim_Uvlvr(A,chi,iter=1001,dtol=1e-10):
    #optimize U, vl, vr
    chiu,chiv,chiw,chiy=chi['u'],chi['v'],chi['w'],chi['y']
    chiyt=A.shape[0]
    chiwt=A.shape[1]

    #initialize
    Uenv=KP(np.eye(chiwt,chiu),np.eye(chiwt,chiu)).reshape(chiwt,chiwt,chiu,chiu)
    U=TensorUpdateSVD(Uenv,2)
    vr=np.eye(chiyt*chiu).reshape(chiyt,chiu,chiyt*chiu)
    norm_old=0

    for i in range(iter):
        AAU=AAU_(A,U)
        vlenvhalf=vlenvhalf_(AAU,vr)
        vlenv=vlenv_(vlenvhalf)
        _,vl,_=compress(vlenv,chiv)
        vrenvhalf=vrenvhalf_(AAU,vl)
        vrenv=vrenv_(vrenvhalf)
        _,vr,_=compress(vrenv,chiv)
        Bhalf=Bhalf_(A,U,vl,vr)
        Uenv=Uenv_(A,vl,vr,Bhalf)
        U=TensorUpdateSVD(Uenv,2)
        if i%100==0:
            norm_U=np.abs(Uenv.flatten()@U.flatten())
            norm_new=norm_U
            print('iter=%d, U.Uenv=%.6g'%(i,norm_U))
            if np.abs(norm_new-norm_old)<dtol*norm_new:
                break
            norm_old=norm_new

    Bhalf=Bhalf_(A,U,vl,vr)
    B=B_(Bhalf)

    #evaluation
    Adouble=Adouble_(A)
    _,val,_=np.linalg.svd(rs2(Adouble,2))
    _,val2,_=np.linalg.svd(rs2(Bhalf,2))
    err1=1-np.sum(val2)/np.sum(val)
    print('error in U, vl, vr optimization: %.6g'%(err1,))

    return B,U,vl,vr,err1

def AAU_(A,U):
    return ncon(A,[2,6,8,0],A,[8,7,5,1],U,[6,7,3,4])
def vlenvhalf_(AAU,vr):
    return ncon(AAU,[0,1,2,3,5,6],vr,[6,5,4])
def vlenv_(vlenvhalf):
    return ncon(vlenvhalf,[4,5,0,1,6],vlenvhalf.conj(),[4,5,2,3,6])
def vrenvhalf_(AAU,vl):
    return ncon(AAU,[0,1,6,5,3,2],vl,[6,5,4])
def vrenv_(vrenvhalf):
    return ncon(vrenvhalf,[4,5,0,1,6],vrenvhalf.conj(),[4,5,2,3,6])
def Uenv_(A,vl,vr,Bhalf):
    return ncon(A,[4,0,10,6],A,[10,1,5,7],vl,[4,2,8],vr,[5,3,9],Bhalf.conj(),[6,7,8,9])
def Bhalf_(A,U,vl,vr):
    return ncon(A,[4,5,6,0],A,[6,7,10,1],U,[5,7,8,9],vl,[4,8,2],vr,[10,9,3])
def B_(Bhalf):
    return ncon(Bhalf,[4,5,2,3],Bhalf.conj(),[4,5,0,1])
def Adouble_(A):
    return ncon(A,[2,3,6,0],A,[6,4,5,1])

############################################
def optim_ylyrD(B,chi):
    #optimize yl, yr, D
    yl,D,yr,_=h_cut(B,chi['y'])
    yl=yl.conj()
    yr=yr.conj()

    #evaluation
    _,val,_=np.linalg.svd(rs2(B.transpose(0,2,1,3),2))
    err2=(np.sum(val)-np.sum(D))/np.sum(val)
    print('error in yl, yr, D optimization: %.6g'%(err2,))

    return yl,yr,D,err2

##############################################
def optim_w(vl,vr,yl,yr,D,chi,iter=101,dtol=1e-8):
    #optimize w

    #initialize
    ring=ring_(vl,vr,yl,yr,D)
    wenv=wenv_(ring)
    _,w,_=compress(wenv,chi['w'])

    Anew=ncon(ring,[4,5,0,2,6,7],w,[6,7,1],w.conj(),[4,5,3])

    #evaluation
    err3=diff(ring,ncon(Anew,[2,7,3,6],w,[0,1,6],w.conj(),[4,5,7]))/diff(ring,0)
    print('error in w optimization: %.6g'%(err3,))

    Anorm=np.linalg.norm(Anew)
    Anew/=Anorm

    return w,Anew,Anorm,err3

def ring_(vl,vr,yl,yr,D):
    return ncon(vl,[7,1,8],vr,[7,0,6],vl.conj(),[10,5,9],vr.conj(),[10,4,11],yl.conj()*np.sqrt(D),[8,9,3],yr.conj()*np.sqrt(D),[6,11,2])


def wenv_(ring):
    return ncon(ring,[4,5,6,7,0,1],ring.conj(),[4,5,6,7,2,3])

####################################################
def fix_gauge(Aold,Anew,chi,iter=10001,dtol=1e-8):
    #fix gauge

    #initialize u
    utrunc=np.eye(Anew.shape[1],Aold.shape[1])
    Aold2=ncon(utrunc,[0,3],utrunc,[1,4],Aold,[2,3,2,4])
    _,uAold=np.linalg.eigh(Aold2)
    Anew2=ncon(Anew,[2,0,2,1])
    _,uAnew=np.linalg.eigh(Anew2)
    u=uAnew.conj()@uAold.T

    #initialize v
    vtrunc=np.eye(Anew.shape[0],Aold.shape[0])
    Aold3=ncon(vtrunc,[0,3],vtrunc,[1,4],Aold,[3,2,4,2]).real
    _,vAold=np.linalg.eigh(Aold3+Aold3.T)
    Anew3=ncon(Anew,[0,2,1,2]).real
    _,vAnew=np.linalg.eigh(Anew3+Anew3.T)
    v=vAnew@vAold.T

    norm_old=0

    #iterations
    for i in range(iter):
        venv=venv_(Aold,Anew,u,v).real
        v=TensorUpdateSVD(venv,1,use_mask=False)
        uenv=uenv_(Aold,Anew,u,v)
        u=TensorUpdateSVD(uenv,1,use_mask=False)

        if i%1000==0:
            # if diff(venv,venv.conj())>1e-5*diff(venv,0):
            #     warnings.warn('venv err')
            # if diff(v,v.conj())>1e-5*diff(v,0):
            #     warnings.warn('v err')
            norm_u=np.abs(uenv.flatten()@u.flatten())
            norm_v=np.abs(venv.flatten()@v.flatten())
            norm_new=norm_v
            print('iter=%d, u.uenv=%.6g, v.venv=%.6g'%(i,norm_u,norm_v))
            if np.abs(norm_new-norm_old)<dtol*norm_new:
                break
            norm_old=norm_new

    Anew=ncon(Anew,[4,5,6,7],u,[5,1],u.conj(),[7,3],v,[6,2],v.conj(),[4,0])

    #evaluation
    As=np.min((Aold.shape,Anew.shape),axis=0)
    Adiff=diff(Anew[:As[0],:As[1],:As[0],:As[1]],Aold[:As[0],:As[1],:As[0],:As[1]])
    print('Adiff %.6g'%(Adiff,)) 
    return Anew,u,v,Adiff

def uenv_(Aold,Anew,u,v):
    utrunc=np.eye(Anew.shape[1],Aold.shape[1])
    vtrunc=np.eye(Anew.shape[0],Aold.shape[0])
    return ncon(utrunc,[1,8],Aold.conj(),[5,8,3,4],Anew,[6,0,2,7],u.conj()@utrunc,[7,4],v@vtrunc,[2,3],v.conj()@vtrunc,[6,5])

def venv_(Aold,Anew,u,v):
    utrunc=np.eye(Anew.shape[1],Aold.shape[1])
    vtrunc=np.eye(Anew.shape[0],Aold.shape[0])
    return ncon(vtrunc,[3,8],Aold.conj(),[5,1,8,4],Anew,[6,0,2,7],u@utrunc,[0,1],u.conj()@utrunc,[7,4],v.conj()@vtrunc,[6,5])

####################################################
#scaling op

def rg_(M,scale_tensors):
    gl=scale_tensors['gl']
    gr=scale_tensors['gr']
    gu=scale_tensors['gu']
    gnw=scale_tensors['gnw']
    gne=scale_tensors['gne']
    return ncon(M,[8,9,10,11,12,13,14,15],gl,[11,13,19,21],gr,[10,12,18,20],gu,[14,15,22,23],gu.conj(),[8,9,16,17],gnw,[20,22,4,6],gne,[21,23,5,7],gnw.conj(),[18,16,2,0],gne.conj(),[19,17,3,1])

def scale_tensors_(tensors):
    A=tensors['A'];U=tensors['U'];vl=tensors['vl'];vr=tensors['vr'];yl=tensors['yl'];yr=tensors['yr'];D=tensors['D'];w=tensors['w'];u=tensors['u'];v=tensors['v']
    gl=gl_(vl)
    gr=gr_(vr)
    gu=gu_(A,U,vl,vr)
    gnw=gnw_(vl,vr,yr,D,w)
    gne=gne_(vl,vr,yl,D,w)
    gnw,gne=gauge_(gnw,gne,u,v)
    scale_tensors={'gl':gl,'gr':gr,'gu':gu,'gnw':gnw,'gne':gne}
    return scale_tensors

def gl_(vl):
    return ncon(vl,[1,4,3],vl.conj(),[0,4,2])
def gr_(vr):
    return ncon(vr,[1,4,3],vr.conj(),[0,4,2])
def gu_(A,U,vl,vr):
    return Bhalf_(A,U,vl,vr)
def gnw_(vl,vr,yr,D,w):
    return ncon(vl.conj(),[4,6,1],vr.conj(),[4,5,7],yr.conj()*np.sqrt(D),[0,7,2],w,[5,6,3])
def gne_(vl,vr,yl,D,w):
    return ncon(vl.conj(),[4,6,7],vr.conj(),[4,5,1],yl.conj()*np.sqrt(D),[0,7,2],w,[5,6,3])
def gauge_(gnw,gne,u,v):
    '''
    Attach gauge transformations u,v to gnw,gne if available
    '''
    if u is None:
        return gnw,gne
    else:
        gnw_gauged=ncon(gnw@u,[0,1,4,3],v.conj(),[4,2])
        gne_gauged=ncon(gne@u,[0,1,4,3],v,[4,2])
        return gnw_gauged,gne_gauged

####################################################
#eval op

def eval_op(MO0,tensors_list):
    '''
    evaluates expectation for operator op
    '''
    RGsteps=len(tensors_list)-1
    A0=tensors_list[0]['A']
    MA=MA_(A0)
    MAtrace=Mtrace_(MA)
    MA_list=[MA/MAtrace]
    MO_list=[MO0/MAtrace]

    for i in range(RGsteps):
        scale_tensors=scale_tensors_(tensors_list[i])
        MA=rg_(MA_list[i],scale_tensors)
        MO=rg_(MO_list[i],scale_tensors)
        MAtrace=Mtrace_(MA)
        MA_list.append(MA/MAtrace)
        MO_list.append(MO/MAtrace)
        expect_O=Mtrace_(MO/MAtrace)
        print('RGsteps:',i,', <O> =',chop(expect_O))

    return expect_O

def MA_(A0):
    return ncon(A0,[2,9,8,0],A0,[8,10,3,1],A0,[4,6,11,9],A0,[11,7,5,10])
def MO_(A0,op):
    return ncon(A0,[2,9,8,0],A0,[8,10,3,1],A0,[4,6,11,9],A0@op,[11,7,5,10])
def Mtrace_(MA):
    return ncon(MA,[0,1,2,2,3,3,0,1])

def MOq_(op,Asplit_tensors,coarsen_list):

    v,w1,vl2,vr2,w2,Anorm=Asplit_tensors['v'],Asplit_tensors['w1'],Asplit_tensors['vl2'],Asplit_tensors['vr2'],Asplit_tensors['w2'],Asplit_tensors['Anorm']

    Az=0.5*(ncon(vr2*np.sqrt(w2),[4,5,0],v,[5,7,1],op@vl2*np.sqrt(w2),[6,7,2],v.conj(),[4,6,3])+ncon(vr2*np.sqrt(w2),[4,5,0],v,[5,7,1],vl2*np.sqrt(w2),[6,7,2],op@v.conj(),[4,6,3]))/Anorm

    for coarsen_tensors in coarsen_list[:-1]:
        A,wl,wr,Anorm=coarsen_tensors['A'],coarsen_tensors['wl'],coarsen_tensors['wr'],coarsen_tensors['Anorm']
        Az=0.5*(Acoarse_(A,Az,wl,wr)+Acoarse_(Az,A,wl,wr))/Anorm

    A0=coarsen_list[-1]['A']

    MOq=0.5*(ncon(A0,[2,9,8,0],A0,[8,10,3,1],A0,[4,6,11,9],Az,[11,7,5,10])+ncon(A0,[2,9,8,0],Az,[8,10,3,1],A0,[4,6,11,9],A0,[11,7,5,10]))
    return MOq

####################################################
# scaleop compress legs

def g_compressed(A,gu,gr,gl,chik,chiq):
    wvenv=wvenv_(A)
    ev,wv,tr=compress(wvenv,chik)
    print('gu compression error:',(tr-np.sum(ev))/tr)
    wrenv=wrenv_(gr)
    ev,wr,tr=compress(wrenv,chiq,do_symmetrize=True)
    print('gr compression error:',(tr-np.sum(ev))/tr)
    wlenv=wlenv_(gl)
    ev,wl,tr=compress(wlenv,chiq,do_symmetrize=True)
    print('gl compression error:',(tr-np.sum(ev))/tr)

    guw=np.einsum(gu,[3,4,1,2],wv.conj(),[3,4,0])
    glw=np.einsum(gl,[3,4,1,2],wl.conj(),[3,4,0])
    grw=np.einsum(gr,[3,4,1,2],wr.conj(),[3,4,0])

    return guw,glw,grw,wv,wr,wl

def wvenv_(A):
    return np.einsum(A,[8,0,6,5],A,[6,1,7,4],A.conj(),[8,2,9,5],A.conj(),[9,3,7,4])
def wrenv_(gr):
    return ncon(gr.conj(),[0,1,4,5],gr,[2,3,4,5])
def wlenv_(gl):
    return ncon(gl.conj(),[0,1,4,5],gl,[2,3,4,5])

####################################################
#initialize A
def Asplit(Ainit,chi):
    '''
    input: Ainit, chi, op: operator acting on a leg
    output:up-down symmetric normalized Aout
    '''
    chiw,chiy,chiu,chiv=chi['w'],chi['y'],chi['u'],chi['v']
    
    #vertical cut
    w1,vtemp,tr1=compress(Ainit,chiw)
    v=vtemp.conj()*np.sqrt(w1)

    #horizontal cut
    vl2,w2,vr2,tr2= h_cut(Ainit,chiy)

    if diff(vl2*w2,(vl2*w2).transpose(1,0,2).conj())>1e-5*diff(vl2*w2,0):
        warnings.warn('Asplit vl2 err')
    if diff(vr2*w2,(vr2*w2).transpose(1,0,2).conj())>1e-5*diff(vr2*w2,0):
        warnings.warn('Asplit vr2 err')

    Aout=ncon(vr2*np.sqrt(w2),[4,5,0],v,[5,7,1],vl2*np.sqrt(w2),[6,7,2],v.conj(),[4,6,3])
    Anorm=np.linalg.norm(Aout)
    Aout/=Anorm

    if diff(Aout,Aout.transpose(0,3,2,1).conj())>1e-5*diff(Aout,0):
        warnings.warn('Asplit Aout err')

    # evaluation
    err1=(tr1-np.sum(w1))/tr1
    err2=(tr2-np.sum(w2))/tr2
    print('vertical truncation err:',err1)
    print('horizontal truncation err:',err2)
    print('Aout shape:',Aout.shape)

    Asplit_tensors={'v':v,'w1':w1,'vl2':vl2,'vr2':vr2,'w2':w2,'Anorm':Anorm,'err1':err1,'err2':err2}

    return Asplit_tensors, Aout

def coarsen(A,chi):
    '''
    performs coarsening pre-step for 1d quantum models
    input: initial A, chi: maximum dimensions
    output: coarsen_tensors: tensors used in the process; Acoarse: coarsened A
    '''
    chiw,chiy,chiu,chiv=chi['w'],chi['y'],chi['u'],chi['v']
    waenv=waenv_(A)
    wl,ev,wr,tr=h_cut(waenv,chiy)
    wl=wl.conj()
    wr=wr.conj()
    Acoarse=Acoarse_(A,A,wl,wr)
    Anorm=np.linalg.norm(Acoarse)
    Acoarse/=Anorm
    tr2=np.sum(ev)
    err=(tr-tr2)/tr
    print('Coarsen truncation error: %.6g'%(err.real,),'Acoarse shape:',Acoarse.shape)
    coarsen_tensors={'A':A,'wl':wl,'wr':wr,'Anorm':Anorm,'err':err.real}
    return coarsen_tensors, Acoarse

def waenv_(A):
    return ncon(A.conj(),[0,8,6,4],A.conj(),[2,5,7,8],A,[1,9,6,4],A,[3,5,7,9])
def Acoarse_(A1,A2,wl,wr):
    return ncon(A1,[4,8,6,3],A2,[5,1,7,8],wl,[6,7,2],wr,[4,5,0])

####################################################
#utils

def h_cut(A,chimax,dtol=1e-10):
    '''
    Input: 
    A: a four leg tensor of shape (N,N,N,N)
    output: vl,w,vr, vl.shape=(N,N,K), w.shape=(K,), vr.shape=(N,N,K)
    which approximates A.transpose(0,2,1,3)~vl.w.vr.transpose(2,0,1)
    '''
    #horizontal cut
    As=A.shape
    Ah=A.transpose(0,2,1,3).reshape(As[0]*As[2],As[1]*As[3])
    u,w,vt=np.linalg.svd(Ah,full_matrices=False)
    tr=np.sum(w)
    vl=u.reshape(As[0],As[2],-1)
    vr=vt.T.reshape(As[1],As[3],-1)
    vl,vr=symmetrize(vl,w,vr)

    mask=(w[:chimax]>dtol*tr)
    w=w[:chimax][mask]
    vl=vl[:,:,:chimax][:,:,mask]
    vr=vr[:,:,:chimax][:,:,mask]
    
    return vl,w,vr,tr

def compress(env,chimax,dtol=1e-10,do_symmetrize=False):
    """
    compress two legs into one leg, env dimension: (N,M,N,M), 
    output: w,v,tr; w: eigenvalues; projector v, dims: (N,M,K), K<=chimax
    env.v.v.ct() ~ trace(env); env ~ v.conj().w.v^T
    """
    env2=rs2(env,2)
    w,v=np.linalg.eigh(env2)
    w = w[::-1]
    v = v[:,::-1]
    tr=np.sum(w)
    if do_symmetrize and env.shape[0]==env.shape[1]:
        vl=v.reshape(env.shape[0],env.shape[1],-1)
        vr=vl.conj()
        vl,vr=symmetrize(vl,w,vr)
        v=rs2(vl,2)

    mask=(w[:chimax]>dtol*tr)
    w=w[:chimax][mask]
    v=v[:,:chimax][:,mask]
    v=v.conj().reshape(env.shape[0],env.shape[1],v.shape[1])
    return w,v,tr

def symmetrize(vl,w,vr):
    '''
    input: vl :(N,N,K), vr: (N,N,K)
    output: symmetrized vl,vr
    '''
    N,_,K=vl.shape
    u=vl.reshape(N*N,K)
    uTu=(vl.transpose(2,1,0).reshape(K,N*N)@u).conj()
    uu=sqrtm(uTu)
    if diff(uu*w,uu*w[:,np.newaxis])>1e-5*diff(uu*w,0):
        warnings.warn('uu does not commute with w')
    vl=vl@uu
    vr=vr@uu.conj()
    if diff(vl*w,(vl*w).transpose(1,0,2).conj())>1e-5*diff(vl*w,0):
        warnings.warn('vl not symmetric')
    if diff(vr*w,(vr*w).transpose(1,0,2).conj())>1e-5*diff(vr*w,0):
        warnings.warn('vr not symmetric')

    return vl,vr

def TensorUpdateSVD(env,leftnum,dtol=1e-10,use_mask=True):
    '''
    input: env, dims[N0,N1,...,N_leftnum-1,M_0,M-1,...]
    returns: isometry 'out' with dim same as env, which maximizes out.env
    '''
    envs = env.shape
    env[np.abs(env)<dtol*np.linalg.norm(env)]=0
    U,S,Vh = np.linalg.svd(rs2(env,leftnum),full_matrices=False)
    tr=np.sum(S)
    mask=(S>dtol*tr) if use_mask else np.full(S.shape,True)
    out=(U[:,mask]@Vh[mask,:]).conj().reshape(envs)
    return out

def rs2(tensor,leftnum):
    ts=tensor.shape
    return tensor.reshape(np.prod(ts[:leftnum]),np.prod(ts[leftnum:]))

def diff(A1,A2):
    return np.linalg.norm(A1-A2)

def KP(*arg):
    if len(arg)<=2:
        return np.kron(*arg)
    else:
        return np.kron(arg[0],KP(*arg[1:]))

def ncon(*arg):
    return np.einsum(*arg,optimize=('greedy',2**100))

def chop(x,tol=1e-10):
    return x.real*(abs(x.real)>tol)+1j*x.imag*(abs(x.imag)>tol)

sx=np.array([[0,1],[1,0]])
sy=np.array([[0,-1j],[1j,0]])
sz=np.array([[1,0],[0,-1]])
s0=np.array([[1,0],[0,1]])

from scipy import pi
R=np.diag(np.exp([0,2*pi*1j/3,-2*pi*1j/3]))
M=np.array([[0,0,1],[1,0,0],[0,1,0]])
###############################################################