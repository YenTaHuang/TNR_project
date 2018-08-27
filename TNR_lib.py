#!/usr/bin/env python3

import numpy as np
import time

# initialize A
chi_i = 2

def boltzmann(i,j,k,l,beta):
    return np.exp(beta*((2*i-1)*(2*j-1)+(2*j-1)*(2*k-1)+(2*k-1)*(2*l-1)+(2*l-1)*(2*i-1)))

def initialize_A(beta):
    A = np.array([[[[boltzmann(i,j,k,l,beta) for l in range(chi_i)] for k in range(chi_i)] for j in range(chi_i)] for i in range(chi_i)])
    A = A/magnitude(A)
    return A

def initialize_A_with_h(beta,h):
    A = np.array([[[[np.exp(beta*h*((2*i-1)+(2*j-1)+(2*k-1)+(2*l-1)))*boltzmann(i,j,k,l,beta) for l in range(chi_i)] for k in range(chi_i)] for j in range(chi_i)] for i in range(chi_i)])
    A = A/magnitude(A)
    return A

def initialize_M(beta):
    A_tilde = np.array([[[[(2*j-1)*boltzmann(i,j,k,l,beta) for l in range(chi_i)] for k in range(chi_i)] for j in range(chi_i)] for i in range(chi_i)])
    A = np.array([[[[boltzmann(i,j,k,l,beta) for l in range(chi_i)] for k in range(chi_i)] for j in range(chi_i)] for i in range(chi_i)])
    mag = magnitude(A)
    A = A/mag
    A_tilde = A_tilde/mag
    M1 = np.einsum('ailh,bcji,edjk,fklg',A_tilde,A,np.conjugate(A),np.conjugate(A))
    M2 = np.einsum('ailh,bcji,edjk,fklg',A,A,np.conjugate(A),np.conjugate(A_tilde))
    return 0.5*(M1+M2)

def initialize_M_with_h(beta,h):
    A_tilde = np.array([[[[(2*j-1)*np.exp(beta*h*((2*i-1)+(2*j-1)+(2*k-1)+(2*l-1)))*boltzmann(i,j,k,l,beta) for l in range(chi_i)] for k in range(chi_i)] for j in range(chi_i)] for i in range(chi_i)])
    A = np.array([[[[np.exp(beta*h*((2*i-1)+(2*j-1)+(2*k-1)+(2*l-1)))*boltzmann(i,j,k,l,beta) for l in range(chi_i)] for k in range(chi_i)] for j in range(chi_i)] for i in range(chi_i)])
    mag = magnitude(A)
    A = A/mag
    A_tilde = A_tilde/mag
    M1 = np.einsum('ailh,bcji,edjk,fklg',A_tilde,A,np.conjugate(A),np.conjugate(A))
    M2 = np.einsum('ailh,bcji,edjk,fklg',A,A,np.conjugate(A),np.conjugate(A_tilde))
    return 0.5*(M1+M2)



def svd_A(A):
    chi = A.shape[0]
    A_matrix = np.reshape(A,(chi**2,chi**2))
    U,S,VH = np.linalg.svd(np.reshape(A,(chi**2,chi**2)))

    return S

def eig_A(A):
    chi = A.shape[0]
    A_matrix = np.reshape(A,(chi**2,chi**2))
    S = np.linalg.eig(A_matrix)[0]

    return S

# define conjugate transpose
def ct(A):
    return np.conjugate(np.transpose(A))



# identity matrix of size (n,m)
def id(x):
    n,m = x
    M = np.zeros((n,m))
    for i in range(min(n,m)):
        M[i,i]=1
    return M


def magnitude(tensor):
    return np.sqrt(np.sum(tensor*np.conjugate(tensor)))

def expectation_value(A,M):
    F = to_F(A)
    Z = np.einsum('abcdbadc',F)
    return np.einsum('abcdbadc',M)/Z

# For step 1 update

def to_F(A):
    return np.einsum('iqup,jkrq,mlrs,nsuo',A,A,np.conjugate(A),np.conjugate(A))

def to_half_B(A,u,vL,vR):
    return np.einsum('oqil,pnjq,kmop,akl,bmn',A,A,u,vL,vR)

def to_B(A,u,vL,vR):
    half_B = to_half_B(A,u,vL,vR)
    return np.einsum('abij,dcij',half_B,np.conjugate(half_B))

def to_half_Pu(u,vL,vR):
    return np.einsum('mnkj,aml,bni',u,vL,vR)

def to_Pu(u,vL,vR):
    half_Pu = to_half_Pu(u,vL,vR)
    return np.einsum('abponm,ablkji',half_Pu,np.conjugate(half_Pu))

def to_F_tilde(F,u,vL,vR):
    Pu = to_Pu(u,vL,vR)
    return np.einsum('abcdefgh,pijkhabc,onmlgfed',F,Pu,np.conjugate(Pu))


def update_step1(F,u,vL,vR,N_iter,B_old=0,delta=0):
    chi = vR.shape[1]
    chi_p = vR.shape[0]
    for n in range(N_iter):
        Gamma_1 = np.einsum('jklrqpoi,stpq,aso,btr',F,np.conjugate(u),np.conjugate(vL),np.conjugate(vR))
        B = np.einsum('dcijkl,mnjk,ami,bnl',Gamma_1,u,vL,vR)
        B_prime = B + delta*B_old
        Gamma_2 = np.einsum('abijkl,mnba',Gamma_1,np.conjugate(B_prime))
        Gamma_vL = np.einsum('jmnlza,ikmn,akl',Gamma_2,u,vR)
        Gamma_vR = np.einsum('lmnjaz,kimn,akl',Gamma_2,u,vL)
        Gamma_u = np.einsum('mijnab,akm,bln',Gamma_2,vL,vR)
        U,S,VH = np.linalg.svd(np.reshape(Gamma_vL,(chi**2,chi_p)),full_matrices=False)
        vL_new = np.reshape(ct(VH)@ct(U),(chi_p,chi,chi))
        U,S,VH = np.linalg.svd(np.reshape(Gamma_vR,(chi**2,chi_p)),full_matrices=False)
        vR_new = np.reshape(ct(VH)@ct(U),(chi_p,chi,chi))
        U,S,VH = np.linalg.svd(np.reshape(Gamma_u,(chi**2,chi**2)))
        u_new = np.reshape(ct(VH)@ct(U),(chi,chi,chi,chi))
    F_tilde_new = np.einsum('abcd,tsnm,qrij,dto,aqp,csl,brk',B,u,np.conjugate(u),vL,np.conjugate(vL),vR,np.conjugate(vR))
    return u_new,vL_new,vR_new,F_tilde_new

# For step 2 update
def to_B_tilde(B,yL,yR):
    return np.einsum('efgh,ieh,iad,jfg,jbc',B,yL,np.conjugate(yL),yR,np.conjugate(yR))

def to_D(B,yL,yR):
    return np.einsum('cdef,acf,bde',B,yL,yR)

def to_Gamma_yL(B,yL,yR,D_old=0,delta=0):
    D = to_D(B,yL,yR)+delta*D_old
    return 0.5*(np.einsum('cf,adeb,fde',np.conjugate(D),B,yR)+np.conjugate(np.einsum('cf,bdea,fde',np.conjugate(D),B,yR)))

def to_Gamma_yR(B,yL,yR,D_old=0,delta=0):
    D = to_D(B,yL,yR+delta*D_old)
    return 0.5*(np.einsum('ec,dabf,edf',np.conjugate(D),B,yL)+np.conjugate(np.einsum('ec,dbac,edc',np.conjugate(D),B,yL)))



def update_yL(B,yL,yR,N_iter,D_old=0,delta=0):
    for n in range(N_iter):
        Gamma_yL = to_Gamma_yL(B,yL,yR,D_old,delta)
        chi_p = yL.shape[0]
        U,S,VH = np.linalg.svd(np.reshape(Gamma_yL,(chi_p**2,chi_p)),full_matrices=False)

        yL = np.reshape(ct(VH)@ct(U),(chi_p,chi_p,chi_p))
        #print(n,"th yL: ",yL)
    return yL

def update_yR(B,yL,yR,N_iter,D_old=0,delta=0):
    for n in range(N_iter):
        Gamma_yR = to_Gamma_yR(B,yL,yR,D_old,delta)
        chi_p = yR.shape[0]
        U,S,VH = np.linalg.svd(np.reshape(Gamma_yR,(chi_p**2,chi_p)),full_matrices=False)

        yR = np.reshape(ct(VH)@ct(U),(chi_p,chi_p,chi_p))
        #print(n,"th yR: ",yR)
    return yR

def to_C(vL,vR,yL,yR,root_D):
    return np.einsum('fln,ejm,dkn,cim,gef,hcd,gb,ah',vL,np.conjugate(vL),vR,np.conjugate(vR),np.conjugate(yL),np.conjugate(yR),root_D,root_D)



# For step 3 update



def to_Pw(w):
    return np.einsum('akl,aij',w,np.conjugate(w))

def to_C_tilde(C,w):
    Pw = to_Pw(w)
    return np.einsum('abmnop,ijmn,opkl',C,Pw,Pw)

def to_A_prime(C,w):
    return np.einsum('dbijkl,aij,ckl',C,w,np.conjugate(w))

def to_Gamma_w(C,w,A_prime_old=0,delta=0):
    A_prime = to_A_prime(C,w)+delta*A_prime_old
    return np.einsum('zcda,acijkl,dkl',np.conjugate(A_prime),C,np.conjugate(w))


def update_w(C,w,N_iter,A_prime_old=0,delta=0):
    for n in range(N_iter):
        Gamma_w = to_Gamma_w(C,w,A_prime_old,delta)
        chi_p = w.shape[0]
        chi = w.shape[1]
        U,S,VH = np.linalg.svd(np.reshape(Gamma_w,(chi**2,chi_p)),full_matrices=False)

        w = np.reshape(ct(VH)@ct(U),(chi_p,chi,chi))
        #print(n,"th yR: ",yR)
    return w

############################################################
# Renoralize observable

def to_GU(A,vL,vR,u):
    return to_half_B(A,u,vL,vR)

def to_GL(vL):
    return np.einsum('aki,bkj',vL,np.conjugate(vL))

def to_GR(vR):
    return np.einsum('aki,bkj',vR,np.conjugate(vR))

def to_GYL(vL,vR,yR,w,root_D):
    return np.einsum('cjk,eik,fed,bf,aij',np.conjugate(vL),np.conjugate(vR),np.conjugate(yR),root_D,w)

def to_GYR(vL,vR,yL,w,root_D):
    return np.einsum('ejk,cik,fed,fb,aij',np.conjugate(vL),np.conjugate(vR),np.conjugate(yL),root_D,w)

def update_M(M,A,vL,vR,u,yL,yR,root_D,w):
    GU = to_GU(A,vL,vR,u)
    GL = to_GL(vL)
    GR = to_GR(vR)
    GYL = to_GYL(vL,vR,yR,w,root_D)
    GYR = to_GYR(vL,vR,yL,w,root_D)
    return np.einsum('ijklmnop,xqij,utnm,rskl,wvpo,ahxw,fguv,bcqr,edts',M,GU,np.conjugate(GU),GL,GR,GYL,np.conjugate(GYL),GYR,np.conjugate(GYR))

######################## Total update ######################
def update_AM(A,M,chi_p,N1,N2,N3,B_old=0,D_old=0,A_prime_old=0,delta=0,Z_old=1):

    chi = A.shape[0]
    lamb = 0
    # initialize u[i][j][k][l], vL[a][i][j] and vR[a][i][j]
    u = np.reshape(id((chi**2,chi**2)),(chi,chi,chi,chi)) + lamb*(np.random.rand(chi,chi,chi,chi)-0.5)
    vL = np.reshape(id((chi_p,chi**2)),(chi_p,chi,chi)) + lamb*(np.random.rand(chi_p,chi,chi)-0.5)
    vR = np.reshape(id((chi_p,chi**2)),(chi_p,chi,chi)) + lamb*(np.random.rand(chi_p,chi,chi)-0.5)
    # initialize yL[a][b][c] and yR[a][b][c]
    yL = np.reshape(id((chi_p,chi_p**2)),(chi_p,chi_p,chi_p)) + lamb*(np.random.rand(chi_p,chi_p,chi_p)-0.5)
    yR = np.reshape(id((chi_p,chi_p**2)),(chi_p,chi_p,chi_p)) + lamb*(np.random.rand(chi_p,chi_p,chi_p)-0.5)
    #initialize w[a][i][j]
    w =  np.reshape(id((chi_p,chi**2)),(chi_p,chi,chi)) + lamb*(np.random.rand(chi_p,chi,chi)-0.5)

    # step 1 update
    F = to_F(A)

    F_tilde = to_F_tilde(F,u,vL,vR)

    start_time = time.time()
    print("########################################")
    #print("Before: ||F-F_tilde|| = ",magnitude(F-F_tilde))

    i=0
    check = True
    while check:
        #time1 = time.time()

        u,vL,vR,F_tilde_new = update_step1(F,u,vL,vR,1,B_old,int(i < 100)*delta)
        #time2 = time.time()
        #print("Time for updating u,vL,vR",time2-time1)
        #time3 = time.time()
        improvement = magnitude(F_tilde_new-F_tilde)
        error1 = magnitude(F_tilde_new-F)
        print(i,"th ||F_tilde_new-F_tilde|| = ",improvement)
        print(i,"th ||F_tilde_new-F|| = ",error1)
        #print("Time for contraction to get F_tilde: ",time3-time2)
        F_tilde = np.copy(F_tilde_new)
        i += 1
        if i == N1:
            check = False
        if i>30 and improvement < 1e-5:
            check = False
        if error1 < 1e-3:
            check = False



    F_tilde = to_F_tilde(F,u,vL,vR)
    print("########################################")
    print("||F-F_tilde||/||F||= ",magnitude(F-F_tilde)/magnitude(F))
    #print("########################################")
    B = to_B(A,u,vL,vR)



    # step 2 update
    B_tilde = to_B_tilde(B,yL,yR)


    #print("Before: ||B-B_tilde|| = ",magnitude(B-B_tilde))


    i=0
    check = True
    while check:
        yL = update_yL(B,yL,yR,3,D_old,int(i < 100)*delta)
        yR = update_yR(B,yL,yR,3,D_old,int(i < 100)*delta)
        B_tilde_new = to_B_tilde(B,yL,yR)
        improvement = magnitude(B_tilde_new-B_tilde)
        error2 = magnitude(B_tilde_new-B)
        print(i,"th ||B_tilde_new-B_tilde|| = ",improvement)
        print(i,"th ||B_tilde_new-B|| = ",error2)
        B_tilde = np.copy(B_tilde_new)
        i+=1
        if i == N2:
            check = False
        if i>30 and improvement < 1e-5:
            check = False
        if error2 < 1e-3:
            check = False

    B_tilde = to_B_tilde(B,yL,yR)

    print("########################################")
    print("||B-B_tilde||/||B|| = ",magnitude(B-B_tilde)/magnitude(B))
    #print("########################################")


    D = to_D(B,yL,yR)
    U_D,S_D,VH_D = np.linalg.svd(D)
    D = np.diag(S_D)
    root_D = np.sqrt(D)
    yL = np.einsum('ab,bcd',ct(U_D),yL)
    yR = np.einsum('ba,bcd',ct(VH_D),yR)


    C = to_C(vL,vR,yL,yR,root_D)
    # step 3 update

    C_tilde = to_C_tilde(C,w)

    #print("Before: ||C-C_tilde|| = ",magnitude(C-C_tilde))

    i=0
    check = True
    while check:
        w = update_w(C,w,3,A_prime_old,int(i < 100)*delta)
        C_tilde_new = to_C_tilde(C,w)
        improvement = magnitude(C_tilde_new-C_tilde)
        error3 = magnitude(C_tilde_new-C)
        print(i,"th ||C_tilde_new-C_tilde|| = ",improvement)
        print(i,"th ||C_tilde_new-C|| = ",error3)
        C_tilde = np.copy(C_tilde_new)
        i+=1
        if i == N3:
            check = False
        if i>30 and improvement < 1e-5:
            check = False
        if error3 < 1e-3:
            check = False

    C_tilde = to_C_tilde(C,w)
    print("########################################")
    print("||C-C_tilde||/||C|| = ",magnitude(C-C_tilde)/magnitude(C))
    #print("########################################")
    A_prime = to_A_prime(C,w)
    #mag = magnitude(A_prime)
    Z_prime = np.einsum('ijij',A_prime)

    A_renormalized = A_prime/Z_prime
    Z = Z_old*Z_prime


    end_time = time.time()
    error = (magnitude(F-F_tilde),magnitude(B-B_tilde),magnitude(C-C_tilde))
    print("################## summary ######################")
    print("Errors:",magnitude(F-F_tilde)/magnitude(F),magnitude(B-B_tilde)/magnitude(B),magnitude(C-C_tilde)/magnitude(C))
    print("Total time spent for this update_A: ",end_time-start_time)
    print("########################################")
    M_renormalized = update_M(M,A,vL,vR,u,yL,yR,root_D,w)/(Z_prime**4)

    return (A_renormalized,M_renormalized,B,D,A_prime ,error,Z)
