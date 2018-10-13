from TNR import *
import os
import pickle


class Model():
    def __init__(self,htemp,chi,coarsensteps):
        self.htemp=htemp
        self.chi=chi
        self.coarsensteps=coarsensteps

    def prestep(self):
        d=int(np.sqrt(self.htemp.shape[0]))
        Ainit=expm(-0.002*self.htemp).reshape(d,d,d,d)
        Asplit_tensors,A00=Asplit(Ainit,self.chi)
        coarsen_list=[{'A':A00}]
        for i in range(self.coarsensteps):
            Aold=coarsen_list[i]['A']
            coarsen_list[i],Acoarse=coarsen(Aold,self.chi)
            coarsen_list.append({'A':Acoarse})
            if coarsen_list[i]['err']>1e-3:
                break
        print()
        self.coarsen_list=coarsen_list
        self.Asplit_tensors=Asplit_tensors
        self.tensors_list=coarsen_list[-1:]

    def run_RG(self,RGsteps):
        i0=len(self.tensors_list)-1
        for i in range(RGsteps):
            print('RG step: ',i0+i)
            Aold=self.tensors_list[i0+i]['A']
            self.tensors_list[i0+i],Anew=doTNR(Aold,self.chi)
            self.tensors_list.append({'A':Anew})
            print('A.shape=',Anew.shape)
            print('chiv=',self.tensors_list[i0+i]['vl'].shape[2])
            print()

def ising1D():
    J=1
    htemp=KP(sz,sz)+J*(KP(sx,s0)+KP(s0,sx))/2
    chi={'w':6,'y':8,'u':6,'v':6,'k':8,'q':36}
    coarsensteps=12
    return htemp,chi,coarsensteps

def potts1D():
    J=1
    htemp=-KP(R,R.conj())-J*(KP(M,np.eye(3))+KP(np.eye(3),M))/2
    htemp+=htemp.T.conj()
    chi={'w':12,'y':9,'u':9,'v':9,'k':15,'q':66}
    coarsensteps=9
    return htemp,chi,coarsensteps

def z3z31D():
    htemp=-KP(R,R.conj())-KP(M,M)
    htemp+=htemp.T.conj()
    chi={'w':18,'y':18,'u':18,'v':18,'k':15,'q':66}
    coarsensteps=15
    return htemp,chi,coarsensteps

if __name__ == "__main__":
    # import sys
    # orig_stdout = sys.stdout

    # with open('out.txt', 'w') as f:
        # sys.stdout=f
    model=Model(*z3z31D())
    model.prestep()
    model.run_RG(RGsteps=5)

    # sys.stdout = orig_stdout


    data={'chi':model.chi,'tensors_list':model.tensors_list,'Asplit_tensors':model.Asplit_tensors,'coarsen_list':model.coarsen_list}
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)