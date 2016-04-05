#coding:utf-8
'''
nie
2016/4/4

This is the implementation of HbEMKM
References:
  - D. Raja Kishor, N. B. Venkateswarlu:
    Hybridization of Expectation-Maximization and K-Means Algorithms for Better Clustering Performance
    http://arxiv.org/abs/1603.07879
'''

from numpy import *
from random import *
import time
seed(9)

def find_k(x,k_centroid):
    dist=sum((x-k_centroid)**2,axis=1)**0.5
    k,=where(dist==max(dist))
    return k[0]

class HbEMKM(object):
    def __init__(self,N,K,data):
        self.N=N
        self.K=K
        self.label=zeros((len(data),1),dtype="int32")
        self.data=data
        self.u=asarray(sample(self.data,self.K))
        self.w=ones((self.K,1))/self.K
        self.sigma=[]
        self.n_j=zeros((self.K,1),dtype="int32")

    def run(self):
        isProgress=True
        num_sample=len(self.data)
        old_change=num_sample
        change=0
        for i in range(num_sample):
            self.label[i]=find_k(self.data[i],self.u)
            self.n_j[self.label[i]]+=1
        self.w=self.n_j/float(self.N)
        #print(self.label)
        self.u=[mean(self.data[where(self.label==i)[0],:],axis=0) if(len(self.data[where(self.label==i)[0],:])>0) else mean(self.data,axis=0) for i in range(self.K)]
        for i in range(self.K):
            if(len(self.data[where(self.label==i)[0],:])>1):
                self.sigma.append(cov(self.data[where(self.label==i)[0],:].T))
            elif(len(self.data[where(self.label==i)[0],:])==1):
                self.sigma.append(identity(self.data.shape[1])*cov(self.data[where(self.label==i)[0],:]))
            else:
                self.sigma.append(identity(self.data.shape[1]))
        #self.sigma=[if(len(self.data[where(self.label==i)[0],:])>1): cov(self.data[where(self.label==i)[0],:].T) elif(len(self.data[where(self.label==i)[0],:])==1): identity(self.data.shape[1])*cov(self.data[where(self.label==i)[0],:]) else: identity(self.data.shape[1]) for i in range(self.K)]
        #print(self.label)
        while( isProgress):
            #E-Step
            for i in range(num_sample):
                P_c_x=zeros((self.K,1))
                for j in range(self.K):
                    #print(self.sigma[j])
                    if(abs(linalg.det(self.sigma[j]))>1e-5):
                        #print(linalg.det(self.sigma[j]))
                        #print(linalg.inv(self.sigma[j]))
                        P_c_x[j]=(linalg.det(self.sigma[j])**(-0.5))*exp((-0.5)*(self.data[i,:]-self.u[j]).T.dot(linalg.inv(self.sigma[j]).dot(self.data[i,:]-self.u[j])))*self.w[j]
                        #print(P_c_x[j])
                    else:
                        P_c_x[j]=1/float(self.K)
                tmp=float(sum(P_c_x))
                P_c_x/=tmp
                
                old=self.label[i]
                self.n_j[self.label[i]]-=1
                #print(self.sigma)
                #print(self.label)
                #print(P_c_x)
                self.label[i]=where(P_c_x==max(P_c_x))[0][0]
                self.n_j[self.label[i]]+=1
                if old!=self.label[i]:
                    change+=1;  
            if(100*abs(old_change-change)>old_change*3 ):
                ## M-Step:compute means u and covariance matrices
                old_change=change
                change=0
                for j in range(self.K):
                    if(abs(linalg.det(self.sigma[j]))<1e-5):
                        self.u[j]=mean(self.data,axis=0)
                    else:
                        tp1=sum(  [(linalg.det(self.sigma[j])**(-0.5))*exp((-0.5)*(self.data[i]-self.u[j]).T.dot(linalg.inv(self.sigma[j]).dot(self.data[i]-self.u[j]))*self.data[i]) for i in range(num_sample)],axis=0)
                        tp2=sum(  [(linalg.det(self.sigma[j])**(-0.5))*exp((-0.5)*(self.data[i]-self.u[j]).T.dot(linalg.inv(self.sigma[j]).dot(self.data[i]-self.u[j]))) for i in range(num_sample)],axis=0)
                        self.u[j]=tp1/float(tp2)
                    
                for i in range(num_sample):
                    old=self.label[i]
                    self.n_j[self.label[i]]-=1
                    self.label[i]=find_k(self.data[i],self.u)
                    self.n_j[self.label[i]]+=1
                    if old!=self.label[i]:
                        change+=1
                self.w=self.n_j/self.N
                self.u=[mean(self.data[where(self.label==i)[0],:],axis=0) if(len(self.data[where(self.label==i)[0],:])>0) else mean(self.data,axis=0) for i in range(self.K)]
                #print(self.label)
                #print(self.data[where(self.label==0)[0],:]);print()
                #print(self.data[where(self.label==1)[0],:]);print()
                #print(self.data[where(self.label==2)[0],:])
                self.sigma=[]
                for i in range(self.K):
                    if(len(self.data[where(self.label==i)[0],:])>1):
                        self.sigma.append(cov(self.data[where(self.label==i)[0],:].T))
                    elif(len(self.data[where(self.label==i)[0],:])==1):
                        self.sigma.append(identity(self.data.shape[1])*cov(self.data[where(self.label==i)[0],:]))
                    else:
                        self.sigma.append(identity(self.data.shape[1]))
                #self.sigma=[if(len(self.data[where(self.label==i)[0],:])>1) cov(self.data[where(self.label==i)[0],:].T) elif(len(self.data[where(self.label==i)[0],:])==1) identity(self.data.shape[1])*cov(self.data[where(self.label==i)[0],:]) else identity(self.data.shape[1]) for i in range(self.K)]
                if(100*abs(old_change-change)>old_change*3):
                    isProgress=True
                    old_change=change
                    change=0
                else:
                    isProgress=False
            else:
                isProgress=False

    def Intracluster_similarity(self):
        #print(self.u)
        S_tra=[1 if(self.n_j[j]==0) else ((1+self.n_j[j])/(1+sum((sum( (self.u[j]-self.data[where(self.label==j)[0],:])**2,axis=1))**0.5)))[0] for j in range(self.K)]
        return S_tra

    def Intercluster_similarity(self):
        return (1+self.N)/( 1+sum(sum((self.u-mean(self.u,axis=0))**2,axis=1)**0.5) )

    def Clustering_fitness(self,lamba=0.5):
        CF=lamba*mean(self.Intracluster_similarity(),axis=0)+(1-lamba)/self.Intercluster_similarity()
        return CF

    def SSE(self):
        SSE=sum([sum(sum( (self.u[j]-self.data[where(self.label==j)[0],:])**2,axis=1)**0.5) for j in range(self.K)])
        return SSE

def test(filename,K):
    rawdata=loadtxt(filename, delimiter=",")
    data=rawdata[:,:-1]
    label=rawdata[:,-1]
    HMK=HbEMKM(len(data),K,data)
    tic=time.time()
    HMK.run()
    end=time.time()
    print 'Intracluster similarity:',
    print(HMK.Intracluster_similarity())
    print("Intercluster similarity: %f"%HMK.Intercluster_similarity())
    print("Clustering fitness:      %f"%HMK.Clustering_fitness())
    print("SSE:                     %f"%HMK.SSE())
    print("run time:                %f"%(end-tic))

if __name__=='__main__':
    print("------------- test data poker-hand-training-true.data----------")
    test("data/poker-hand-training-true.data",10)

    print("------------- test data magic04.data----------")
    #test("data/magic04.data",2)

    print("------------- test data letter-recognition.data----------")
    #test("data/letter-recognition.data")
                

