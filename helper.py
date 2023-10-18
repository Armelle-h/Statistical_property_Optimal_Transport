import numpy as np
import ot #package for optimal transport solver

def f_n(n, seed=9001):
    '''
    input:
        n: int, the sample size
        seed: int the seed number for reproducibility
    output:
        f_n: array of size n where the i th element corresponds to the i th optimal matching between a size n deterministic array equally
             distributed on [0,1] and a size n random array distributed according to a uniform(0,1) distribution
    '''

    np.random.seed(seed) #setting seed for reproducibility
    X=np.random.uniform(size=n)
    Y=np.linspace(1,n,n)/(n+1)
    C=np.zeros((n,n))#cost matrix, C_ij is the distance between X_i and Y_j

    for i in range(n):
        for j in range(n):
            C[i,j]=np.abs(X[i]-Y[j])**2#we're in dimension 1, okay, in higher dimension will need the norm
            
    a=np.ones(n)/n
    b=np.ones(n)/n
    P_test=ot.emd(a, b, C, numItermax=100000, log=False, center_dual=True, numThreads=1, check_marginals=True)
    
    # defining the interpolating function
    f=np.zeros(n+2) 

    sigma=np.zeros(n)
    X_p=np.zeros(n)

    for i in range(n):
        index=np.argwhere(P_test[i,:]>0)
        X_p[index[0,0]]=X[i]


    for i in range(0,n):
        f[i+1]=Y[i]-X_p[i]
        
    return np.sqrt(n)*f #normalizing to compare more easily

def B_t(t_1, n, seed=9001):
    '''
    input:
        t_1: int in ]0,n+2[
        n: int the number of simulations of f_n (which is defined above)
        seed: int the seed number of reproducibility
    output:
        B_t_1: an array of size n where each component is a realization of f_n at time t_1
    '''
    if t_1>=n+2 or t_1<=0:
        print("Warning t_1 not in ]0,1[") #to be changed into a proper warning
              
    np.random.seed(seed) #fixing seed for reproducibility
    t_1=np.random.randint(1,n+1,1)[0] #outputs one random index in range(size(f_n)) excluding t_1=0 and t_1=n+2
    
    B_t_1=np.zeros(n) #the value of the brownian process at time t_1 for different realisations of X

    for i in range(n):
        B_t=f_n(n,i) #for reproducibility we use fixed seeds in [0:99]
        B_t_1[i]=B_t[t_1] 
    
    return B_t_1
    
    
    
    
    
    
    
    