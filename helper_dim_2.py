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
            C[i,j]=np.linalg.norm(X[i]-Y[j])**2#we're in dimension 1, okay, in higher dimension will need the norm
            
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