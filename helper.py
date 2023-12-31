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
        
    return np.sqrt(n)*f #normalizing

def B_t(n, num_occ=10000,seed=9001):
    '''
    input:
        n: int, where n+2 is the number of interpolating points of f_n
        num_occ: int, the number of realisations of f_n
        seed: int the seed number of reproducibility
    output:
        B: an array of size n*n where each row is a realization of f_n
    '''   
    np.random.seed(seed) #fixing seed for reproducibility
    
    B=[] #a list of numpy arrays where the ith row corresponds to the ith realisation of the brownian bridge and 
    #where column t_1 corresponds to the realisation of f_n(t_1)

    for i in range(num_occ):
        B_t=f_n(n,i) #for reproducibility we use fixed seeds in [0:n-1].   NB: the seed HAS to change as we want various realizations of f_n
        B.append(B_t)    
    return B 

def t_cov(s,n):
    '''
    input:
        s: int where s is in {0,...,n+1}
        n: int where n+2 is the number of interpolating points of our brownian bridge
    output:
        new_t: numpy array where new_t[i] is the covariance between B_s and _{i/n+2}, {B_t}_t being a brownian bridge
    '''
    #write proper warning
    if s<0 or s>n+1:
        print("issue, s should be between 0 and ", n+1)
    
    t=np.linspace(0,1,n+2)
    s_norm=s/(n+2)
    list_1=(1-s_norm)*t[t<s_norm]
    list_2=s_norm*(1-t[t>s_norm])
    new_t=np.concatenate((list_1, list_2))
    return new_t 

    
    
    
    
    
    