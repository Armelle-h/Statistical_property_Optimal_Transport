import numpy as np
import ot #package for optimal transport solver

def to_two(x,n): #converting 1d coordinates to 2d coordinates
    '''
    input:
            x: float, the value to convert into two under the order (0,0), (0,1),...,(0,n-1),(1,0),...,(n-1,n-1)
            n: int, where the coordinates (i,j) are such that i and j range from 0 to n-1
    output: 
            (i,j), the x th position in the array (0,0), (0,1),...,(0,n-1),(1,0),...,(n-1,n-1) 
    '''
    return(int((x-x%n)/n), x%n)


def f_n_2(n=20, seed=981):
    '''
    input:
            n: int, such that we assign optimaly n**2 points to the grid of size n times n defined as {1/n+1,...,n/n+1} x {1/n+1,...,n/n+1}
    output: 
            f_2_n: a 2xn array such that f[i][j] corresponds to the matching error of (i/n+1, j/n+1) to its assigned 2d uniform random                   variable 
            X_coord,Y_coord: pair of nxn array such that {1/n+1,...,n/n+1} x {1/n+1,...,n/n+1}={X_coord[i][j], Y_coord[i][j]}_{i,j}
    '''
    np.random.seed(seed) #setting seed for reproducibility
    X=np.random.uniform(size=(n**2,2)) 

    x = np.linspace(1,n,n)/(n+1)
    y = np.linspace(1,n,n)/(n+1)

    X_coord, Y_coord = np.meshgrid(x, y) #describes a grid where (X_coord[i][j], Y_coord[i][j]) described the meshed points 

    X_grid=np.array([[X_coord[i][j], Y_coord[i][j]] for j in range(n) for i in range(n)])
    #a grid of the shape  [ (0,0), (0,1), ..., (0,n-1), ... , (n-1,0),...,(n-1,n-1)   ]

    C=np.zeros((n**2,n**2))#cost matrix, C_ij is the distance between X_i and Y_j, of size n**2 times n**2

    for i in range(n**2):
        for j in range(n**2):
            C[i,j]=np.linalg.norm(X[i]-X_grid[j])**2
        
    a=np.ones(n**2)/n**2
    b=np.ones(n**2)/n**2
    P_test=ot.emd(a, b, C, numItermax=100000, log=False, center_dual=True, numThreads=1, check_marginals=True)
    
    f_2=np.zeros((n,n))  #f_2 is defined on the interior of the unit square

    X_p=np.zeros((n**2,2))

    for i in range(n**2):
        index=np.argwhere(P_test[i,:]>0)
    
        X_p[index[0,0]]=X[i]


    for i in range(0,n**2):
        x_f,y_f=to_two(i,n)
        f_2[x_f][y_f]=X_grid[i,0]-X_p[i,0] #only considering the difference of first coordinate as in expectation
        #in optimal transport, expectation of optimal cost is expectation of sum of square of square of norm which is 2 times expectation 
        #of error of the first coordinates, assuming first and second coordinates are identically distributed (which they are)
        
    return n*f_2, X_coord, Y_coord #normalizing f_2 by sqrt(n**2)

def c_padded(X_coord, Y_coord):
    '''
    input: 
            X_coord, Y_coord, a couple of nxn grid such that {1/n+1,...,n/n+1} x {1/n+1,...,n/n+1}={X_coord[i][j], Y_coord[i][j]}_{i,j}
    output: 
            X_coord_padded, Y_coord_padded, a couple of (n+2)x(n+2) grid such that 
            {0,1/n+1,...,n/n+1,1} x {0,1/n+1,...,n/n+1,1}={X_coord_padded[i][j], Y_coord_padded[i][j]}_{i,j}
    '''
    #padding the X_coordinate
    X_coord_temp=np.insert(X_coord, 0, X_coord[0], axis=0)
    X_coord_temp=np.insert(X_coord_temp, 0, X_coord[0], axis=0)
    X_coord_padded=np.insert(X_coord_temp, 0, 0, axis=1)
    X_coord_padded=np.insert(X_coord_padded, np.shape(X_coord_padded)[1], 1, axis=1)
    
    #padding the Y_coordinate
    Y_coord_temp=np.insert(Y_coord.T, 0, Y_coord.T[0], axis=0).T
    Y_coord_temp=np.insert(Y_coord_temp.T, 0, Y_coord_temp.T[0], axis=0).T
    Y_coord_padded=np.insert(Y_coord_temp, 0, 0, axis=0)
    Y_coord_padded=np.insert(Y_coord_padded, np.shape(Y_coord_padded)[0], 1, axis=0)
    
    return X_coord_padded, Y_coord_padded

def simulation_data(n=50, num_occ=10000, seed=6006):
    '''
    input:
            n: int such that we assign optimaly n**2 points to the grid of size n times n defined as {1/n+1,...,n/n+1} x {1/n+1,...,n/n+1}
            num_occ: int the number of realisations of f_n_2 
            seed: int, the seed for reproducibility
    output:
            B: a (num_occ x n x n) array where the B[i,:,:] is the ith realisation of f_n_2  
    '''
    np.random.seed(seed) #fixing seed for reproducibility
    
    B=[] #a list of 2 dimensional numpy arrays where each of these numpy array correspond to a realisation of f_n_2
    #hence if you take the coordinate (i,j) for each of the array in that list, they correspond to a realisation of 
    #the value f_n_2(i,j)

    for i in range(num_occ):
        B_t,_,_=f_n_2(n,i) #for reproducibility we use fixed seeds in [0:n-1].   NB: the seed HAS to change as we want various realizations of f_n
        B.append(B_t)
    return np.array(B)