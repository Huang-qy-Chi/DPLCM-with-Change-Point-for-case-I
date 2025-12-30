
import numpy as np
import numpy.random as ndm
import math

def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_case_2(n, corr, Theta, zeta):
    Z = ndm.binomial(1, 0.5, n) #parametric
    Z_2 = ndm.normal(loc=2, scale=1, size=n) #change point
    Z_2 = np.clip(Z_2, 1.5, 2.5)
    ZC = np.vstack((Z, Z*(Z_2>zeta)))
    #nonparametric X
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    def multivariatet(mu,Sigma,N,M):
        d = len(Sigma)
        g = np.tile(np.random.gamma(N/2,1/2,M),(d,1)).T
        Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
        return mu + Z/np.sqrt(g/N)
    
    X = multivariatet(mean,cov,5,n)
    X = np.clip(X, 0, 2) #t-distributed with [0,2]
    pi = math.pi
    g_X = X[:,0]**2/2 + 2*np.log(X[:,1]+1)/5 \
        + 3*np.sqrt(X[:,2])/10 + np.exp(X[:,3])/5 + X[:,4]**3/10 - 1.62 
    h_X = np.sin(2*pi*X[:,0]) + np.exp(X[:,1])/5 \
        + 3*np.sqrt(X[:,2])/5 + np.log(X[:,3]+1)/3 \
            + X[:,4]**2/3 - 1.38  
    #g_X = -X[:,0]/2-X[:,1]/3 - X[:,2]/4-X[:,3]/5 \
             #- X[:,4]/6 + 0.94
    #g_X = 0
    #h_X = X[:,0]/3+X[:,1]/4 + X[:,2]/5+X[:,3]/6 \
             #+ X[:,4]/7 - 0.71
    Ind = (Z_2>zeta)
    g_X_C = g_X+h_X*Ind
    Y = ndm.rand(n)  #U(0,1)
    T = (-1 * np.log(Y) * np.exp(-ZC.T @ Theta - g_X_C)) ** 2 #Lambda(t)=np.sqrt(t)
    U = uniform_data(n, 0, 2) #set tau=10
    De = (T <= U)  #censoring indicator
    return {
        'Z': np.array(Z, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'T': np.array(T, dtype='float32'),
        'U': np.array(U, dtype='float32'),
        'De': np.array(De, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32'),
        'h_X': np.array(h_X, dtype='float32'),
        'g_X_C': np.array(g_X_C, dtype='float32'),
        'Z_2': np.array(Z_2, dtype = 'float32')
    }




def generate_case_22(n, corr, Theta, zeta):
    Z = ndm.binomial(1, 0.5, n) #parametric
    Z_2 = ndm.normal(loc=2, scale=1, size=n) #change point
    Z_2 = np.clip(Z_2, 1.5, 2.5)
    ZC = np.vstack((Z, Z*(Z_2>zeta)))
    #nonparametric X
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    def multivariatet(mu,Sigma,N,M):
        d = len(Sigma)
        g = np.tile(np.random.gamma(N/2,1/2,M),(d,1)).T
        Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
        return mu + Z/np.sqrt(g/N)
    
    X = multivariatet(mean,cov,5,n)
    X = np.clip(X, 0, 2) #t-distributed with [0,2]
    pi = math.pi
    g_X = X[:,0]**2/2 + 2*np.log(X[:,1]+1)/5 \
        + 3*np.sqrt(X[:,2])/10 + np.exp(X[:,3])/5 + X[:,4]**3/10 - 1.62 
    #h_X = np.sin(2*pi*X[:,0]) + np.exp(X[:,1])/5 \
        #+ 3*np.sqrt(X[:,2])/5 + np.log(X[:,3]+1)/3 \
            #+ X[:,4]**2/3 - 1.38  
    #g_X = -X[:,0]/2-X[:,1]/3 - X[:,2]/4-X[:,3]/5 \
             #- X[:,4]/6 + 0.94
    #g_X = 0
    #h_X = X[:,0]/3+X[:,1]/4 + X[:,2]/5+X[:,3]/6 \
             #+ X[:,4]/7 - 0.71
    h_X = X[:,0]**2/2 + 2*(X[:,1])**2 \
        + 3*np.exp(X[:,2])/5 + np.sqrt(X[:,3]+1) + np.log(X[:,4]+1) -6.021 
    Ind = (Z_2>zeta)
    g_X_C = g_X+h_X*Ind
    Y = ndm.rand(n)  #U(0,1)
    T = (-1 * np.log(Y) * np.exp(-ZC.T @ Theta - g_X_C)) ** 2 #Lambda(t)=np.sqrt(t)
    U = uniform_data(n, 0, 2) #set tau=10
    De = (T <= U)  #censoring indicator
    return {
        'Z': np.array(Z, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'T': np.array(T, dtype='float32'),
        'U': np.array(U, dtype='float32'),
        'De': np.array(De, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32'),
        'h_X': np.array(h_X, dtype='float32'),
        'g_X_C': np.array(g_X_C, dtype='float32'),
        'Z_2': np.array(Z_2, dtype = 'float32')
    }

def generate_case_21(n, corr, Theta, zeta):
    Z = ndm.binomial(1, 0.5, n) #parametric
    Z_2 = ndm.normal(loc=2, scale=1, size=n) #change point
    Z_2 = np.clip(Z_2, 1.5, 2.5)
    ZC = np.vstack((Z, Z*(Z_2>zeta)))
    #nonparametric X
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    def multivariatet(mu,Sigma,N,M):
        d = len(Sigma)
        g = np.tile(np.random.gamma(N/2,1/2,M),(d,1)).T
        Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
        return mu + Z/np.sqrt(g/N)
    
    X = multivariatet(mean,cov,5,n)
    X = np.clip(X, 0, 2) #t-distributed with [0,2]
    pi = math.pi
    g_X = X[:,0]**2/2 + 2*np.log(X[:,1]+1)/5 \
        + 3*np.sqrt(X[:,2])/10 + np.exp(X[:,3])/5 + X[:,4]**3/10 - 1.62 
    h_X = np.sin(2*pi*X[:,0]) + np.exp(X[:,1])/5 \
        + 3*np.sqrt(X[:,2])/5 + np.log(X[:,3]+1)/3 \
            + X[:,4]**2/3 - 1.38  
    #g_X = -X[:,0]/2-X[:,1]/3 - X[:,2]/4-X[:,3]/5 \
             #- X[:,4]/6 + 0.94
    #g_X = 0
    #h_X = X[:,0]/3+X[:,1]/4 + X[:,2]/5+X[:,3]/6 \
             #+ X[:,4]/7 - 0.71
    Ind = (Z_2>zeta)
    g_X_C = g_X+h_X*Ind
    Y = ndm.rand(n)  #U(0,1)
    T = (-5 * np.log(Y) * np.exp(-ZC.T @ Theta - g_X_C)) ** 2 #Lambda(t)=np.sqrt(t)/5
    U = uniform_data(n, 0, 10) #set tau=10
    De = (T <= U)  #censoring indicator
    return {
        'Z': np.array(Z, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'T': np.array(T, dtype='float32'),
        'U': np.array(U, dtype='float32'),
        'De': np.array(De, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32'),
        'h_X': np.array(h_X, dtype='float32'),
        'g_X_C': np.array(g_X_C, dtype='float32'),
        'Z_2': np.array(Z_2, dtype = 'float32')
    }
