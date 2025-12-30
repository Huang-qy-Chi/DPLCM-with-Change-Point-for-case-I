import numpy as np
import scipy.optimize as spo

def g_LCP3(train_data,X_test,Z_2_test,Lambda_U,Theta,Theta0,zeta0):
    #input data into torch.tensor
    De_train = train_data['De']
    Z_train = train_data['Z']
    Z_2_train = train_data['Z_2']
    X_train = train_data['X']
    
    X_train1 = np.hstack((np.ones((X_train.shape[0],1)),X_train))
    n = X_train1.shape[0]
    d = X_train1.shape[1]
    ind = (Z_2_train>zeta0)
    for i in range(d-1):
        ind = np.vstack((ind,Z_2_train>2))
    ind = ind.T
    ZC1 = np.hstack((X_train1,X_train1*ind))
    X_test1 = np.hstack((np.ones((X_test.shape[0],1)),X_test))
    p = ZC1.shape[1]
    def TF(*args):  
        ZC0 = np.vstack((Z_train,Z_train*(Z_2_train>zeta0)))
        X_train1 = np.hstack((np.ones((X_train.shape[0],1)),X_train))
        n = X_train1.shape[0]
        d = X_train1.shape[1]
        ind = (Z_2_train>zeta0)
        for i in range(d-1):
            ind = np.vstack((ind,Z_2_train>zeta0))
        ind = ind.T
        g_X = ZC0.T@Theta0 
        X_train1 = np.hstack((np.ones((X_train.shape[0],1)),X_train))
        XC = np.hstack((X_train1,X_train1*ind))
        Lam = Lambda_U * np.exp(XC @ args[0] + g_X)
        Loss_F = np.mean(-De_train * np.log(1 - np.exp(-Lam) + 1e-5) + (1 - De_train) * Lam)
        return Loss_F
    result = spo.minimize(TF,np.zeros(p),method='SLSQP') #nonconvex optimaization
    res_cor = result['x']

    #calculate g(X) and h(X) for both train and test data
    
    g_train = X_train1@res_cor[0:d]
    h_train = X_train1@res_cor[d:p] 
    g_test = X_test1@res_cor[0:d]
    h_test = X_test1@res_cor[d:p]
    #Z_2_train = train_data['Z_2']
    Res_train = g_train+h_train*(Z_2_train>zeta0)
    Res_test = g_test+h_test*(Z_2_test>zeta0)
    return {
        'g_train': g_train,
        'h_train': h_train,
        'g_test': g_test,
        'h_test': h_test,
        'Res_train': Res_train,
        'Res_test': Res_test
    }


def g_LCP(train_data,X_test,Z_2_test,Lambda_U,Theta,Theta0,zeta0):
    #input data into torch.tensor
    De_train = train_data['De']
    Z_train = train_data['Z']
    Z_2_train = train_data['Z_2']
    X_train = train_data['X']
    
    X_train1 = np.hstack((np.ones((X_train.shape[0],1)),X_train))
    n = X_train1.shape[0]
    d = X_train1.shape[1]
    ind = (Z_2_train>zeta0)
    for i in range(d-1):
        ind = np.vstack((ind,Z_2_train>zeta0))
    ind = ind.T
    ZC0 = np.vstack((Z_train,Z_train*(Z_2_train>zeta0)))
    ZC0 = ZC0.T
    ZC1 = np.hstack((X_train1,X_train1*ind))
    X_test1 = np.hstack((np.ones((X_test.shape[0],1)),X_test))
    ZXC1 = np.hstack((ZC0,ZC1))
    p = ZXC1.shape[1]
    def TF(*args):  
        ZC0 = np.vstack((Z_train,Z_train*(Z_2_train>zeta0)))
        ZC0 = ZC0.T
        X_train1 = np.hstack((np.ones((X_train.shape[0],1)),X_train))
        n = X_train1.shape[0]
        d = X_train1.shape[1]
        ind = (Z_2_train>zeta0)
        for i in range(d-1):
            ind = np.vstack((ind,Z_2_train>2))
        ind = ind.T
        #g_X = ZC0.T@Theta0 
        #X_train1 = np.hstack((np.ones((X_train.shape[0],1)),X_train))
        XC = np.hstack((X_train1,X_train1*ind))
        ZXC = np.hstack((ZC0,XC))
        Lam = Lambda_U * np.exp(ZXC @ args[0])
        Loss_F = np.mean(-De_train * np.log(1 - np.exp(-Lam) + 1e-5) + (1 - De_train) * Lam)
        return Loss_F
    result = spo.minimize(TF,np.zeros(p),method='SLSQP') #nonconvex optimaization
    res_cor = result['x']

    #calculate g(X) and h(X) for both train and test data
    Theta0 = res_cor[0:2]
    g_train = X_train1@res_cor[2:d+2]
    h_train = X_train1@res_cor[d+2:p+2] 
    g_test = X_test1@res_cor[2:d+2]
    h_test = X_test1@res_cor[d+2:p+2]
    #Z_2_train = train_data['Z_2']
    Res_train = g_train+h_train*(Z_2_train>zeta0)
    Res_test = g_test+h_test*(Z_2_test>zeta0)
    return {
        'Theta': Theta0,
        'g_train': g_train,
        'h_train': h_train,
        'g_test': g_test,
        'h_test': h_test,
        'Res_train': Res_train,
        'Res_test': Res_test
    }


















