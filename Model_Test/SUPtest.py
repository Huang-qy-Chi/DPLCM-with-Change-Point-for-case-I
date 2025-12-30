import numpy as np
import scipy.optimize as spo
import torch
from torch import nn
import numpy.random as ndm
from Beta_estimate import Beta_est
from C_estimation import C_est
from I_spline import I_S
from g_deep import g_D
from iteration_deep import Est_deep
from data_generator import generate_case_3




def SUPstat(train_data,test_data,Z_2_train,n_lr,k=5,n_layer=3,n_node=64,n_epoch=200):  
    #1.basic set-ups
    Z_train = train_data['Z']
    U_train = train_data['U']
    De_train = train_data['De']
    g_train = train_data['g_X']
    X_train = train_data['X']
    n = Z_2_train.shape[0]
    r = X_train.shape[1]
    p = 3 
    m = 10 
    tau = 10
    X_test = test_data['X']
    c0 = np.array(0.1*np.ones(m+p), dtype="float32") 
    Beta = -1
    Beta0 = np.array(0, dtype='float32')
    u_value = np.array(np.linspace(0, tau, 50), dtype="float32") 
    Lambda_true = np.sqrt(u_value) 
    nodevec = np.array(np.linspace(0, tau, m+2), dtype="float32")
    m0 = 4 
    nodevec0 = np.array(np.linspace(0, 2, m0+2), dtype="float32")
   
    #2.estimate by DPLCM without change point
    Est_hat = Est_deep(train_data=train_data,X_test=X_test,Beta=Beta,Beta0=Beta0,n_layer=n_layer,\
                    n_node=n_node,n_lr=n_lr,n_epoch=n_epoch,nodevec=nodevec,m=m,c0=c0)

    Betat = Est_hat['Beta']
    Lambdat = I_S(m,Est_hat['c'],U_train,nodevec)
    gt = Est_hat['g_train']

    #3.calculate the SUP statistics functional with indicator
    Lam = Lambdat * np.exp(Z_train * Betat + gt) 
    St = np.exp(-Lam)
    X_train1 = X_train.T
    Z2min = np.min(Z_2_train)
    Z2max = np.max(Z_2_train)
    Xmin = np.min(X_train)
    Xmax = np.max(X_train)
    zeta11 = np.linspace(Z2min,Z2max,k)
    thres = np.linspace(Xmin,Xmax,k)
    SUPrecord = []
    for u in zeta11:  #all zeta
        for v in thres:  #all Gateaux derivative of threshold for indicator
            for w in range(r):  #all dimension for X_train
                #Qy = (-De_train * np.log(1 - np.exp(-Lam) + 1e-5) + (1 - De_train) * Lam )* (Z_2_train>u)
                #Qy1 = np.vstack((Qy,Qy))
                Qy = ((De_train * (St/(1-St)) - (1 - De_train)) * Lam )*(Z_2_train>u)
                Qy2 = np.diag(Qy)
                Qy22 = Qy2@Qy2
                func = X_train1[w,:]<v
                ZX = np.vstack((Z_train,func))
                q = ZX.shape[0]
                UZX = ZX@Qy.T
                ZXD = ZX
                #SigmaZX = np.linalg.inv(Qy**2*ZXD@ZXD.T+1e-8*np.eye(2))
                SigmaZX = np.linalg.inv(ZXD@Qy22@ZXD.T+1e-6*np.eye(q))
                #SUPstat = np.trace(UZX.T@SigmaZX@UZX)
                SUPrecord.append(UZX.T@SigmaZX@UZX)

    #func = X_train1[1,:]<1.5
    #ZX = np.vstack((Z_train,func))
    #UZX = ZX@Qy.T
    #ZXD = ZX
    #SigmaZX = np.linalg.inv(ZXD@Qy22@ZXD.T)
    #SUPstat = (UZX.T@SigmaZX@UZX)

    #4.find the SUP statistics
    SUPstat = np.max(SUPrecord)
    return SUPstat



def SUPtest(train_data,test_data,alpha=0.05,n_lr=4e-4,n_layer=3,n_node=64,n_epoch=200,k=5,B=50):
    Z_train = train_data['Z']
    U_train = train_data['U']
    De_train = train_data['De']
    g_train = train_data['g_X']
    X_train = train_data['X']
    Z_2_train = train_data['Z_2']
    n = Z_2_train.shape[0]
    #1.test statistics
    SUPstat1 = SUPstat(train_data,test_data,Z_2_train,n_lr,k,n_layer,n_node,n_epoch)
    #2.permutation quantile
    SUPpermu = []
    for b in range(B):
        Z_2_train = ndm.choice(Z_2_train, size = n, replace=False)
        SUPQUAN = SUPstat(train_data,test_data,Z_2_train,n_lr,k,n_layer,n_node,n_epoch)
        SUPpermu.append(SUPQUAN)
    SUPpermu.sort()
    Quantile = SUPpermu[int(B*(1-alpha))]
    #3.decision
    Decision = 0
    if(SUPstat1>Quantile):
        Decision = 1
    #4.quasi p-value
    pquan = np.abs(SUPpermu-SUPstat1)
    loc1 = np.argmin(pquan)
    pvalue = 1-(loc1+1)/B
    return{
        'Decision': Decision,
        'SUPstat': SUPstat1,
        'Quantile': Quantile,
        'p_value': pvalue
    }


#################################################################################################################


def SUPstat1(train_data,test_data,Z_2_test,n_lr,k=5,n_layer=3,n_node=64,n_epoch=200):  
    #1.basic set-ups
    Z_train = train_data['Z']
    U_train = train_data['U']
    De_train = train_data['De']
    g_train = train_data['g_X']
    X_train = train_data['X']
    Z_2_train = train_data['Z_2']
    n = Z_2_train.shape[0]
    r = X_train.shape[1]
    p = 3 
    m = 10 
    tau = 10
    X_test = test_data['X']
    c0 = np.array(0.1*np.ones(m+p), dtype="float32") 
    Beta = -1
    Beta0 = np.array(0, dtype='float32')
    u_value = np.array(np.linspace(0, tau, 50), dtype="float32") 
    Lambda_true = np.sqrt(u_value) 
    nodevec = np.array(np.linspace(0, tau, m+2), dtype="float32")
    m0 = 4 
    nodevec0 = np.array(np.linspace(0, 2, m0+2), dtype="float32")
   
    #2.estimate by DPLCM without change point
    Est_hat = Est_deep(train_data=train_data,X_test=X_test,Beta=Beta,Beta0=Beta0,n_layer=n_layer,\
                    n_node=n_node,n_lr=n_lr,n_epoch=n_epoch,nodevec=nodevec,m=m,c0=c0)

    Betat = Est_hat['Beta']
    #Lambdat = I_S(m,Est_hat['c'],U_train,nodevec)
    gt = Est_hat['g_train']
    gt1 = Est_hat['g_test']
    #3.calculate the SUP statistics functional with indicator
    Z_test = test_data['Z']
    U_test = test_data['U']
    De_test = test_data['De']
    g_test = test_data['g_X']
    X_test = test_data['X']
    Lambdat = I_S(m,Est_hat['c'],U_test,nodevec)
    Lam = Lambdat * np.exp(Z_test * Betat + gt1) 
    St = np.exp(-Lam)
    X_test1 = X_test.T
    Z2min = np.min(Z_2_test)
    Z2max = np.max(Z_2_test)
    Xmin = np.min(X_test)
    Xmax = np.max(X_test)
    zeta11 = np.linspace(Z2min,Z2max,k)
    thres = np.linspace(Xmin,Xmax,k)
    SUPrecord = []
    for u in zeta11:  #all zeta
        for v in thres:  #all Gateaux derivative of threshold for indicator
            for w in range(r):  #all dimension for X_train
                #Qy = (-De_train * np.log(1 - np.exp(-Lam) + 1e-5) + (1 - De_train) * Lam )* (Z_2_train>u)
                #Qy1 = np.vstack((Qy,Qy))
                Qy = ((De_test * (St/(1-St)) - (1 - De_test)) * Lam )*(Z_2_test>u)
                Qy2 = np.diag(Qy)
                Qy22 = Qy2@Qy2
                func = X_test1[w,:]<v
                ZX = np.vstack((Z_test,func))
                q = ZX.shape[0]
                UZX = ZX@Qy.T
                ZXD = ZX
                #SigmaZX = np.linalg.inv(Qy**2*ZXD@ZXD.T+1e-8*np.eye(2))
                SigmaZX = np.linalg.inv(ZXD@Qy22@ZXD.T+1e-6*np.eye(q))
                #SUPstat = np.trace(UZX.T@SigmaZX@UZX)
                SUPrecord.append(UZX.T@SigmaZX@UZX)

    #func = X_train1[1,:]<1.5
    #ZX = np.vstack((Z_train,func))
    #UZX = ZX@Qy.T
    #ZXD = ZX
    #SigmaZX = np.linalg.inv(ZXD@Qy22@ZXD.T)
    #SUPstat = (UZX.T@SigmaZX@UZX)

    #4.find the SUP statistics
    SUPstat = np.max(SUPrecord)
    return SUPstat




def SUPtest1(train_data,test_data,alpha=0.05,n_lr=4e-4,n_layer=3,n_node=64,n_epoch=200,k=5,B=50):
    Z_train = train_data['Z']
    U_train = train_data['U']
    De_train = train_data['De']
    g_train = train_data['g_X']
    X_train = train_data['X']
    Z_2_train = train_data['Z_2']
    n = Z_2_train.shape[0]
    #1.test statistics
    Z_2_test = test_data['Z_2']
    SUPstat1 = SUPstat(train_data,test_data,Z_2_test,n_lr,k,n_layer,n_node,n_epoch)
    #2.permutation quantile
    SUPpermu = []
    for b in range(B):
        Z_2_test = ndm.choice(Z_2_test, size = n, replace=False)
        SUPQUAN = SUPstat(train_data,test_data,Z_2_test,n_lr,k,n_layer,n_node,n_epoch)
        SUPpermu.append(SUPQUAN)
    SUPpermu.sort()
    Quantile = SUPpermu[int(B*(1-alpha))]
    #3.decision
    Decision = 0
    if(SUPstat1>Quantile):
        Decision = 1
    #4.quasi p-value
    pquan = np.abs(SUPpermu-SUPstat1)
    loc1 = np.argmin(pquan)
    pvalue = 1-(loc1+1)/B
    return{
        'Decision': Decision,
        'SUPstat': SUPstat1,
        'Quantile': Quantile,
        'p_value': pvalue
    }


