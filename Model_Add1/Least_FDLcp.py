import torch
from torch import nn
import numpy as np

#for the score and information
def LFDLCP(train_data,Lambda_U,Res_train,Theta,zeta):
    X_train = train_data['X']
    Z_train = train_data['Z']
    Z_2_train = train_data['Z_2']
    Z_train = np.vstack((Z_train, Z_train*(Z_2_train>zeta)))
    Z_train = Z_train.T
    #Z_1_train = Z1
    De_train = train_data['De']
    X_U = np.c_[train_data['X'], train_data['U']]
    h_v = Lambda_U * np.exp(Z_train@Theta + Res_train)   #CUMULATIVE HAZARD
    Q_y = h_v * (De_train * np.exp(-h_v)/(1-np.exp(-h_v)+1e-8) - (1-De_train))
    S_y = np.exp(-h_v)  #survival

    X_train1 = np.hstack((np.ones((X_train.shape[0],1)),X_train))
    n = X_train1.shape[0]
    d = X_train1.shape[1]
    ind = (Z_2_train>zeta)
    for i in range(d-1):
        ind = np.vstack((ind,Z_2_train>2))
    ind = ind.T
    ZX = np.hstack((Z_train,X_train1,X_train1*ind))
    RX = Q_y**2
    LFD = np.mean(np.diag(RX)@ZX, axis = 0)/np.mean(RX)
    #Info = (ZX-LFD).T@np.diag(RX)@(ZX-LFD)
    Info = (ZX-LFD).T@np.diag(RX)@(ZX-LFD)/n
    dinfo = Info.shape[0]
    minvar = np.linalg.inv(Info + np.eye(dinfo)*1e-6)/n
    vars = np.diag(minvar)
    se1 = np.sqrt(vars[0])
    se2 = np.sqrt(vars[1])

    

    
    return {
        'Infobound': minvar,
        'se1': se1,
        'se2': se2,
        'LFD': LFD,
        'Q_y': Q_y,
    }
