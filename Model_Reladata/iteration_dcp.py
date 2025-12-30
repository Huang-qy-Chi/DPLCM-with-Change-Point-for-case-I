import numpy as np
#from Beta_estimate import Beta_est
from Theta_estimate import Theta_est
from CCP_estimation import CCP_est
from I_spline import I_S
from g_dcp import g_DCP1
from zeta_estimate import zeta_est
from g_dcp import g_DCP3
from g_dcp import g_DCP3H

def Est_dcp(train_data,X_test,Z_2_test,Theta,Theta0,zeta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0):
    Z_train = train_data['Z']
    U_train = train_data['U']
    De_train = train_data['De']
    X_train = train_data['X']
    g_train_true = train_data['g_X']
    Theta0 = np.array([Theta0])
    Lambda_U = I_S(m,c0,U_train,nodevec)
    C_index = 0
    Z_2_train = train_data['Z_2']
    zeta0 = np.mean(Z_2_train)
    #zeta0 = zeta0
    Z_2_train = train_data['Z_2']
    n1 = len(Z_2_train)   #sample size of train data
    n2 = len(Z_2_test)   #sample size of test data
    locx_tra1 = np.where(Z_2_train>zeta0)  #Z_2>zeta in train data
    locx_tra0 = np.where(Z_2_train<=zeta0)
    locx_tes1 = np.where(Z_2_test>zeta0)  #Z_2>zeta in test data
    locx_tes0 = np.where(Z_2_test<=zeta0)
    ZC = np.vstack((Z_train,Z_train*(Z_2_train>zeta0)))
    ZC0 = ZC.T[locx_tra0]
    ZC1 = ZC.T[locx_tra1]
    Lambda_U0 = Lambda_U[locx_tra0]
    Lambda_U1 = Lambda_U[locx_tra1]
    X_test0 = X_test[locx_tes0]
    X_test1 = X_test[locx_tes1]
    for loop in range(100):
        print('Loop=',loop)
        Z_2_train = train_data['Z_2']
        n1 = len(Z_2_train)   #sample size of train data
        n2 = len(Z_2_test)   #sample size of test data
        locx_tra1 = np.where(Z_2_train>zeta0)  #Z_2>zeta in train data
        locx_tra0 = np.where(Z_2_train<=zeta0)
        locx_tes1 = np.where(Z_2_test>zeta0)  #Z_2>zeta in test data
        locx_tes0 = np.where(Z_2_test<=zeta0)
        ZC = np.vstack((Z_train,Z_train*(Z_2_train>zeta0)))
        ZC0 = ZC.T[locx_tra0]
        #print(ZC0.shape)
        #ZC0 = ZC0.T
        ZC1 = ZC.T[locx_tra1]
        #print(ZC1.shape)
        #ZC1 = ZC1.T
        Lambda_U0 = Lambda_U[locx_tra0]
        Lambda_U1 = Lambda_U[locx_tra1]
        X_test0 = X_test[locx_tes0]
        X_test1 = X_test[locx_tes1]

        #input data into torch.tensor
        Z_train0 = (Z_train[locx_tra0])
        X_train0 = (X_train[locx_tra0,:])
        U_train0 = (U_train[locx_tra0])
        De_train0 = (De_train[locx_tra0])
        g_train_true0 = (g_train_true[locx_tra0])
        #Z_2_train0 = (Z_2_train[locx_tra0])
        #Lambda_U0 = (Lambda_U0)
        #X_test0 = (X_test0)
        #ZC0 = torch.Tensor(ZC0)

        Z_train1 = (Z_train[locx_tra1])
        X_train1 = (X_train[locx_tra1,:])
        U_train1 = (U_train[locx_tra1])
        De_train1 = (De_train[locx_tra1])
        g_train_true1 = (g_train_true[locx_tra1])
        #Z_2_train1 = (Z_2_train[locx_tra1])
        #Lambda_U1 = torch.Tensor(Lambda_U1)   
        #X_test1 = torch.Tensor(X_test1)
        #ZC1 = torch.Tensor(ZC1)
        g_X_0 = g_DCP1(ZC0,X_train0,U_train0,De_train0,g_train_true0,Lambda_U0,X_test0,\
                        Theta,Theta0,n_layer,n_node,n_lr,n_epoch)
        g_train0 = g_X_0['g_train']
        g_test0 = g_X_0['g_test']
        g_X_1 = g_DCP1(ZC1,X_train1,U_train1,De_train1,g_train_true1,Lambda_U1,X_test1,\
                        Theta,Theta0,n_layer,n_node,n_lr,n_epoch)
        g_train1 = g_X_1['g_train']
        g_test1 = g_X_1['g_test']
        #reorganize the data
        g_train = np.array(np.zeros(n1))
        g_train[locx_tra0] = g_train0
        g_train[locx_tra1] = g_train1

        g_test = np.array(np.zeros(n2))
        g_test[locx_tes0] = g_test0
        g_test[locx_tes1] = g_test1
        c1 = CCP_est(m,U_train,De_train,Z_train,Theta0,g_train,Z_2_train,zeta0,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        Theta1 = Theta_est(De_train,Z_train,Z_2_train,Lambda_U,g_train,zeta0)
        zeta1 = zeta_est(De_train,Z_train,Z_2_train,g_train,Lambda_U,Theta1)
        if (np.linalg.norm(Theta0-Theta1,ord=2) <= 1e-5):  #too strict, L1 norm
            C_index = 1
            break
        c0 = c1
        Theta0 = Theta1
        zeta0 = zeta1
        #print(loop)
        #print(Theta1)
        #print(zeta1)
    return {
        'g_train': g_train,
        'g_test': g_test,
        'c': c1,
        'Theta': Theta1,
        'zeta': zeta1,
        'C_index': C_index,
    }



############################################################################################################################

def Est_dcp1(train_data,X_test,Z_2_test,Theta,Theta0,zeta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0):
    Z_train = train_data['Z']
    U_train = train_data['U']
    De_train = train_data['De']
    X_train = train_data['X']
    g_train_true = train_data['g_X']
    Theta0 = np.array([Theta0])
    Lambda_U = I_S(m,c0,U_train,nodevec)
    C_index = 0
    Z_2_train = train_data['Z_2']
    zeta0 = np.mean(Z_2_train)
    #zeta0 = zeta0
    #Z_2_train = train_data['Z_2']
    n1 = len(Z_2_train)   #sample size of train data
    n2 = len(Z_2_test)   #sample size of test data
    locx_tra1 = np.where(Z_2_train>zeta0)  #Z_2>zeta in train data
    locx_tra0 = np.where(Z_2_train<=zeta0)
    locx_tes1 = np.where(Z_2_test>zeta0)  #Z_2>zeta in test data
    locx_tes0 = np.where(Z_2_test<=zeta0)
    ZC = np.vstack((Z_train,Z_train*(Z_2_train>zeta0)))
    #ZC0 = ZC.T[locx_tra0]
    #ZC1 = ZC.T[locx_tra1]
    ZC = ZC.T
    Lambda_U0 = Lambda_U[locx_tra0]
    Lambda_U1 = Lambda_U[locx_tra1]
    X_test0 = X_test[locx_tes0]
    X_test1 = X_test[locx_tes1]
    for loop in range(100):
        Z_2_train = train_data['Z_2']
        n1 = len(Z_2_train)   #sample size of train data
        n2 = len(Z_2_test)   #sample size of test data
        locx_tra1 = np.where(Z_2_train>zeta0)  #Z_2>zeta in train data
        locx_tra0 = np.where(Z_2_train<=zeta0)
        locx_tes1 = np.where(Z_2_test>zeta0)  #Z_2>zeta in test data
        locx_tes0 = np.where(Z_2_test<=zeta0)
        ZC = np.vstack((Z_train,Z_train*(Z_2_train>zeta0)))
        ZC = ZC.T
        g_X = g_DCP1(ZC,X_train,U_train,De_train,g_train_true,Lambda_U,X_test,\
                        Theta,Theta0,n_layer,n_node,n_lr,n_epoch)
        g_train = g_X['g_train']
        g_test = g_X['g_test']
        ############################
        c1 = CCP_est(m,U_train,De_train,Z_train,Theta0,g_train,Z_2_train,zeta0,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        Theta1 = Theta_est(De_train,Z_train,Z_2_train,Lambda_U,g_train,zeta0)
        zeta1 = zeta_est(De_train,Z_train,Z_2_train,g_train,Lambda_U,Theta1)
        if (np.linalg.norm(Theta0-Theta1,ord=2) <= 1e-5):
            C_index = 1
            break
        c0 = c1
        Theta0 = Theta1
        zeta0 = zeta1
        #print(loop)
        #print(Theta1)
        #print(zeta1)
    return {
        'g_train': g_train,
        'g_test': g_test,
        'c': c1,
        'Theta': Theta1,
        'zeta': zeta1,
        'C_index': C_index,
    }


#########################################################################################################
def Est_dcp2(train_data,X_test,Z_2_test,Theta,Theta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0):
    U_train = train_data['U']
    De_train = train_data['De']
    Z_train = train_data['Z']
    Z_2_train = train_data['Z_2']
    Lambda_U = I_S(m,c0,U_train,nodevec)
    C_index = 0
    Z_2_train = train_data['Z_2']
    zeta0 = np.mean(Z_2_train)
    for loop in range(100):
        Res_g = g_DCP3(train_data,X_test,Z_2_test,Lambda_U,Theta,Theta0,zeta0,n_layer,n_node,n_lr,n_epoch)
        g_train = Res_g['g_train']
        g_test = Res_g['g_test']
        h_train = Res_g['h_train']
        h_test = Res_g['h_test']
        Res_train = Res_g['Res_train']
        Res_test = Res_g['Res_test']
        c1 = CCP_est(m,U_train,De_train,Z_train,Theta0,Res_train,Z_2_train,zeta0,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        Theta1 = Theta_est(De_train,Z_train,Z_2_train,Lambda_U,Res_train,zeta0)
        zeta1 = zeta_est(De_train,Z_train,Z_2_train,Res_train,Lambda_U,Theta1,seq=0.01)
        if (np.max(abs(Theta0-Theta1)) <= 0.01):
            C_index = 1
            break
        c0 = c1
        Theta0 = Theta1
        zeta0 = zeta1
    return {
        'g_train': g_train,
        'g_test': g_test,
        'h_train': h_train,
        'h_test': h_test,
        'Res_train':Res_train,
        'Res_test': Res_test,
        'c': c1,
        'Theta': Theta1,
        'zeta': zeta1,
        'C_index': C_index,
    }


def Est_dcp2exp(train_data,X_test,Z_2_test,Theta,Theta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0):
    zeta0 = 2
    U_train = train_data['U']
    De_train = train_data['De']
    Z_train = train_data['Z']
    Z_2_train = train_data['Z_2']
    Lambda_U = I_S(m,c0,U_train,nodevec)
    C_index = 0
    Z_2_train = train_data['Z_2']
    zeta0 = np.mean(Z_2_train)
    for loop in range(100):
        Res_g = g_DCP3(train_data,X_test,Z_2_test,Lambda_U,Theta,Theta0,zeta0,n_layer,n_node,n_lr,n_epoch)
        g_train = Res_g['g_train']
        g_test = Res_g['g_test']
        h_train = Res_g['h_train']
        h_test = Res_g['h_test']
        Res_train = Res_g['Res_train']
        Res_test = Res_g['Res_test']
        c1 = CCP_est(m,U_train,De_train,Z_train,Theta0,Res_train,Z_2_train,zeta0,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        Theta1 = Theta_est(De_train,Z_train,Z_2_train,Lambda_U,Res_train,zeta0)
        zeta1 = zeta_est(De_train,Z_train,Z_2_train,Res_train,Lambda_U,Theta1,seq=0.01)
        if (np.max(abs(Theta0-Theta1)) <= 0.01):
            C_index = 1
            break
        c0 = c1
        Theta0 = Theta1
        zeta0 = zeta1
    return {
        'g_train': g_train,
        'g_test': g_test,
        'h_train': h_train,
        'h_test': h_test,
        'Res_train':Res_train,
        'Res_test': Res_test,
        'c': c1,
        'Theta': Theta1,
        'zeta': zeta1,
        'C_index': C_index,
    }


def Est_dcp2H(train_data,X_test,Z_2_test,Theta,Theta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0,seq):
    U_train = train_data['U']
    De_train = train_data['De']
    Z_train = train_data['Z']
    Z_2_train = train_data['Z_2']
    Lambda_U = I_S(m,c0,U_train,nodevec)
    C_index = 0
    Z_2_train = train_data['Z_2']
    zeta0 = np.mean(Z_2_train)
    for loop in range(100):
        Res_g = g_DCP3H(train_data,X_test,Z_2_test,Lambda_U,Theta,Theta0,zeta0,n_layer,n_node,n_lr,n_epoch)
        g_train = Res_g['g_train']
        g_test = Res_g['g_test']
        h_train = Res_g['h_train']
        h_test = Res_g['h_test']
        Res_train = Res_g['Res_train']
        Res_test = Res_g['Res_test']
        c1 = CCP_est(m,U_train,De_train,Z_train,Theta0,Res_train,Z_2_train,zeta0,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        Theta1 = Theta_est(De_train,Z_train,Z_2_train,Lambda_U,Res_train,zeta0)
        zeta1 = zeta_est(De_train,Z_train,Z_2_train,Res_train,Lambda_U,Theta1,seq)
        if (np.max(abs(Theta0-Theta1)) <= 0.01):
            C_index = 1
            break
        c0 = c1
        Theta0 = Theta1
        zeta0 = zeta1
    return {
        'g_train': g_train,
        'g_test': g_test,
        'h_train': h_train,
        'h_test': h_test,
        'Res_train':Res_train,
        'Res_test': Res_test,
        'c': c1,
        'Theta': Theta1,
        'zeta': zeta1,
        'C_index': C_index,
    }