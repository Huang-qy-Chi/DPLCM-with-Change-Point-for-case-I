import numpy as np
#from Beta_estimate import Beta_est
from Theta_estimate import Theta_est
from CCP_estimation import CCP_est
from I_spline import I_S
from zeta_estimate import zeta_est
from g_Lcp import g_LCP3
from g_Lcp import g_LCP

def Est_Lcp2(train_data,X_test,Z_2_test,Theta,Theta0,nodevec,m,c0):
    U_train = train_data['U']
    De_train = train_data['De']
    Z_train = train_data['Z']
    Z_2_train = train_data['Z_2']
    Lambda_U = I_S(m,c0,U_train,nodevec)
    C_index = 0
    Z_2_train = train_data['Z_2']
    zeta0 = np.mean(Z_2_train)
    for loop in range(100):
        Res_g = g_LCP3(train_data,X_test,Z_2_test,Lambda_U,Theta,Theta0,zeta0)
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


def Est_Lcp(train_data,X_test,Z_2_test,Theta,Theta0,nodevec,m,c0):
    U_train = train_data['U']
    De_train = train_data['De']
    Z_train = train_data['Z']
    Z_2_train = train_data['Z_2']
    Lambda_U = I_S(m,c0,U_train,nodevec)
    C_index = 0
    Z_2_train = train_data['Z_2']
    zeta0 = np.mean(Z_2_train)
    for loop in range(100):
        Res_g = g_LCP(train_data,X_test,Z_2_test,Lambda_U,Theta,Theta0,zeta0)
        Theta1 = Res_g['Theta']
        g_train = Res_g['g_train']
        g_test = Res_g['g_test']
        h_train = Res_g['h_train']
        h_test = Res_g['h_test']
        Res_train = Res_g['Res_train']
        Res_test = Res_g['Res_test']
        c1 = CCP_est(m,U_train,De_train,Z_train,Theta0,Res_train,Z_2_train,zeta0,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        #Theta1 = Theta_est(De_train,Z_train,Z_2_train,Lambda_U,Res_train,zeta0)
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
