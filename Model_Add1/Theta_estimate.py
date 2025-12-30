
import numpy as np
import scipy.optimize as spo
#estimating the regression parameter
def Theta_est(De, Z, Z_2, Lambda_U, g_X, zeta):
    ZC0 = np.vstack((Z,Z*(Z_2>zeta)))
    p = ZC0.shape[0]
    def TF(*args):   
        ZC = np.vstack((Z,Z*(Z_2>zeta)))
        Lam = Lambda_U * np.exp(ZC.T @ args[0] + g_X)
        Loss_F = np.mean(-De * np.log(1 - np.exp(-Lam) + 1e-5) + (1 - De) * Lam)
        return Loss_F
    result = spo.minimize(TF,np.zeros(p),method='SLSQP') #nonconvex optimaization
    return result['x']
