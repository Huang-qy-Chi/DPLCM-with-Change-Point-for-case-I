
import numpy as np
import scipy.optimize as spo
from I_spline import I_U
#for Lambda's parameter
def CCP_est(m, U, De, Z, Theta, g_X, Z_2, zeta, nodevec):
    Iu = I_U(m, U, nodevec)
    def LF(*args):
        a = args[0]
        ZC = np.vstack((Z,Z*(Z_2>zeta)))
        Lam1 = np.dot(Iu,a) * np.exp(ZC.T@Theta.T+g_X)
        Loss_F1 = np.mean(-De * np.log(1-np.exp(-Lam1)+1e-5) + (1-De)*Lam1)
        return Loss_F1
    bnds = []
    for i in range(m+3):
        bnds.append((0,1000))
    result = spo.minimize(LF,np.ones(m+3),method='SLSQP',bounds=bnds)
    return result['x']


