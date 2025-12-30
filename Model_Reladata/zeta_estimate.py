import numpy as np



def zeta_est(De, Z, Z_2, g_X, Lambda_U,Theta, seq =0.01):
    #establish the grid of zeta
    Z_min = np.min(Z_2)
    Z_max = np.max(Z_2)
    #num = np.floor((Z_max-Z_min)/seq)
    #np.arange(0, 3, 0.1)
    zeta_grid = np.arange(Z_min, Z_max, seq)  #the search grid

    #define the log-likelihood loss
    def BZ2(*args):
        ZC = np.vstack((Z,Z*(Z_2>args[0])))
        Lam = Lambda_U * np.exp(ZC.T @ Theta + g_X)
        Loss_F = np.mean(-De * np.log(1 - np.exp(-Lam) + 1e-5) + (1 - De) * Lam)
        return Loss_F
    
    zeta_loss = []
    for zeta in zeta_grid:
        zeta_loss.append(BZ2(zeta))
    loss_min = min(zeta_loss)
    loc = zeta_loss.index(loss_min)
    zeta_est = zeta_grid[loc]
    zeta_est = zeta_est.astype(np.float32)
    return zeta_est
        











