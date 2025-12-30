import numpy as np
import numpy.random as ndm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch import nn
import pandas as pd
#import matplotlib.pyplot as plt
from data_generator import generate_case_3
from g_deep import g_D
from CCP_estimation import CCP_est
from iteration_dcp import Est_dcp
from iteration_dcp import Est_dcp1
from I_spline import I_S
from Least_FD import LFD
from g_dcp import g_DCP1
from Theta_estimate import Theta_est
from zeta_estimate import zeta_est

def set_seed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed) 

set_seed(114514)
tau = 10
p = 3 
n = 2000
corr = 0.5 
n_lr = 1.5e-4
n_layer = 3
n_node = 50
n_epoch = 200 #
#Set_lr = np.array([2.8e-4, 3.2e-4, 4.2e-4]) #learning rate
#n_lr = 3.2e-4
Theta = [-1, 1]
zeta = 2

#dim_x = X_test.shape[0]
u_value = np.array(np.linspace(0, tau, 50), dtype="float32") 
Lambda_true = np.sqrt(u_value)/5 
m = 10 
nodevec = np.array(np.linspace(0, tau, m+2), dtype="float32")

m0 = 4 
nodevec0 = np.array(np.linspace(0, 2, m0+2), dtype="float32")

node_D = np.array([35, 30, 30])   #deep
lr_D = np.array([4e-4, 4e-4, 4e-4])

#c0 = np.array(0.1*np.ones(m+p), dtype="float32") 
#Theta0 = np.array([0,0], dtype='float32')

test_data = generate_case_3(200, corr, Theta, zeta)
X_test = test_data['X']
g_true = test_data['g_X']
Z_2_test = test_data['Z_2']


#n_epoch = 200 #

#Algorithm 1: classification tthe data
G_test_deep = []
ThetaM = []
zetaM = []
Vm = 1
for b in range(200):
    print('Alg=', Vm,'n=', n, 'b=', b)
    set_seed(12 + b)
    c0 = np.array(0.1*np.ones(m+p), dtype="float32")
    Theta0 = np.array([-0.9,0.9], dtype='float32')
    train_data = generate_case_3(n, corr, Theta, zeta)
    Z_train = train_data['Z']
    U_train = train_data['U']
    De_train = train_data['De']
    g_train = train_data['g_X']
    Z_2_train = train_data['Z_2']
    zeta0 = np.mean(Z_2_train)
    result = Est_dcp(train_data,X_test,Z_2_test,Theta,Theta0,zeta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0)
    Theta_res = result['Theta'] # vector to add row by row
    ThetaM.append(Theta_res)
    zeta_res = result['zeta']
    zetaM.append(zeta_res)
    #C_res = result['C_index']
    g_test_res = result['g_test']
    G_test_deep.append(result['g_test']) # vector to add row by row
Error_dcp = np.mean(np.array(G_test_deep), axis=0) - g_true
Theta_dcp = np.mean(np.array(ThetaM),axis=0)
zeta_dcp = np.mean(np.array(zetaM))
print(Error_dcp)
print(Theta_dcp)
print(zeta_dcp)


#Algorithm 2: piecewise by zeta
G_test_deep2 = []
ThetaM2 = []
zetaM2 = []
Vm = 2
for b in range(200):
    print('Alg=', Vm,'n=', n, 'b=', b)
    set_seed(12 + b)
    c0 = np.array(0.1*np.ones(m+p), dtype="float32")
    Theta0 = np.array([-0.9,0.9], dtype='float32')
    train_data = generate_case_3(n, corr, Theta, zeta)
    Z_train = train_data['Z']
    U_train = train_data['U']
    De_train = train_data['De']
    g_train = train_data['g_X']
    Z_2_train = train_data['Z_2']
    zeta0 = np.mean(Z_2_train)
    result = Est_dcp1(train_data,X_test,Z_2_test,Theta,Theta0,zeta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0)
    Theta_res = result['Theta'] # vector to add row by row
    ThetaM2.append(Theta_res)
    zeta_res = result['zeta']
    zetaM2.append(zeta_res)
    #C_res = result['C_index']
    g_test_res = result['g_test']
    G_test_deep2.append(result['g_test']) # vector to add row by row
Error_dcp2 = np.mean(np.array(G_test_deep2), axis=0) - g_true
Theta_dcp2 = np.mean(np.array(ThetaM2),axis=0)
zeta_dcp2 = np.mean(np.array(zetaM2))
print(Error_dcp2)
print(Theta_dcp2)
print(zeta_dcp2)

















