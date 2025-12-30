
import torch
from torch import nn
import numpy as np

#for the score and information
def LFDCP(Z1, train_data,Lambda_U,Res_train,Theta,zeta,n_layer,n_node,n_lr,n_epoch):
    Z_train = train_data['Z']
    Z_2_train = train_data['Z_2']
    Z_train = np.vstack((Z_train, Z_train*(Z_2_train>zeta)))
    Z_train = Z_train.T
    Z_train = torch.Tensor(Z_train)
    Z_1_train = torch.Tensor(Z1)
    #Z_2_train = torch.Tensor(Z_2_train)
    De_train = torch.Tensor(train_data['De'])
    X_U = torch.Tensor(np.c_[train_data['X'], train_data['U']])
    Lambda_U = torch.Tensor(Lambda_U)
    #Beta = torch.Tensor(np.array([Beta]))
    Theta = torch.Tensor(Theta)
    class DNNAB(torch.nn.Module):
        def __init__(self):
            super(DNNAB, self).__init__()
            layers = []
            layers.append(nn.Linear(6, n_node)) #5-dim X and 1-dim U
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 1))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred

    model = DNNAB()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def Loss(De, Z1, Z, Theta, Lambda_U, g_X, a_b):
        h_v = Lambda_U * torch.exp( Z@Theta + g_X)
        Q_y = h_v * (De * torch.exp(-h_v)/(1-torch.exp(-h_v)+1e-8) - (1-De))
        Loss_f = torch.mean(Q_y**2 * (Z1-a_b)**2)
        return Loss_f

    for epoch in range(n_epoch):
        pred_ab = model(X_U)
        loss = Loss(De_train, Z_1_train, Z_train, Theta, Lambda_U, Res_train, pred_ab[:, 0])
        loss.requires_grad_(True) #record the gradient
        optimizer.zero_grad() #reset the gradient as 0
        loss.backward() #back propagation
        optimizer.step() #renew the parameter

    ab_train = model(X_U)
    ab_train = ab_train[:,0].detach().numpy()
    return ab_train


#simultaneously read the a_b
def LFDCP1(Z1, train_data,Lambda_U,Res_train,Theta,zeta,n_layer,n_node,n_lr,n_epoch):
    Z_train = train_data['Z']
    Z_2_train = train_data['Z_2']
    Z_train = np.vstack((Z_train, Z_train*(Z_2_train>zeta)))
    Z_train = Z_train.T
    Z_train = torch.Tensor(Z_train)
    Z_1_train = torch.Tensor(Z1)
    #Z_2_train = torch.Tensor(Z_2_train)
    De_train = torch.Tensor(train_data['De'])
    X_U = torch.Tensor(np.c_[train_data['X'], train_data['U']])
    Lambda_U = torch.Tensor(Lambda_U)
    #Beta = torch.Tensor(np.array([Beta]))
    Theta = torch.Tensor(Theta)
    class DNNAB(torch.nn.Module):
        def __init__(self):
            super(DNNAB, self).__init__()
            layers = []
            layers.append(nn.Linear(6, n_node)) #5-dim X and 1-dim U
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 2))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred

    model = DNNAB()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def Loss(De, Z1, Z, Theta, Lambda_U, g_X, a_b):
        h_v = Lambda_U * torch.exp( Z@Theta + g_X)
        Q_y = h_v * (De * torch.exp(-h_v)/(1-torch.exp(-h_v)+1e-8) - (1-De))
        Loss_f = torch.mean(Q_y**2 * (Z1-a_b)**2)
        return Loss_f

    for epoch in range(n_epoch):
        pred_ab = model(X_U)
        loss = Loss(De_train, Z_1_train, Z_train, Theta, Lambda_U, Res_train, pred_ab[:, 0])
        loss.requires_grad_(True) #record the gradient
        optimizer.zero_grad() #reset the gradient as 0
        loss.backward() #back propagation
        optimizer.step() #renew the parameter

    ab_train = model(X_U)
    ab_train = ab_train[:,0].detach().numpy()
    return ab_train