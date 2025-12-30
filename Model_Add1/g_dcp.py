
import torch
from torch import nn
import numpy as np
#Using DNN to approximate the function with change point
def g_DCP1(Z_train,X_train,U_train,De_train,g_train_true,Lambda_U,X_test,Theta,Theta0,n_layer,n_node,n_lr,n_epoch):
    #input data into torch.tensor
    Z_train = torch.Tensor(Z_train)
    X_train = torch.Tensor(X_train)
    U_train = torch.Tensor(U_train)
    De_train = torch.Tensor(De_train)
    g_train_true = torch.Tensor(g_train_true)
    X_test = torch.Tensor(X_test)
    Lambda_U = torch.Tensor(Lambda_U)
    Theta0 = torch.Tensor(Theta0.T)

    class DNNModel(torch.nn.Module): #set the neural network
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(5, n_node)) #linear layer
            layers.append(nn.ReLU()) #ReLU layer
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 1))
            self.model = nn.Sequential(*layers) #complete the neural network
        def forward(self, x):   #forward propagation
            y_pred = self.model(x)
            return y_pred

    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def my_loss(De, Z, Theta, Lambda_U, g_X):
        Lam1 = Lambda_U * torch.exp(Z@Theta + g_X)
        loss_fun = -torch.mean(De*torch.log(1-torch.exp(-Lam1)+1e-5) - (1-De)*Lam1)
        return loss_fun

    for epoch in range(n_epoch):
        pred_g_X = model(X_train)
        loss = my_loss(De=De_train, Z=Z_train, Theta=Theta0, Lambda_U=Lambda_U, g_X=pred_g_X[:, 0])
        loss.requires_grad_(True)
        optimizer.zero_grad() #reset the gradient as 0
        loss.backward()   #back propagation
        optimizer.step()  #renew the parameter

    g_train = model(X_train)
    g_test = model(X_test)
    g_train = g_train[:,0].detach().numpy()
    g_test = g_test[:,0].detach().numpy()
    return {
        'g_train': g_train,
        'g_test': g_test
    }






def g_DCP2(Z_train,X_train,U_train,De_train,g_train_true,Lambda_U,X_test,Theta,Theta0,n_layer,n_node,n_lr,n_epoch):
    #input data into torch.tensor
    Theta0 = np.array(Theta0, dtype = 'float32')
    Z_train = torch.Tensor(Z_train)
    X_train = torch.Tensor(X_train)
    U_train = torch.Tensor(U_train)
    De_train = torch.Tensor(De_train)
    g_train_true = torch.Tensor(g_train_true)
    X_test = torch.Tensor(X_test)
    Lambda_U = torch.Tensor(Lambda_U)
    Theta0 = torch.Tensor(Theta0)

    class DNNModel(torch.nn.Module): #set the neural network
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(5, n_node)) #linear layer
            layers.append(nn.ReLU()) #ReLU layer
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 1))
            self.model = nn.Sequential(*layers) #complete the neural network
        def forward(self, x):   #forward propagation
            y_pred = self.model(x)
            return y_pred

    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def my_loss(De, Z, Theta, Lambda_U, g_X):
        Lam1 = Lambda_U * torch.exp(Z[:,0]*Theta[0] +Z[:,1]*Theta[1] + g_X)
        loss_fun = -torch.mean(De*torch.log(1-torch.exp(-Lam1)+1e-5) - (1-De)*Lam1)
        return loss_fun

    for epoch in range(n_epoch):
        pred_g_X = model(X_train)
        loss = my_loss(De=De_train, Z=Z_train, Theta=Theta0, Lambda_U=Lambda_U, g_X=pred_g_X[:, 0])
        loss.requires_grad_(True)
        optimizer.zero_grad() #reset the gradient as 0
        loss.backward()   #back propagation
        optimizer.step()  #renew the parameter

    g_train = model(X_train)
    g_test = model(X_test)
    g_train = g_train[:,0].detach().numpy()
    g_test = g_test[:,0].detach().numpy()
    return {
        'g_train': g_train,
        'g_test': g_test
    }


def g_DCP3(train_data,X_test,Z_2_test,Lambda_U,Theta,Theta0,zeta0,n_layer,n_node,n_lr,n_epoch):
    #input data into torch.tensor
    Z_train = train_data['Z']
    Z_2_train = train_data['Z_2']
    Z_train = np.vstack((Z_train, Z_train*(Z_2_train>zeta0)))
    Z_train = Z_train.T
    Z_train = torch.Tensor(Z_train)
    X_train = torch.Tensor(train_data['X'])
    U_train = torch.Tensor(train_data['U'])
    De_train = torch.Tensor(train_data['De'])
    g_train_true = torch.Tensor(train_data['g_X'])
    X_test = torch.Tensor(X_test)
    Lambda_U = torch.Tensor(Lambda_U)
    Theta0 = torch.Tensor(Theta0)
    Z_2_train = torch.Tensor(train_data['Z_2'])

    class DNNModel(torch.nn.Module): #set the neural network
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(5, n_node)) #linear layer
            layers.append(nn.ReLU()) #ReLU layer
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 2))  #output g(X) and h(X)
            self.model = nn.Sequential(*layers) #complete the neural network
        def forward(self, x):   #forward propagation
            y_pred = self.model(x)
            return y_pred

    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def my_loss(De, Z, Z_2, Theta, Lambda_U, zeta, g_X):
        matrix = torch.ones((Z.shape[0], 2))
        matrix[:, 1] = (Z_2 > zeta).float() #ensures the order of the output
        g_X_1 = torch.sum(matrix * g_X, dim=1)
        Lam1 = Lambda_U * torch.exp(Z@Theta + g_X_1)
        loss_fun = -torch.mean(De*torch.log(1-torch.exp(-Lam1)+1e-5) - (1-De)*Lam1)
        return loss_fun

    for epoch in range(n_epoch):
        pred_g_X = model(X_train)  #a n*2 matrix of (g(X_i),h(X_i))
        loss = my_loss(De_train, Z_train, Z_2_train, Theta0, Lambda_U, zeta0, pred_g_X)
        loss.requires_grad_(True)
        optimizer.zero_grad() #reset the gradient as 0
        loss.backward()   #back propagation
        optimizer.step()  #renew the parameter
    
    #calculate g(X) and h(X) for both train and test data
    res_train = model(X_train)
    g_train = res_train[:,0]
    h_train = res_train[:,1] 
    res_test = model(X_test)
    g_test = res_test[:,0]
    h_test = res_test[:,1]

    # The default output of the neural network is in the form of a matrix n*1 matrix, which needs to be converted into a vector
    g_train = g_train.detach().numpy() 
    h_train = h_train.detach().numpy()
    g_test = g_test.detach().numpy()
    h_test = h_test.detach().numpy()
    Z_2_train = train_data['Z_2']
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
