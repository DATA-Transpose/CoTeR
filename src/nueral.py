import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


"""##Relevance Estimating Neural Network"""

import os
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Linear):  # glorot_uniform weight, zero bias
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()

def try_gpu(i=0):
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        if gpu_count >= i + 1:
            return torch.device(f'cuda:{i}')
        else:
            warnings.warn("There are no enough devices, use the cuda:0 instead.")
            return torch.device(f'cuda:{0}')
    return torch.device('cpu')


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=True),
            nn.Sigmoid()
        )
        self.device = try_gpu(0)
        self.seq.to(self.device)
        self.opt = optim.Adam(self.seq.parameters(), lr=0.001, weight_decay=0.001)

    def forward(self, x):
        x = self.seq(x)
        return x.flatten()

    def loss(self, y_true, y_pred):
        diff = (0 - y_pred) **2 +  y_true * ((1-y_pred)**2 - (0-y_pred)**2)
        return torch.mean(diff)

    def train(self, x, y, epochs, trial, batch_size=400):
        x = torch.Tensor(x).to(self.device)
        y = torch.Tensor(y).to(self.device)

        shuffle_idx = torch.randperm(x.shape[0])
        x = x[shuffle_idx]
        y = y[shuffle_idx]
        
        batch_count = math.ceil(x.size()[0] / batch_size)
        self.seq.train()
        
        loss_list = []

        for e in range(epochs):
            lossInE = []

            for i in range(batch_count):
                start_index = i * batch_size
                train_x = x[start_index : start_index + batch_size]
                train_y = y[start_index : start_index + batch_size]

                predict = self(train_x)

                self.opt.zero_grad()
                loss = self.loss(train_y, predict)
                lossInE.append(loss.item())
                loss.backward()
                self.opt.step()
            loss_list.append(np.mean(lossInE))
        
        if(trial % 1000 == 99):
            self.seq.eval()
            lossInE = []
            predict = self(x)
            loss = self.loss(y, predict)
            lossInE.append(loss.item())
            epLoss = np.mean(np.array(lossInE))
            print(f'train loss in trial {trial}: {epLoss}')
        
        return loss_list
    
    def predict(self, x):
        x = torch.Tensor(x).to(self.device)
        self.seq.eval()
        return self(x).detach().cpu().numpy()
