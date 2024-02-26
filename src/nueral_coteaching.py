import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import nueral as BASENN


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

def unbiased_loss(y_true, y_pred):
    diff = (0 - y_pred) **2 +  y_true * ((1-y_pred)**2 - (0-y_pred)**2)

    return diff

class COTNET(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(COTNET, self).__init__()

        self.device = try_gpu(0)

        self.net1 = BASENN.MLP(in_dim, hidden_dim, out_dim)
        self.net1.apply(weights_init)
        self.net2 = BASENN.MLP(in_dim, hidden_dim, out_dim)
        self.net2.apply(weights_init)

    def forward(self, x):
        x = self.net1(x)
        return x.flatten()

    def predict(self, x, net_idx=1):
        with torch.no_grad():
            x = torch.Tensor(x).to(self.device)
            self.net1.seq.eval()
            self.net2.seq.eval()

            pred = [
                self.net1(x).detach().cpu().numpy(),
                self.net2(x).detach().cpu().numpy()
            ]
            if net_idx == 1:
                predict = pred[0]
            else:
                predict = pred[1]

            return predict.flatten()

    def co_loss(self, y_1, y_2, label, loss_top_rate=1):
        label_1, label_2 = label
        loss_top = int(label_1.shape[0] * float(loss_top_rate))
        loss_1 = unbiased_loss(label_1, y_1)
        ind_1_sorted = np.argsort(loss_1.data.cpu()).to(self.device)

        loss_2 = unbiased_loss(label_2, y_2)
        ind_2_sorted = np.argsort(loss_2.data.cpu()).to(self.device)

        ind_1_update = ind_1_sorted[:loss_top]
        ind_2_update = ind_2_sorted[:loss_top]

        loss_1_update = unbiased_loss(label_1.gather(0, ind_2_update), y_1.gather(0, ind_2_update))
        loss_2_update = unbiased_loss(label_2.gather(0, ind_1_update), y_2.gather(0, ind_1_update))

        return (
            torch.mean(loss_1),
            torch.mean(loss_2),
            torch.mean(loss_1_update),
            torch.mean(loss_2_update),
        )

    def co_loss_plus(self, y_1, y_2, label, loss_top_rate=1):
        label_1, label_2 = label
        loss_1 = unbiased_loss(label_1, y_1)
        loss_2 = unbiased_loss(label_2, y_2)

        diff = torch.abs(y_1 - y_2)
        diff_ind_sort = torch.argsort(diff, descending=True)
        top_disagree = int(diff_ind_sort.shape[0] * float(loss_top_rate))
        diff_ind_sort = diff_ind_sort[:top_disagree]
        
        sorted_label_1 = label_1[diff_ind_sort]
        sorted_label_2 = label_2[diff_ind_sort]
        sorted_y_1 = y_1[diff_ind_sort]
        sorted_y_2 = y_2[diff_ind_sort]

        _, _, top_loss_1, top_loss_2 = self.co_loss(sorted_y_1, sorted_y_2, (sorted_label_1, sorted_label_2), loss_top_rate=loss_top_rate)

        return (
            torch.mean(loss_1),
            torch.mean(loss_2),
            top_loss_1,
            top_loss_2,
        )
        

    def train(self, x, y, epochs, trial, doc_num, top_rate=0.5, batch_size=400, forget_rate=0.4):
        num_gradual = 10
        epoch_decay_start = 80
        mom1 = 0.9
        mom2 = 0.1
        alpha_plan = [0.001] * epochs
        beta1_plan = [mom1] * epochs
        forget_rate = float(doc_num) / float(trial)
        L = 4000
        forget_rate = -(forget_rate) / (2 * L) * trial + forget_rate
        forget_rate = max(0.4, forget_rate)
        for i in range(epoch_decay_start, epochs):
            alpha_plan[i] = float(epochs - i) / (epochs - epoch_decay_start) * 0.001
            beta1_plan[i] = mom2
        def adjust_learning_rate(optimizer, epoch):
            for param_group in optimizer.param_groups:
                param_group['lr']=alpha_plan[epoch]
                param_group['betas']=(beta1_plan[epoch], 0.999)
        y1 = np.copy(y)
        y2 = np.copy(y)

        x = torch.Tensor(x).to(self.device)
        y1 = torch.Tensor(y1).to(self.device)
        y2 = torch.Tensor(y2).to(self.device)

        shuffle_idx = torch.randperm(x.shape[0])
        x = x[shuffle_idx]
        y1 = y1[shuffle_idx]
        y2 = y2[shuffle_idx]
        
        batch_count = math.ceil(x.size()[0] / batch_size)

        loss_1_list = []
        loss_2_list = []
        top_loss_1_list = []
        top_loss_2_list = []

        rate_schedule = np.ones(epochs)*forget_rate
        rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)

        for e in range(epochs):
            self.net1.seq.train()
            self.net2.seq.train()
            loss_1_in_e = []
            loss_2_in_e = []
            top_loss_1_in_e = []
            top_loss_2_in_e = []
            for i in range(batch_count):
                start_index = i * batch_size
                train_x = x[start_index : start_index + batch_size]
                train_y_1 = y1[start_index : start_index + batch_size]
                train_y_2 = y2[start_index : start_index + batch_size]

                logits1 = self.net1(train_x)
                logits2 = self.net2(train_x)
                loss_1, loss_2, top_loss_1, top_loss_2 = self.co_loss(logits1, logits2, (train_y_1, train_y_2), 1 - rate_schedule[e])

                # net 1 grad
                self.net1.opt.zero_grad()
                (loss_1 + top_loss_1).backward()
                self.net1.opt.step()

                # net 2 grad
                self.net2.opt.zero_grad()
                (loss_2 + top_loss_2).backward()
                self.net2.opt.step()

                loss_1_in_e.append(loss_1.item())
                loss_2_in_e.append(loss_2.item())
                top_loss_1_in_e.append(top_loss_1.item())
                top_loss_2_in_e.append(top_loss_2.item())

            def mean_loss(loss_list, loss_in_e):
                ep_loss = np.mean(np.array(loss_in_e))
                loss_list.append(ep_loss)
                return loss_list

            loss_1_list = mean_loss(loss_1_list, loss_1_in_e)
            loss_2_list = mean_loss(loss_2_list, loss_2_in_e)
            top_loss_1_list = mean_loss(top_loss_1_list, top_loss_1_in_e)
            top_loss_2_list = mean_loss(top_loss_2_list, top_loss_2_in_e)
        
        if (trial % 1000 == 99):
            self.net1.seq.eval()
            self.net2.seq.eval()
            loss_1_in_e = []
            loss_2_in_e = []

            predict1 = self.net1(x)
            
            loss1 = unbiased_loss(y1, predict1)
            loss_1_in_e.append(torch.mean(loss1).item())
            epLoss1 = np.mean(np.array(loss_1_in_e))
            print(f'train loss in trial {trial}: {epLoss1}')
        
        return loss_1_list, loss_2_list, top_loss_1_list, top_loss_2_list
