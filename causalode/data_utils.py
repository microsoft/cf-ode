import pytorch_lightning as pl
import sys
sys.path.insert(0,"../")
from causalode.utils import DATA_DIR
import causalode.utils
from causalode.utils import str2bool
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import argparse
import numpy as np
from scipy.integrate import odeint
import pandas as pd

def create_pendulum_data(N,gamma, noise_std, seed = 421, continuous_treatment = False, fixed_length = False, static_x = False, strong_confounding = False, linspace_theta = False):

    np.random.seed(seed)
    g = 9.81
    if fixed_length:
        l_s = torch.ones(N)*(0.5*4 + 0.5)
    else:
        l_s = np.random.rand(N) * 4 + 0.5

    A = 10
    phi, delta = 1,1

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def df_dt(x,t, l):
        return x[1], -(g/l)*x[0]

    def dfu_dt(x,t,phi,delta):
            return (phi*x[1]*x[3]-delta*x[2]*x[3], -phi*x[2], phi*x[1], -delta*x[3])

    def df_dt_complete(x,t,l,phi,delta):
        return (x[1], -(g/l)*x[0]*(1+x[2])) + dfu_dt(x[2:],t,phi,delta)

    def fun_u(t):
        return 10*sigmoid(4*t-5)*(1-sigmoid(4*t-6))
    
    def df_dt_fun(x,t,l):
        return (x[1], -(g/l)*x[0]*(1+fun_u(t-10)))

    def vfun(x):
        return 0.02*(np.cos(5*x-0.2) * (5-x)**2)**2
        #return 0.2*(np.cos(10*x) * (3-x)**2)**2
    
    X = []
    Y_0 = []
    Y_1 = []
    if linspace_theta:
        thetas_0 = np.linspace(0.5,1.5)
    else:
        thetas_0 = np.random.rand(N)+0.5
    v_treatment = []

    t = np.linspace(0,15,31)
    t_ = t[t>=10]
    t_x = t[t<=10]
    t_y = t[t>10]

    for i in range(N):
        theta_0 = thetas_0[i]
        
        y0 = np.array([theta_0,0])
        y = odeint(df_dt, y0, t, args = (l_s[i],))

        v_treatment.append( y[t==10,1].item() )

        if not continuous_treatment:
            v_new = y[t==10,1].item() + vfun(theta_0)
            y0_ = np.array([y[t==10,0].item(),v_new])
            y_ = odeint(df_dt, y0_, t_, args = (l_s[i],))
        else:
            #if absolute_fun:
            #y0_ = y[t==10][0]
            #y_ = odeint(df_dt_fun,y0_,t_,args = (l_s[i],))
            #else:
            if strong_confounding:
                A_ = A * vfun(theta_0)
                A_ = A * theta_0
            else:
                A_ = A
            y0_ = np.concatenate((y[t==10],np.array([0,1,0,A_])[None,:]),1)
            y_ = odeint(df_dt_complete,y0_[0],t_,args=(l_s[i],phi,delta))

        x = y[t<=10,0]
        y_0 = y[t>10,0]
        y_1 = y_[t_>10,0]
        
        X.append(torch.Tensor(x))
        Y_0.append(torch.Tensor(y_0))
        Y_1.append(torch.Tensor(y_1))
        
    v_treatment = np.array(v_treatment)
    
    p = 1-sigmoid(gamma*(thetas_0-1))
    #p = sigmoid(gamma*(v_treatment-1))
    
    T = torch.zeros(N)
    T[np.random.rand(N)<p] = 1

    Y_0 = torch.stack(Y_0) + noise_std * torch.randn(N,len(t_y))
    Y_1 = torch.stack(Y_1) + noise_std * torch.randn(N,len(t_y))
    X = torch.stack(X) + noise_std * torch.randn(N,len(t_x))

    Y_fact = Y_0 * (1-T)[:,None] + Y_1 * T[:,None]
    Y_cf = Y_0 * (T)[:,None] + Y_1 * (1-T)[:,None]

    if strong_confounding:
        T = T * thetas_0

    t_X = torch.Tensor(np.tile(t_x[None,:],(X.shape[0],1)))
    t_Y = torch.Tensor(np.tile(t_y[None,:],(Y_fact.shape[0],1))) - t_x[-1]

    if static_x:
        # returns only the non-treated occurence in the dataset
        treatment_mask = (T>=0)
        X = np.concatenate((thetas_0[treatment_mask,None],l_s[treatment_mask,None]),-1)
        X = X-X.mean(0)
        std_ = X.std(0)
        std_[std_==0] = 1
        X = X/(std_)
        return X , T[treatment_mask], Y_fact[treatment_mask,...,None], Y_cf[treatment_mask,...,None], p[treatment_mask], thetas_0[treatment_mask]
    
    return X[...,None], T, Y_fact[...,None], Y_cf[...,None], p, thetas_0, t_X, t_Y



class PendulumDataset(Dataset):
    def __init__(self,N, gamma,noise_std, seed, continuous_treatment, fixed_length, static_x, strong_confounding):

        X, T, Y_fact, Y_cf, p, thetas_0, t_X, t_Y = create_pendulum_data(N, gamma, noise_std, seed, continuous_treatment, fixed_length, static_x, strong_confounding)
        self.X = X
        self.T = T
        self.Y_fact = Y_fact
        self.Y_cf = Y_cf
        self.T_cf = (~T.bool()).float()
        self.p = p
        self.thetas_0 = thetas_0
        self.t_X  = t_X
        self.t_Y  = t_Y

    def __getitem__(self,idx):
        return self.X[idx], self.Y_fact[idx], self.T[idx], self.Y_cf[idx], self.p[idx], self.thetas_0[idx], self.t_X[idx], self.t_Y[idx]
    def __len__(self):
        return self.X.shape[0]

class PendulumDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, N_ts, gamma, noise_std, num_workers = 4, continuous_treatment = False, fixed_length= False, static_x = False, strong_confounding = False, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True

        if static_x:
            self.input_dim = 2
        else:
            self.input_dim = 1
        self.output_dim = 1

        self.N = N_ts
        self.gamma = gamma
        self.noise_std = noise_std

        self.continuous_treatment = continuous_treatment

        self.fixed_length = fixed_length

        self.static_x = static_x
        self.strong_confounding = strong_confounding

    def prepare_data(self):

        dataset = PendulumDataset(self.N, self.gamma, self.noise_std, self.seed, self.continuous_treatment, self.fixed_length, self.static_x, self.strong_confounding)       
        
        train_idx = np.arange(len(dataset))[:int(0.5*len(dataset))]
        val_idx = np.arange(len(dataset))[int(0.5*len(dataset)):]
        test_idx = val_idx[int(len(val_idx)/2):]
        val_idx = val_idx[:int(len(val_idx)/2)]

        if self.batch_size==0:
            self.train_batch_size = len(train_idx)
            self.val_batch_size = len(val_idx)
            self.test_batch_size = len(test_idx)
        else:
            self.train_batch_size = self.batch_size
            self.val_batch_size = self.batch_size
            self.test_batch_size = self.batch_size

        self.train = Subset(dataset,train_idx)
        self.val = Subset(dataset,val_idx)
        self.test = Subset(dataset,test_idx)
    
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
            )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--N_ts', type=int, default=1000)
        parser.add_argument('--gamma', type=float, default=0)
        parser.add_argument('--noise_std', type=float, default=0)
        parser.add_argument('--continuous_treatment', type=str2bool, default=False)
        parser.add_argument('--fixed_length', type=str2bool, default=False)
        parser.add_argument('--static_x', type=str2bool, default=False, help = "If true, returns the initial theta value as X")
        parser.add_argument('--strong_confounding', type=str2bool, default=False, help = "If true, increases the confuonding byt a particular function")
        return parser

if __name__=="__main__":
    datam = PendulumDataModule(batch_size = 32, seed = 42, noise_std = 0., N_ts=1000, gamma = 0)
    datam.prepare_data()
