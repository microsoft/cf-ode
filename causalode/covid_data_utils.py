import pytorch_lightning as pl
from causalode.utils import DATA_DIR
import causalode.utils as utils
from causalode.utils import str2bool
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import argparse
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import warnings

def fluids_input(t):
    return 5*np.exp(-((t-5)/5)**2)

def v_fun(x):
    return 0.02*(np.cos(5*x-0.2) * (5-x)**2)**2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dx_dt(x, t, params):
    # Parameters:
    k_IR = params["k_IR"]
    k_PF = params["k_PF"]
    k_O = params["k_O"]
    E_max = params["E_max"]
    E_C = params["E_C"]
    k_Dex = params["k_Dex"]
    k_DP = params["k_DP"]
    k_IIR = params["k_IIR"]
    k_DC = params["k_DC"]
    h_P = params["h_P"]
    k_1  = params["k_1"]
    k_2  = params["k_2"]
    k_3  = params["k_3"]
    h_C = params["h_C"]

    t_treatment = params["t_treatment"]

    # Unknown parameters:
    
    if (params["treatment"]) and (t>=t_treatment):
        additive_term = 10
    else:
        additive_term = 0

    return k_IR * x[3] + k_PF * x[3]*x[0] - k_O*x[0] + (E_max*(x[0]**h_P))/(E_C+(x[0]**h_P)) - k_Dex *x[0]*x[1], -k_2*x[1] + k_3*x[2], -k_3 * x[2] + additive_term, k_DP *x[3]-k_IIR*x[3]*x[0] - k_DC*x[3]*(x[4]**h_C), k_1*x[0]
    
def init_random_state():

    x0 = np.random.exponential(10)
    x1 = np.random.exponential(1/100)
    x2 = np.random.exponential(1/100)
    x3 = np.random.exponential(10)
    x4 = np.random.exponential(10)
    init_state = np.stack((x0,x1,x2,x3,x4))
   
    k_Dex = 1+np.random.rand()*15
    return init_state, k_Dex


def create_covid_data(N,gamma,noise_std, t_span = 30, t_treatment = 15, seed = 421, normalize = True, output_dims = [0], input_dims = [0,4] ):

    np.random.seed(seed)

    X = []
    Y_0 = []
    Y_1 = []
    k_Dex_list = []
    
    params = {"k_IR": 0.2,
            "k_PF": 0.2,
            "k_O": 1.0 ,
            "E_max": 1.,
            "E_C": 1.,
            "k_DP": 4.0,
            "k_IIR": 0.1,
            "k_DC": 0.1,
            "h_P": 2,
            "k_1": 1,
            "k_2": 1,
            "k_3": 1,
            "h_C": 8,
            "t_treatment" : t_treatment
            }
    
    params_treatment = params.copy()
    params_treatment["treatment"]=True
    params_notreatment = params.copy()
    params_notreatment["treatment"]=False
    
    t = np.linspace(0,t_span,30).astype(float)
    
    i=0
    warnings.filterwarnings("error")
    while i < N:
        init_state, k_Dex = init_random_state()
        params_treatment["k_Dex"] = k_Dex
        try:
            y1 = odeint(dx_dt,init_state,t,args=tuple([params_treatment]))
            
            params_notreatment["k_Dex"] = k_Dex
            y0 = odeint(dx_dt,init_state,t,args=tuple([params_notreatment]))
            
        except Warning:
            print("Warning")
            continue
            import ipdb; ipdb.set_trace()
             
        X.append(torch.Tensor(init_state))
        Y_0.append(torch.Tensor(y0))
        Y_1.append(torch.Tensor(y1))
        k_Dex_list.append(torch.Tensor([k_Dex]))
        i+=1
    
    warnings.filterwarnings("default")
    k_Dex = torch.stack(k_Dex_list)[:,0]
    p = 1-torch.sigmoid(gamma*((k_Dex-1)/15-0.5))
    T = torch.zeros(N)
    T[torch.rand(N)<p] = 1

    Y_0 = torch.stack(Y_0)
    Y_1 = torch.stack(Y_1)
    Y_0 += noise_std * torch.randn(Y_0.shape)
    Y_1 += noise_std * torch.randn(Y_1.shape)
    X = torch.stack(X)
    X += noise_std * torch.randn(X.shape)

    Y_fact = Y_0 * (1-T)[:,None,None] + Y_1 * T[:,None,None]
    Y_cf = Y_0 * (T)[:,None,None] + Y_1 * (1-T)[:,None,None]

    if normalize:
        mu = Y_fact.mean([0,1])
        std = Y_fact.std([0,1])

        Y_fact = (Y_fact - mu)/std
        Y_cf = (Y_cf - mu)/std
        
        mu_X = X.mean([0,1])
        std_X = X.std([0,1])
        X = (X-mu_X)/std_X
    
    pre_treat_mask = (t<=t_treatment)
    post_treat_mask = (t>t_treatment)
    
    X_static = X
    X = Y_fact[:,pre_treat_mask][:,:,input_dims]
    X_ = Y_cf[:,pre_treat_mask][:,:,input_dims]

    Y_fact = Y_fact[:,post_treat_mask][:,:,output_dims] 
    Y_cf = Y_cf[:,post_treat_mask][:,:,output_dims]

    t_x = t[pre_treat_mask]
    t_y = t[post_treat_mask]

    t_X = torch.Tensor(np.tile(t_x[None,:],(X.shape[0],1)))
    t_Y = torch.Tensor(np.tile(t_y[None,:],(Y_fact.shape[0],1))) - t_x[-1]


    return X, X_static, T, Y_fact, Y_cf, p, k_Dex, t_X, t_Y

class CovidDataset(Dataset):
    def __init__(self,N, gamma,noise_std, t_span, t_treatment, seed):

        X, X_static, T, Y_fact, Y_cf, p, k_Dex, t_X, t_Y = create_covid_data(N = N, gamma = gamma, noise_std = noise_std, t_span = t_span, t_treatment = t_treatment, seed = seed)

        self.X = X
        self.T = T
        self.Y_fact = Y_fact
        self.Y_cf = Y_cf
        self.T_cf = (~T.bool()).float()
        self.p = p
        self.param = k_Dex
        self.t_X = t_X
        self.t_Y = t_Y

    def __getitem__(self,idx):
        return self.X[idx], self.Y_fact[idx], self.T[idx], self.Y_cf[idx], self.p[idx], self.param[idx], self.t_X[idx], self.t_Y[idx]
    def __len__(self):
        return self.X.shape[0]

class CovidDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, N_ts, gamma, noise_std, t_span, t_treatment, num_workers = 4, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True

        self.input_dim = 2
        self.output_dim = 1 # number of dimensions to reconstruct in the time series

        self.N = N_ts
        self.gamma = gamma
        self.noise_std = noise_std
        self.t_span = t_span
        self.t_treatment = t_treatment

    def prepare_data(self):

        dataset = CovidDataset(N = self.N, gamma = self.gamma,noise_std =  self.noise_std, seed = self.seed, t_span = self.t_span, t_treatment = self.t_treatment)       
        
        train_idx = np.arange(len(dataset))[:int(0.5*len(dataset))]
        val_idx = np.arange(len(dataset))[int(0.5*len(dataset)):]
        test_idx = val_idx[int(len(val_idx)/2):]
        val_idx = val_idx[:int(len(val_idx)/2)]


        self.train = Subset(dataset,train_idx)
        self.val = Subset(dataset,val_idx)
        self.test = Subset(dataset,test_idx)
    
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
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
        parser.add_argument('--t_span', type=float, default=1.)
        parser.add_argument('--t_treatment', type=float, default=0.5)
        return parser

   
if __name__=="__main__":
    
    datam = CVDataModule(batch_size=32, N_ts = 1000, gamma = 0, seed = 4221, noise_std = 0., num_workers = 0 )
    datam.prepare_data()

