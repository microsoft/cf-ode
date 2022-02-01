
import sys
sys.path.insert(0,"../")

import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, MLFlowLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
import torch

import matplotlib.pyplot as plt

import numpy as np

from causalode import models, data_utils
from azureml.core.run import Run
from causalode.utils import str2bool



def update_dict(main_dict, new_dict):
    for key in new_dict.keys():
        main_dict[key].append(new_dict[key])
    return main_dict


#Evaluating the predictions

def evaluate_preds(model,dataset,t_lim):
    """
    computes the reconstruction of the model on the test dataset.
    Returns the factual and counterfactual means and standard deviations of the prediction as well as the original data.
    
    The model is evaluated at the interger time points between t=0 and t_lim (_o subscript) but also in a more continuous fashion ( linspace(0,t_lim,100).
    
    Inputs:
     * model : loaded model returning means and standard deviations of predictions as output
     * dataset : pl datamodule (the test part will be used only)
    Outputs:
     * mu_hat : means of the factual preds [batch x time x dim]
     * mu_cf_hat : means of the counterfactual preds
     * mu_hat_o : means of the factual preds evaluated at the integer time points only.
     * std_hat : std of the factual preds [batch x time x dim]
     * std_cf_hat : std of the counterfactual preds [batch x time x dim]
     * std_hat_o : std of the factual preds evaluated at the integer time points only.
     * Y : original data (factual)
     * Y_cf : original data (cfactual)
     * p : propensity score (probability of receiving treatment) 
     * T : Treatment assignments [batch]
     * times_plot : times vector that maps to mu_hat (linspace(0,t_lim,100))
    """

    model.eval()

    mu_hat_list = []
    mu_cf_hat_list =[]
    std_hat_list = []
    std_cf_hat_list =[]
    mu_hat_o_list =[]
    std_hat_o_list = []
    Y_list = []
    Y_cf_list = []
    p_list = []
    T_list = []

    with torch.no_grad():
        for i,batch in enumerate(dataset.test_dataloader()):
            X,Y, T, Y_cf, p, thetas_0 = batch
            times = torch.linspace(0,t_lim,101)
            #times = torch.arange(11).float()
            Y_hat, decoded_u = model(X,T,return_decoded_u=True, times=times)
            Y_cf_hat, decoded_cf_u = model(X,(~T.bool()).float(),return_decoded_u=True, times=times)

            mu_hat, var_hat = torch.chunk(Y_hat,2,dim=-1)
            std_hat = torch.sqrt(torch.sigmoid(var_hat))

            mu_cf_hat, var_cf_hat = torch.chunk(Y_cf_hat,2,dim=-1)
            std_cf_hat = torch.sqrt(torch.sigmoid(var_cf_hat))

            times_o = torch.arange(t_lim+1).float()
            Y_hat_o, _ =  model(X,T,return_decoded_u=True, times=times_o)

            mu_hat_o, var_hat_o = torch.chunk(Y_hat_o,2,dim=-1)
            std_hat_o = torch.sqrt(torch.sigmoid(var_hat_o))

            mu_hat_list.append(mu_hat)
            mu_cf_hat_list.append(mu_cf_hat)
            mu_hat_o_list.append(mu_hat_o)

            std_hat_list.append(std_hat)
            std_cf_hat_list.append(std_cf_hat)
            std_hat_o_list.append(std_hat_o)

            Y_list.append(Y)
            Y_cf_list.append(Y_cf)

            p_list.append(p)
            T_list.append(T)

    times_plot = times[1:]

    mu_hat = torch.cat(mu_hat_list,0)
    mu_cf_hat = torch.cat(mu_cf_hat_list,0)
    mu_hat_o = torch.cat(mu_hat_o_list,0)

    std_hat = torch.cat(std_hat_list,0)
    std_cf_hat = torch.cat(std_cf_hat_list,0)
    std_hat_o = torch.cat(std_hat_o_list,0)

    Y = torch.cat(Y_list,0)
    Y_cf = torch.cat(Y_cf_list,0)

    p = torch.cat(p_list,0)
    T = torch.cat(T_list,0)
    
    return mu_hat, mu_cf_hat, mu_hat_o, std_hat, std_cf_hat, std_hat_o, Y, Y_cf, p, T, times_plot

def plot_aggregated(curve_dict, start_idx = 5, rec_type = "", path = None):
    """
    start_idx : index from which to start plotting (because there might be some wacky stuff in the begining for the lack of samples)
    """
    fig,ax = plt.subplots()
    key_label_map = {"random": "Random", "propensity": "Propensity", "uncertainty": "Uncertainty Based"}
    for key in curve_dict.keys():
        curves = np.stack(curve_dict[key])
        ax.plot(np.linspace(0,100,curves.shape[1])[5:],curves.mean(0)[5:], label = key_label_map[key])
        ax.fill_between(np.linspace(0,100,curves.shape[1])[5:],(curves.mean(0)-curves.std(0))[5:],(curves.mean(0)+curves.std(0))[5:], alpha = 0.1 )
        ax.set_ylim(0.1,1.5)
        ax.set_xlabel("% of data retained")
        ax.set_ylabel("Normalized RMSE")
        ax.legend()
        ax.set_title(f"Normalized {rec_type} RMSE")    
    
    if path is not None:
        plt.savefig(path+".pdf")
    return fig

def plot_trimming_prop(Y,mu,std,t_lim_start, t_lim_end,times_plot, data_type, p=None, normalize = False):

    mask_available = np.isin(times_plot,1+torch.arange(t_lim_start,t_lim_end))

    mse = np.sqrt((Y[:,t_lim_start:t_lim_end] - mu[:,mask_available,:]).pow(2)).mean(1)
    if normalize:
        mse = mse/mse.mean()
    std_ = std[:,t_lim_start:t_lim_end].mean(1)
    sort_idx = torch.sort(std_,0,descending = False)[1]
    mse_sorted = mse[sort_idx]

    random_idx = np.random.choice(len(sort_idx),len(sort_idx),replace = False)
    mse_random = mse[random_idx]
    
    if p is not None:
        p_idx = torch.sort(p,0)[1]
        mse_p = mse[p_idx]
    else:
        p_idx = random_idx
        mse_p =torch.zeros(mse_random.shape)
        
    mse_kept = []
    mse_kept_random = []
    mse_kept_p = []
    for i in range(len(mse)):
        mse_kept.append(mse_sorted[:i].mean())
        mse_kept_random.append(mse_random[:i].mean())
        mse_kept_p.append(mse_p[:i].mean())
    
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(sort_idx))/len(sort_idx),np.array(mse_kept),label = "Uncertainty based")
    ax.plot(np.arange(len(random_idx))/len(random_idx),np.array(mse_kept_random), label = "Random")
    ax.plot(np.arange(len(p_idx))/len(p_idx),np.array(mse_kept_p), label = "Propensity")

    ax.set_xlabel("Proportion of data kept in the dataset")
    ax.set_ylabel("Average MSE")
    ax.set_title(f"Comparison of approaches for data trimming - {data_type}")
    ax.legend()
    #plt.show()
    
    return fig, {"uncertainty" : np.array(mse_kept), "random" : np.array(mse_kept_random), "propensity" : np.array(mse_kept_p)}
