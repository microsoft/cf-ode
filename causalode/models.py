import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
import pandas as pd
import numpy as np
import plotly.express as px
from causalode.utils import str2bool, _stable_division, LinearScheduler, GaussianNLLLoss
from torchdiffeq import odeint
import matplotlib.pyplot as plt

#from causalode.sinkhorn import SinkhornDistance

from torch import Tensor
import torch.nn.functional as F
import torchsde


class MCDropout(torch.nn.modules.dropout._DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, True, self.inplace)



class ContinuousTreatmentODE(nn.Module):
    def __init__(self,h_dim,u_dim,shared_u_dim, continuous_treatment, fun_treatment, dropout_p):
        """
        h_dim : dimension of the hidden for the general dynamics
        u_dim : dimension of the hidden for the treatment dynamics
        shared_u_dim : how many dimension of u are used to impact h
        """
        super().__init__()

        if ((continuous_treatment) and (not fun_treatment)):
            h_dim_gen = h_dim+shared_u_dim

            self.h_fun = nn.Sequential(nn.Linear(h_dim_gen,h_dim_gen),MCDropout(p=dropout_p),nn.ReLU(), nn.Linear(h_dim_gen,h_dim_gen),MCDropout(p=dropout_p), nn.ReLU(), nn.Linear(h_dim_gen, h_dim), nn.ReLU(), nn.Linear(h_dim,h_dim))
            self.u_fun = nn.Sequential(nn.Linear(u_dim,u_dim, bias =False),MCDropout(p=dropout_p),nn.ReLU(), nn.Linear(u_dim,u_dim, bias = False),MCDropout(p=dropout_p), nn.ReLU(), nn.Linear(u_dim, u_dim, bias = False),MCDropout(p=dropout_p), nn.ReLU(), nn.Linear(u_dim,u_dim, bias = False))

            self.h_dim_gen = h_dim_gen
        elif (continuous_treatment and fun_treatment):
            #self.decoder = nn.Sequential(nn.Linear(h_dim+1,h_dim),MCDropout(p=dropout_p),nn.ReLU(), nn.Linear(h_dim,h_dim),MCDropout(p=dropout_p), nn.ReLU(), nn.Linear(h_dim, h_dim),MCDropout(p=dropout_p), nn.ReLU(), nn.Linear(h_dim,h_dim))
            #self.treatment_fun = nn.Sequential(nn.Linear(1,u_dim),MCDropout(p=dropout_p),nn.ReLU(), nn.Linear(u_dim,u_dim),MCDropout(p=dropout_p), nn.ReLU(), nn.Linear(u_dim, int(u_dim/2)),MCDropout(p=dropout_p), nn.ReLU(), nn.Linear(int(u_dim/2),1))
            self.decoder = MLPSimple(input_dim = h_dim+1, output_dim = h_dim, hidden_dim = 4*h_dim, depth = 4, activations = [nn.ReLU() for _ in range(4)])
            self.treatment_fun = MLPSimple(input_dim = 1, output_dim = 1, hidden_dim = u_dim, depth = 4, activations = [nn.ReLU() for _ in range(4)] )
        
        elif (not continuous_treatment):

            #self.decoder = nn.Sequential(nn.Linear(h_dim+u_dim,h_dim+u_dim),nn.ReLU(), nn.Linear(h_dim+u_dim,h_dim+u_dim), nn.ReLU(), nn.Linear(h_dim+u_dim, h_dim+u_dim), nn.ReLU(),MCDropout(p=dropout_p), nn.Linear(h_dim+u_dim,h_dim+u_dim))
            self.decoder = MLPSimple(input_dim = h_dim + u_dim , output_dim = h_dim+u_dim, hidden_dim = 2*(h_dim + u_dim), depth = 4, activations = [nn.ReLU() for _ in range(4)])

        self.h_dim = h_dim
        self.u_dim = u_dim

        self.continuous_treatment = continuous_treatment
        self.fun_treatment = fun_treatment

    def forward_ode(self,h,t):
        h_gen = h[...,:self.h_dim_gen]
        u = h[...,self.u_dim:]
        h_ = self.h_fun(h_gen)
        u_ = self.u_fun(u)
        return torch.cat((h_,u_),-1)

    def forward_fun_treatment(self,h,t):
        treatment_impact = self.treatment_fun(torch.tile(t,(h.shape[0],1)))
        treatment_impact[(h[:,self.h_dim:]==0).all(1)] = 0.
        x_ = torch.cat((h[:,:self.h_dim],treatment_impact),1)
        x_out = self.decoder(x_)
        return torch.cat((self.decoder(x_), torch.zeros_like(x_out)),1)
    
    def forward_non_continuous(self,h,t):
        return self.decoder(h)

    def forward(self,h,t): 
        if ((self.continuous_treatment) and (not self.fun_treatment)):
            return self.forward_ode(h,t)
        elif (self.continuous_treatment and self.fun_treatment):
            return self.forward_fun_treatment(h,t)
        elif (not self.continuous_treatment):
            return self.forward_non_continuous(h,t)


class MLPSimple(nn.Module):
    def __init__(self,input_dim,output_dim, hidden_dim, depth, activations = None, dropout_p = None):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim,output_dim))
        if activations is None:
            activations = [nn.ReLU() for _ in range(depth)]
        if dropout_p is None:
            dropout_p = [0. for _ in range(depth)]
        assert len(activations) == depth
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Dropout(dropout_p[i]),activations[i]) for i in range(depth)])
    def forward(self,x):
        x = self.input_layer(x)
        for mod in self.layers:
            x = mod(x)
        x = self.output_layer(x)
        return x


class RNNModule(pl.LightningModule):

    def __init__(self,input_dim,  output_dim, hidden_dim, depth, embedding_dim, horizon, dropout_p, lr, ODE_mode, checkpoint_propensity, continuous_treatment_ode, exact_iptw, fun_treatment, linear_output_fn, ipm_regul, std_dev, cf_var_regul, MLP_decoding_mode = False, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.encoder = torch.nn.GRU(input_dim, hidden_dim,num_layers = depth, batch_first = True, dropout = dropout_p)
        self.out_encoder = torch.nn.Linear(hidden_dim, embedding_dim)
        self.treatment_embedding = nn.Embedding(num_embeddings = 2, embedding_dim = embedding_dim)

        if kwargs.get("static_x",False):
            self.static_x = True #Only static covariates are available
            self.encoder = MLPSimple(input_dim= input_dim, output_dim= embedding_dim, hidden_dim = hidden_dim, depth = 3)
        else:
            self.static_x = False
        self.dropout_p = dropout_p
        if std_dev:
            outdim_factor = 2
        else:
            outdim_factor = 1

        self.output_dim = outdim_factor * output_dim 
        
        self.std_dev = std_dev

        self.MLP_decoding_mode = MLP_decoding_mode
        
        if fun_treatment:
            assert continuous_treatment_ode #fun treatment only makes sense if the treatment is continuous
        if not ODE_mode:
            if MLP_decoding_mode: # extra entry for the times at which to evaluate the model.
                self.decoder = MLPSimple(input_dim= 1 + 2*embedding_dim, output_dim= self.output_dim , hidden_dim = embedding_dim, depth = 3)
            else:
                self.decoder = MLPSimple(input_dim= 2*embedding_dim, output_dim= self.output_dim * horizon , hidden_dim = embedding_dim, depth = 3)
        else:
            self.decoder = ContinuousTreatmentODE(embedding_dim,embedding_dim,1,continuous_treatment_ode, fun_treatment, dropout_p = dropout_p)
            if continuous_treatment_ode:
                if linear_output_fn:
                    self.output_fun = nn.Linear(embedding_dim,outdim_factor)
                else:
                    #self.output_fun = nn.Sequential(nn.Linear(embedding_dim,embedding_dim),nn.ReLU(), nn.Linear(embedding_dim,embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, int(embedding_dim/2)), nn.Tanh(), nn.Linear(int(embedding_dim/2),outdim_factor))
                    self.output_fun = MLPSimple(input_dim = embedding_dim, output_dim= self.output_dim , hidden_dim = embedding_dim, depth = 3, activations = [nn.ReLU(),nn.ReLU(),nn.Tanh()])

            else:
                
                #self.output_fun = nn.Sequential(nn.Linear(2*embedding_dim,embedding_dim),nn.ReLU(), nn.Linear(embedding_dim,embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim,outdim_factor))
                self.output_fun = MLPSimple(input_dim= 2*embedding_dim, output_dim= self.output_dim , hidden_dim = embedding_dim, depth = 3, activations = [nn.ReLU(),nn.ReLU(),nn.Tanh()])
        
        self.lr = lr

        self.horizon = horizon

        if self.std_dev:
            self.loss = torch.nn.GaussianNLLLoss()
        else:
            self.loss = nn.MSELoss(reduction = "none")
        
        self.MSE_loss = nn.MSELoss(reduction = "none")

        self.ODE_mode = ODE_mode

        self.continuous_treatment = continuous_treatment_ode

        self.embedding_dim = embedding_dim

        if checkpoint_propensity is not None:
            self.propensity_model = PropensityScoreModule.load_from_checkpoint(checkpoint_propensity, input_dim = input_dim, hidden_dim = hidden_dim, depth= depth, embedding_dim = embedding_dim, dropout_p = dropout_p, lr=lr )
            self.propensity_model.freeze()
        else:
            self.propensity_model = None

        self.exact_iptw = exact_iptw

        self.fun_treatment = fun_treatment

        self.linear_output_fn = linear_output_fn

        if ipm_regul:
            self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
            #self.lambda_reg = lamdba_reg 
            self.alpha_reg = kwargs["alpha_reg"]
            self.norm_encoding = kwargs["norm_encoding"]
        else:
            self.alpha_reg = 0
            self.norm_encoding = False
        self.ipm_regul = ipm_regul

        self.cf_var_regul = cf_var_regul

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer


    def compute_iptw_weights(self,X,T,p):

        if self.propensity_model is not None:
            weights = (1/(T.unsqueeze(-1)*self.propensity_model(X) + (1-T.unsqueeze(-1))*(1-self.propensity_model(X))))[:,None,:]
        elif self.exact_iptw:
            weights = (1/(T.unsqueeze(-1)*p[:,None] + (1-T.unsqueeze(-1))*(1-p[:,None])))[:,None,:]
        else:
            weights = 1
        return weights


    def compute_wass(self,encoding,T):
        X_t = encoding[T==1]
        X_nt = encoding[T==0]

        dist, C, P = self.sinkhorn(X_t,X_nt)
        
        return dist

    def forward(self,X,T, return_decoded_u = False, times= None):
        """
        return_decoded_u : if true, will return the treatment driving process (only present when ODE_mode is true and continuous_treatment is True).
        times : the times at which we wish to evalute the ODE (if None : torch.arange(horizon)+1)
        """
        if self.static_x: # MLP encoder
            encoding_ts = self.encoder(X)
        else: # RNN encoder
            output, hn = self.encoder(X)
            encoding_ts = self.out_encoder(hn)[-1]

        if self.ipm_regul:
            if self.norm_encoding:
                norm_encoding = torch.sqrt(encoding_ts.pow(2).sum(axis=1))[:,None]
                encoding_ts = encoding_ts/norm_encoding
            
            reg = self.compute_wass(encoding_ts,T)
        else:
            reg = 0

        treatment_embeds = self.treatment_embedding(T.long())
        if self.continuous_treatment:
            treatment_embeds[T==0] = 0
        encoding = torch.cat((encoding_ts,treatment_embeds),-1)
        if self.ODE_mode:
            decoding, decoded_u = self.forward_ODE(encoding, times)
            if return_decoded_u:
                return decoding, decoded_u
        else:
            if self.MLP_decoding_mode:
                t = 1 + torch.arange(self.horizon, device = self.device).float()
                t_tiled = torch.tile(t[:,None,None],(1,)+encoding.shape[:1]+(1,))
                encoding_tiled = torch.tile(encoding,(self.horizon,1,1))
                cat_input = torch.cat((encoding_tiled,t_tiled),-1)
                in_dim = cat_input.shape[-1]
                batch_dim = encoding.shape[0]
                decoding  = self.decoder(cat_input).permute(1,0,2)
            else:
                decoding = self.decoder(encoding).reshape(encoding.shape[0],-1,self.output_dim)
        
        return decoding, reg

    def ode_fun(self,t,x):
        #if self.fun_treatment:
        #    treatment_impact = self.treatment_fun(torch.tile(t,(x.shape[0],1)))
        #    treatment_impact[(x[:,self.embedding_dim:]==0).all(1)] = 0.
        #    x_ = torch.cat((x[:,:self.embedding_dim],treatment_impact),1)
        #    x_out = self.decoder(x_)
        #    return torch.cat((self.decoder(x_), torch.zeros_like(x_out)),1)
        return self.decoder(x,t)        


    def forward_ODE(self, encoding, times = None, return_hiddens = False):
        if times is not None:
            t = times
        else:
            t = torch.arange(self.horizon+1).float()
        #out = odeint(self.ode_fun,encoding,t, atol=1e-10, rtol=1e-8, method = "dopri5")[1:,:,:] # Num time points (horizon) x batch_size x dimension
        out = odeint(self.ode_fun,encoding,t, options= {"step_size":0.1}, method = "rk4")[1:,:,:] # Num time points (horizon) x batch_size x dimension
        if self.continuous_treatment:
            out_u = out[...,self.embedding_dim:]
            out = out[...,:self.embedding_dim]
        else:
            out_u = None
        out_x = self.output_fun(out)[:,:,:].permute(1,0,2) # batch_size x Num time points (horizon)
        if return_hiddens:
            return out_x, out_u, out
        return out_x, out_u


    def training_step(self, batch, batch_idx):
        X, Y, T, Y_cf, p, thetas_0, t_X, t_Y = batch

        Y_hat, reg = self(X,T)
        
        if self.cf_var_regul !=0:

            T_cf = (~T.bool()).long()
            Y_hat_cf,_ = self(X,T_cf)
        
            loss, mse, mse_cf, mse_ite = self.compute_losses(X,Y,Y_cf,T,Y_hat,Y_hat_cf,reg,p)
        else:
            loss, mse = self.compute_losses(X,Y,Y_cf,T,Y_hat,None,reg,p)

        self.log("train_loss",loss, on_step = True, on_epoch = True)
        self.log("train_rmse",mse, on_step = True, on_epoch = True)
        
        return loss

    def compute_losses(self,X,Y,Y_cf, T,Y_hat,Y_hat_cf,reg,p):
        if self.std_dev:
            mu, var = torch.chunk(Y_hat,2,-1)
            fact_loss = self.loss(Y[:,:self.horizon,:],mu,torch.sigmoid(var))
            mse = torch.sqrt(self.MSE_loss(Y[:,:self.horizon,:],mu)).mean()
        else:
            fact_loss = self.loss(Y[:,:self.horizon,:],Y_hat)
            mse = torch.sqrt(fact_loss).mean()
        
        loss = fact_loss + self.alpha_reg * reg

        weights = self.compute_iptw_weights(X,T,p)
        loss = (weights*loss).mean()
        
        if Y_hat_cf is not None:
            #CF loss
            T_cf = (~T.bool()).long()

            if self.std_dev:
                mu_cf, var = torch.chunk(Y_hat_cf,2,-1)
                mse_cf = torch.sqrt(self.MSE_loss(Y_cf[:,:self.horizon,:],mu_cf)).mean()
                if self.cf_var_regul !=0:
                    loss += self.cf_var_regul * var.mean()
            else:
                mse_cf = torch.sqrt(self.loss(Y_cf[:,:self.horizon,:],Y_hat_cf)).mean()

            #ITE
            ite  = (Y_cf[:,:self.horizon,:] - Y[:,:self.horizon,:])
            
            if self.std_dev:
                ite_hat = mu_cf - mu
                mse_ite = torch.sqrt(self.MSE_loss(ite, ite_hat)).mean()
            else:
                ite_hat = (Y_hat_cf-Y_hat)
                mse_ite = torch.sqrt(self.loss(ite, ite_hat)).mean()

            return loss, mse, mse_cf, mse_ite

        return loss, mse

    def validation_step(self, batch, batch_idx):
        X, Y, T, Y_cf, p, thetas_0, t_X, t_Y = batch
        Y_hat,reg = self(X,T)

        T_cf = (~T.bool()).long()
        Y_hat_cf,_ = self(X,T_cf)
        
        loss, mse, mse_cf, mse_ite = self.compute_losses(X,Y,Y_cf,T,Y_hat,Y_hat_cf,reg,p)
        
        self.log("val_loss",loss, on_epoch = True)
        self.log("val_rmse", mse, on_epoch = True)


        self.log("val_rmse_cf",mse_cf, on_epoch = True)

        self.log("Val PEHE", mse_ite, on_epoch = True)
        
        #Plotting
        if batch_idx==0:
            self.plot_trajectories( X, Y, Y_cf, Y_hat, Y_hat_cf, chart_type = "val" )

        return {"RMSE_val": mse, "RMSE_val_cf": mse_cf}

    def test_step(self, batch, batch_idx):
        X, Y, T, Y_cf, p, thetas_0, t_X, t_Y = batch
        Y_hat, reg = self(X,T)

        T_cf = (~T.bool()).long()
        Y_hat_cf, _ = self(X,T_cf)
        
        loss, mse, mse_cf, mse_ite = self.compute_losses(X,Y,Y_cf,T,Y_hat,Y_hat_cf,reg,p)
        
        self.log("test_loss",loss, on_epoch = True)
        self.log("test_rmse", mse, on_epoch = True)

        self.log("test_rmse_cf",mse_cf, on_epoch = True)

        self.log("Test PEHE", mse_ite, on_epoch = True)       
        
        if batch_idx==0:
            self.plot_trajectories( X, Y, Y_cf, Y_hat, Y_hat_cf, chart_type = "test" )

        return {"RMSE_test": mse, "RMSE_test_cf": mse_cf}

    def plot_trajectories(self, X, Y, Y_cf, Y_hat, Y_hat_cf, chart_type = "val" ):
       
        X_df_list = []
        for dim in range(X.shape[-1]):
            if self.static_x:
                X_df = pd.DataFrame(columns = ["CancerVolume","time","type"])
                break
            else:
                X_df = pd.DataFrame(X.cpu().numpy()[0,:,dim], columns = ["CancerVolume"])
                X_df["time"]  = np.arange(X_df.shape[0])
                X_df["type"] = f"Input_{dim}"
                X_df_list.append(X_df)
        X_df = pd.concat(X_df_list)
        time_x_end = X_df.time.max() + 1
        
        for dim in range(Y.shape[-1]):

            if self.std_dev:
                Y_hat, var_hat = torch.chunk(Y_hat,2,-1)
                Y_hat_cf, var_hat_cf = torch.chunk(Y_hat_cf,2,-1)
                std_hat = torch.sqrt(torch.sigmoid(var_hat)[0,:]).cpu().numpy()
                std_hat_cf = torch.sqrt(torch.sigmoid(var_hat_cf)[0,:]).cpu().numpy()
            else:
                std_hat = 0
                std_hat_cf = 0

            Y_fact_df = pd.DataFrame(Y.cpu().numpy()[0,:,dim], columns = ["CancerVolume"])
            Y_fact_df["time"]  = time_x_end + np.arange(Y_fact_df.shape[0])
            Y_fact_df["type"] = "Factual"

            Y_cfact_df = pd.DataFrame(Y_cf.cpu().numpy()[0,:,dim], columns = ["CancerVolume"])
            Y_cfact_df["time"]  = time_x_end + np.arange(Y_cfact_df.shape[0])
            Y_cfact_df["type"] = "Counterfactual"

            Y_hat_fact_df = pd.DataFrame(Y_hat.cpu().numpy()[0,:,dim], columns = ["CancerVolume"])
            Y_hat_fact_df["time"]  = time_x_end + np.arange(Y_hat_fact_df.shape[0])
            Y_hat_fact_df["type"] = "Factual Prediction"
            Y_hat_fact_df["std"] = std_hat

            Y_hat_cfact_df = pd.DataFrame(Y_hat_cf.cpu().numpy()[0,:,dim], columns = ["CancerVolume"])
            Y_hat_cfact_df["time"]  = time_x_end + np.arange(Y_hat_cfact_df.shape[0])
            Y_hat_cfact_df["type"] = "Counterfactual Prediction"
            Y_hat_cfact_df["std"] = std_hat_cf
            
            df = pd.concat([X_df, Y_fact_df, Y_cfact_df, Y_hat_fact_df, Y_hat_cfact_df])
            fig = px.line(df, x = "time",y = "CancerVolume",color="type", error_y ="std", title = f"{chart_type} longitudinal predictions - dimension {dim}") 

            if type(self.logger).__name__ == "WandbLogger":
                self.logger.experiment.log({f"{chart_type} chart {dim}":fig},step = self.logger.experiment.step)
            else:
                print("Missing graph")

    @classmethod
    def add_model_specific_args(cls, parent):
        parser = argparse.ArgumentParser(parents=[parent])
        parser.add_argument("--hidden_dim", type=int, default=146)
        parser.add_argument("--depth", type=int, default=2)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--dropout_p", type=float, default=0.0)
        parser.add_argument("--embedding_dim", type = int, default = 16)
        parser.add_argument("--horizon", type = int, default = 5)
        parser.add_argument("--ODE_mode",type=str2bool,default = False)
        parser.add_argument("--fun_treatment",type=str2bool,default = False, help = "If true, the treatment response is an arbitrary function ( parametrized by a MLP )")
        parser.add_argument("--linear_output_fn",type=str2bool,default = False, help = "If true, the emission function is linear")
        parser.add_argument("--ipm_regul",type=str2bool,default = False, help = "If true, uses IPM regularization")
        parser.add_argument("--alpha_reg",type=float,default = 0.0, help = "Strength of alpha regularization. Only used if ipm_regul is True")
        parser.add_argument("--norm_encoding",type=str2bool,default = True, help = "If true, regularises the encoding. Only used if ipm_regul is True")
        parser.add_argument("--std_dev",type=str2bool,default = False, help = "If yes, stdandard deviations of the predictions are also returned and the model is trained with likelihood")
        parser.add_argument("--continuous_treatment_ode",type=str2bool,default = False, help = "If yes, uses an external ODE to model the treatments or an external function")
        parser.add_argument("--cf_var_regul",type=float,default = 0.0, help = "Counterfactual variance loss term")
        parser.add_argument("--MLP_decoding_mode",type=str2bool,default = False, help = "If true, uses an MLP with the times as input for the decoding")
        
        return parser



class ODE_head(nn.Module):
    def __init__(self,hidden_dim,depth_ode, depth_outfun, horizon):
        super().__init__()

        self.ode_fun_list = torch.nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU()) for _ in range(depth_ode)]+[nn.Linear(hidden_dim,hidden_dim)])
        
        self.output_fun_list = torch.nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU()) for _ in range(depth_outfun)]+[nn.Linear(hidden_dim,1)])

        self.horizon = horizon
    def ode_fun(selt,t,x):
        for mod in self.ode_fun_list:
            x = mod(x)
        return x

    def output_fun(self,x):
        for mod in self.output_fun_list:
            x = mod(x)
        return x

    def forward(self, encoding, times = None, return_hiddens = False):
        if times is not None:
            t = times
        else:
            t = torch.arange(self.horizon+1).float()
        out = odeint(self.ode_fun,encoding,t, atol=1e-10, rtol=1e-8, method = "dopri5")[1:,:,:] # Num time points (horizon) x batch_size x dimension
        out_x = self.output_fun(out)[:,:,:].permute(1,0,2) # batch_size x Num time points (horizon)
        if return_hiddens:
            return out_x, out
        return out_x


class LatentSDE(torchsde.SDEIto):
    def __init__(self,theta,mu, sigma, embedding_dim):
        super().__init__(noise_type="diagonal")
    
        self.theta = theta
        self.mu = mu
        self.register_buffer("sigma", torch.tensor([[sigma]]))
        
        u_dim = int(embedding_dim/5)
        self.treatment_fun = MLPSimple(input_dim = 1, output_dim = u_dim, hidden_dim = 20, depth = 4, activations = [nn.ReLU() for _ in range(4)] )
        
        self.sde_drift = MLPSimple(input_dim = embedding_dim + u_dim, output_dim = embedding_dim, hidden_dim = 4*embedding_dim, depth = 4, activations = [nn.Tanh() for _ in range(4)])
   
    
    def fun_treatment(self,t):
        return self.treatment_fun(t)

    def g(self,t,y):
        return self.sigma.repeat(y.shape)

    def h(self,t,y):
        return self.theta * (self.mu-y)

    def f(self,t,y, T):
        u = self.fun_treatment(t.repeat(y.shape[0],1))
        u_t = u * T[:,None]
        y_and_u = torch.cat((y,u_t),-1)
        return self.sde_drift(y_and_u) - self.h(t,y)
    
    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        T = y[:,-1]
        y = y[:, 0:-2]
        f, g, h = self.f(t, y, T), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp, torch.zeros_like(f_logqp)], dim=1)
    
    def h_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        T = y[:,-1]
        y = y[:, 0:-2]
        f, g, h = self.f(t, y, T), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([h, f_logqp, torch.zeros_like(f_logqp)], dim=1)
    
    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:-2]
        g = self.g(t, y)
        g_logqp = torch.zeros((y.shape[0],2), device = y.device)
        return torch.cat([g, g_logqp], dim=1)



class VariationalSDE(RNNModule):

    def __init__(self,input_dim,  output_dim, hidden_dim, depth, embedding_dim, horizon, dropout_p,lr, ODE_mode, checkpoint_propensity, continuous_treatment_ode, exact_iptw, fun_treatment, linear_output_fn, ipm_regul, std_dev, cf_var_regul,  sigma_sde, output_scale, kl_param = 0.01 , start_scheduler=500, iter_scheduler = 1000, ood_fact = 0, num_samples = 1, mc_dropout = 0., **kwargs):
        super().__init__(input_dim,  output_dim, hidden_dim, depth, embedding_dim, horizon, dropout_p,lr, ODE_mode, checkpoint_propensity, continuous_treatment_ode, exact_iptw, fun_treatment, linear_output_fn, ipm_regul, std_dev, cf_var_regul, **kwargs)
 
        self.save_hyperparameters()
        self.sde = LatentSDE(theta=0.1, mu = 0, sigma = sigma_sde, embedding_dim = embedding_dim)

        self.output_fun = MLPSimple(input_dim = embedding_dim, output_dim= self.output_dim , hidden_dim = embedding_dim, depth = 3, activations = [nn.ReLU(),nn.ReLU(),nn.Tanh()], dropout_p = [dropout_p,dropout_p,dropout_p])
        #self.output_fun = nn.Sequential(nn.Linear(embedding_dim,embedding_dim),MCDropout(p=dropout_p),nn.ReLU(),nn.Linear(embedding_dim,embedding_dim), nn.Tanh(), nn.Linear(embedding_dim,self.output_dim))
        #self.output_fun = nn.Sequential(nn.Linear(embedding_dim,embedding_dim),nn.ReLU(),nn.Linear(embedding_dim,embedding_dim), nn.Tanh(), nn.Linear(embedding_dim,self.output_dim))
        #self.output_fun = nn.Linear(embedding_dim,self.output_dim)

        self.output_scale = torch.tensor([output_scale], requires_grad = False, device = self.device)
        
        self.loss = GaussianNLLLoss(reduction = "none")

        self.kl_scheduler = LinearScheduler(start = start_scheduler, iters = iter_scheduler)

        self.kl_param = kl_param

        self.num_samples = num_samples

        self.ood_fact = ood_fact
        
        self.mc_dropout = mc_dropout
        if self.mc_dropout >0:
            self.dropout_lin_layer = nn.Linear(embedding_dim, embedding_dim)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        
        scheduler = {"monitor": "val_loss", "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode = "min", factor = 0.5, patience = 50, verbose = True)}
        return {"optimizer": optimizer, "lr_scheduler":scheduler}
    

    def forward_sde(self,x0,times,T, prior_mode = False):
        """
        If prior_mode is True : integrates using the prior SDE
        """
        if times is not None:
            t = times
        else:
            t = torch.arange(self.horizon+1,device = self.device).float()
        
        batch_size = x0.shape[1]
        aug_y0 = torch.cat([x0, torch.zeros(self.num_samples,batch_size, 1).to(x0), T.repeat(self.num_samples,1)[...,None].to(x0)], dim=-1)
        dim_aug = aug_y0.shape[-1]
        
        aug_y0 = aug_y0.reshape(-1,dim_aug)
        #aug_y0 = aug_y0[0]

        if prior_mode:
            drift = "h_aug"
        else:
            drift = "f_aug"
        
        aug_ys = torchsde.sdeint(sde=self.sde,
                y0=aug_y0,
                ts=t,
                method="euler",
                dt=0.05,
                adaptive=False,
                rtol=1e-3,
                atol=1e-3,
                names={'drift': drift, 'diffusion': 'g_aug'})

        aug_ys = aug_ys.reshape(len(t),self.num_samples,-1,dim_aug) # reshape for # times x num_samples x batch_size x dim
        #aug_ys = aug_ys[:,None,...]

        ys, logqp_path = aug_ys[:, :, :, 0:-2], aug_ys[-1,: , :, -2]
        logqp0 = 0
        logqp = (logqp0 + logqp_path)  # KL(t=0) + KL(path).
        preds = self.output_fun(ys).permute(1,2,0,3) # num_samples x batch x times x dims

        return preds[:,:,1:,:], logqp # dropping the first time step (=0)
        
    def forward(self,X,T, return_decoded_u = False, times= None):
        """
        return_decoded_u : if true, will return the treatment driving process (only present when ODE_mode is true and continuous_treatment is True).
        times : the times at which we wish to evalute the ODE (if None : torch.arange(horizon)+1)
        """
        if self.static_x: # MLP encoder
            encoding_ts = self.encoder(X.float())
        else: # RNN encoder
            output, hn = self.encoder(X)
            encoding_ts = self.out_encoder(hn)[-1]

        if self.mc_dropout>0:
            encoding_ts = F.dropout(self.dropout_lin_layer(encoding_ts),self.mc_dropout, True, False)

        if self.ipm_regul:
            if self.norm_encoding:
                norm_encoding = torch.sqrt(encoding_ts.pow(2).sum(axis=1))[:,None]
                encoding_ts = encoding_ts/norm_encoding
            
            reg = self.compute_wass(encoding_ts,T)
        else:
            reg = 0
        
        encoding= encoding_ts
        encoding = encoding.repeat(self.num_samples,1,1)
        decoding, logqp = self.forward_sde(encoding,times, T)

        return decoding.contiguous() , reg, logqp

    def compute_losses(self,X,Y,Y_cf, T,Y_hat,Y_hat_cf,reg,p, logqp, ood = False):
        if self.std_dev:
            mu, var = torch.chunk(Y_hat,2,-1)
            Y_true = Y[:,:self.horizon,:].repeat(self.num_samples,1,1,1)
            fact_loss = self.loss(Y_true,mu,torch.sigmoid(var))
            mse = torch.sqrt(self.MSE_loss(Y[:,:self.horizon,:],mu.mean(0))).mean()
            std_preds = torch.sqrt(torch.sigmoid(var)).mean()
        else:
            Y_true = Y[:,:self.horizon,:].repeat(self.num_samples,1,1,1)
            fact_loss = self.loss(Y_true,Y_hat, self.output_scale.repeat(Y_hat.shape).to(self.device))
            fact_loss = fact_loss.sum((2,3))
            mse = torch.sqrt(self.MSE_loss(Y[:,:self.horizon,:],Y_hat.mean(0))).mean()
            std_preds = Y_hat.std(0).mean()


        loss = fact_loss.mean() + self.alpha_reg * reg + self.kl_param * logqp.mean() * self.kl_scheduler.val 
        
        weights = self.compute_iptw_weights(X,T,p)
        loss = (weights*loss).mean()
        
                
        if Y_hat_cf is not None:
            #CF loss
            T_cf = (~T.bool()).long()

            if self.std_dev:
                mu_cf, var = torch.chunk(Y_hat_cf,2,-1)
                mse_cf = torch.sqrt(self.MSE_loss(Y_cf[:,:self.horizon,:],mu_cf.mean(0))).mean()
                std_preds_cf = torch.sqrt(torch.sigmoid(var)).mean()
                if self.cf_var_regul !=0:
                    raise("Not Implemented - changes required for the cf_var regul with SDE")
                    loss += self.cf_var_regul * var.mean()
            else:
                mse_cf = torch.sqrt(self.MSE_loss(Y_cf[:,:self.horizon,:],Y_hat_cf.mean(0))).mean()
                std_preds_cf = Y_hat_cf.std(0).mean()
                if self.cf_var_regul !=0:
                    loss += self.cf_var_regul * std_preds_cf

            #ITE
            ite  = (Y_cf[:,:self.horizon,:] - Y[:,:self.horizon,:])
            
            if self.std_dev:
                ite_hat = mu_cf.mean(0) - mu.mean(0)
                mse_ite = torch.sqrt(self.MSE_loss(ite, ite_hat)).mean()
            else:
                ite_hat = (Y_hat_cf.mean(0)-Y_hat.mean(0))
                mse_ite = torch.sqrt(self.MSE_loss(ite, ite_hat)).mean()

            return loss, mse, mse_cf, mse_ite, std_preds, std_preds_cf

        return loss, mse, std_preds

    def process_irregular_ts(self, X,T,t_Y):
        """
        This function processes the predictions in case of irregular sampling.
        First, we order the unique time steps in the mini-batch and integrate the SDE at those unique times
        Then, we reassign the predicted values for each time in the batch.
        """
        t_Y = t_Y[:,:self.horizon]
        un_sorted = torch.sort(torch.unique(t_Y))[0]
        t_idx = (t_Y[...,None]==un_sorted).int().argmax(dim=-1)
        Y_hat, reg, logqp = self(X,T,times = torch.cat((torch.zeros(1,device = un_sorted.device),un_sorted)))
        t_idx_ = torch.tile(t_idx[None,...,None],(Y_hat.shape[0],1,1,Y_hat.shape[-1]))
        Y_hat_gathered = torch.gather(Y_hat,dim=2,index = t_idx_)
        return Y_hat_gathered, reg, logqp
        

    def training_step(self, batch, batch_idx):
        X, Y, T, Y_cf, p, thetas_0, t_X, t_Y = batch

        if len(torch.unique(t_Y))==t_Y.shape[1]: #time series are aligned
            Y_hat, reg, logqp = self(X,T)
        else:
            Y_hat, reg, logqp = self.process_irregular_ts(X,T,t_Y)
        
        if self.cf_var_regul !=0:

            T_cf = (~T.bool()).long()
            Y_hat_cf,_, _ = self(X,T_cf)
        
            loss, mse, mse_cf, mse_ite, std_preds, std_preds_cf = self.compute_losses(X,Y,Y_cf,T,Y_hat,Y_hat_cf,reg,p, logqp)
            self.log("train_std_preds_cf",std_preds_cf, on_step = True, on_epoch = True)
        else:
            loss, mse, std_preds = self.compute_losses(X,Y,Y_cf,T,Y_hat,None,reg,p, logqp)

        if self.ood_fact !=0 :
            X_ood = X + torch.randn(X.shape, device = self.device)
            Y_hat, reg, logqp = self(X_ood,T)
            _ , mse_ood, loss_ood = self.compute_losses(X_ood,Y,Y_cf,T,Y_hat,None,reg,p, logqp, ood = True)
            loss = loss + self.ood_fact * loss_ood
            self.log("train_loss_ood", loss_ood, on_step = True, on_epoch = True)

        self.log("train_loss",loss, on_step = True, on_epoch = True)
        self.log("train_rmse",mse, on_step = True, on_epoch = True)
        self.log("train_std_preds",std_preds, on_step = True, on_epoch = True)

        self.kl_scheduler.step()
        
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, T, Y_cf, p, thetas_0, t_X, t_Y = batch

        if len(torch.unique(t_Y))==t_Y.shape[1]: #time series are aligned
            Y_hat, reg, logqp = self(X,T)
        else:
            Y_hat, reg, logqp = self.process_irregular_ts(X,T,t_Y)

        T_cf = (~T.bool()).long()
        Y_hat_cf,_, _ = self(X,T_cf)
        
        loss, mse, mse_cf, mse_ite, std_preds, std_preds_cf = self.compute_losses(X,Y,Y_cf,T,Y_hat,Y_hat_cf,reg,p, logqp)
        
        if self.ood_fact !=0 :
            X_ood = X + torch.randn(X.shape, device = self.device)
            Y_hat, reg, logqp = self(X_ood,T)
            _ , mse_ood, loss_ood = self.compute_losses(X_ood,Y,Y_cf,T,Y_hat,None,reg,p, logqp, ood = True)
            loss = loss + loss_ood
            self.log("val_loss_ood", loss_ood, on_step = True, on_epoch = True)
        
        self.log("val_loss",loss, on_epoch = True)
        self.log("val_rmse", mse, on_epoch = True)
        self.log("val_std_preds",std_preds, on_step = True, on_epoch = True)
        self.log("val_std_preds_cf",std_preds_cf, on_step = True, on_epoch = True)


        self.log("val_rmse_cf",mse_cf, on_epoch = True)

        self.log("Val PEHE", mse_ite, on_epoch = True)
        
        #Plotting
        if batch_idx==0:
            self.plot_trajectories( X, Y, Y_cf, Y_hat[0], Y_hat_cf[0], chart_type = "val" )

        return {"RMSE_val": mse, "RMSE_val_cf": mse_cf}

    def test_step(self, batch, batch_idx):
        X, Y, T, Y_cf, p, thetas_0, t_X, t_Y = batch

        if len(torch.unique(t_Y))==t_Y.shape[1]: #time series are aligned
            Y_hat, reg, logqp = self(X,T)
        else:
            Y_hat, reg, logqp = self.process_irregular_ts(X,T,t_Y)

        T_cf = (~T.bool()).long()
        Y_hat_cf,_, _ = self(X,T_cf)
        
        loss, mse, mse_cf, mse_ite, std_preds, std_preds_cf = self.compute_losses(X,Y,Y_cf,T,Y_hat,Y_hat_cf,reg,p, logqp)
        
        if self.ood_fact !=0 :
            X_ood = X + torch.randn(X.shape, device = self.device)
            Y_hat, reg, logqp = self(X_ood,T)
            _ , mse_ood, loss_ood = self.compute_losses(X_ood,Y,Y_cf,T,Y_hat,None,reg,p, logqp, ood = True)
            loss = loss + loss_ood
            self.log("test_loss_ood", loss_ood, on_step = True, on_epoch = True)
        
        self.log("test_loss",loss, on_epoch = True)
        self.log("test_rmse", mse, on_epoch = True)
        self.log("test_std_preds",std_preds, on_step = True, on_epoch = True)
        self.log("test_std_preds_cf",std_preds_cf, on_step = True, on_epoch = True)

        self.log("test_rmse_cf",mse_cf, on_epoch = True)

        self.log("Test PEHE", mse_ite, on_epoch = True)
        
        #Plotting
        if batch_idx==0:
            self.plot_trajectories( X, Y, Y_cf, Y_hat[0], Y_hat_cf[0], chart_type = "val" )

        return {"RMSE_val": mse, "RMSE_val_cf": mse_cf}

    @classmethod
    def add_model_specific_args(cls, parent):
        parser = argparse.ArgumentParser(parents=[parent])
        parser.add_argument("--hidden_dim", type=int, default=146)
        parser.add_argument("--depth", type=int, default=2)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--dropout_p", type=float, default=0.0)
        parser.add_argument("--embedding_dim", type = int, default = 16)
        parser.add_argument("--horizon", type = int, default = 5)
        parser.add_argument("--ODE_mode",type=str2bool,default = False)
        parser.add_argument("--fun_treatment",type=str2bool,default = False, help = "If true, the treatment response is an arbitrary function ( parametrized by a MLP )")
        parser.add_argument("--linear_output_fn",type=str2bool,default = False, help = "If true, the emission function is linear")
        parser.add_argument("--ipm_regul",type=str2bool,default = False, help = "If true, uses IPM regularization")
        parser.add_argument("--alpha_reg",type=float,default = 0.0, help = "Strength of alpha regularization. Only used if ipm_regul is True")
        parser.add_argument("--norm_encoding",type=str2bool,default = True, help = "If true, regularises the encoding. Only used if ipm_regul is True")
        parser.add_argument("--std_dev",type=str2bool,default = False, help = "If yes, stdandard deviations of the predictions are also returned and the model is trained with likelihood")
        parser.add_argument("--continuous_treatment_ode",type=str2bool,default = False, help = "If yes, uses an external ODE to model the treatments or an external function")
        parser.add_argument("--cf_var_regul",type=float,default = 0.0, help = "Counterfactual variance loss term")
        parser.add_argument("--MLP_decoding_mode",type=str2bool,default = False, help = "If true, uses an MLP with the times as input for the decoding")
        parser.add_argument("--sigma_sde",type=float,default = 0.1, help = "Diffusion parameter in the SDE prior")
        parser.add_argument("--output_scale",type=float,default = 0.01, help = "standard deviation of the output_distribution")
        parser.add_argument("--kl_param",type=float,default = 0.01, help = "beta parameter for the KL divergence term (theoretically 1. In practice , ....)")
        parser.add_argument("--start_scheduler",type=int,default = 500, help = "iteration at which to start the linear scheduler for the KL term")
        parser.add_argument("--iter_scheduler",type=int,default = 1000, help = "number of iterations in the the scheduler for the KL term")
        parser.add_argument("--ood_fact",type=float,default = 0., help = "Regularization for the OOD loss")
        parser.add_argument("--num_samples",type=int,default = 1, help = "number of sde samples to draw at each pass")
        parser.add_argument("--mc_dropout",type=float,default = 0., help = "MCDropout in the encoder. Dropout probability to apply")
        
        return parser


