#!/usr/bin/env python
import argparse
import sys
sys.path.insert(0,"../")

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, MLFlowLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os

import models, data_utils, cv_data_utils, covid_data_utils
from azureml.core.run import Run
from utils import str2bool

def get_logger(logger_type, model_type, dataset_name, entity = "default"):
    if logger_type=="MLFlow":
        from azureml.core import Workspace

        run = Run.get_context()
        if "Offline" in run.id:
            ws = Workspace.from_config(path="~",_file_name="azure_config.json")
            experiment_name = f"MLFow_tests"
            logger = MLFlowLogger(experiment_name = experiment_name ,tracking_uri = ws.get_mlflow_tracking_uri(), artifact_location = "./mlflow_artifacts")

        else:
            ws = run.experiment.workspace
            experiment_name = run.experiment.name

            logger = MLFlowLogger(experiment_name = experiment_name ,tracking_uri = ws.get_mlflow_tracking_uri())
            logger._run_id = run.id
        
    elif logger_type=="wandb":
        logger = WandbLogger(
            name=f"{dataset_name}_{model_type}",
            project="causalode",
            entity=entity,
            log_model=False)
    elif logger_type=="tensorboard":
        logger = TensorBoardLogger(save_dir = "./logs",name = f"{dataset_name}_{model_type}")
    else:
        raise ValueError("Logger not supported")

    return logger

def get_logdir(logger,logger_type):
    if logger_type=="MLFlow":
        #log_dir = getattr(logger, "save_dir")
        log_dir = os.path.join("./mlflow_artifacts",logger.name,logger._run_id)
    else:
        log_dir = getattr(logger, 'log_dir', False) or logger.experiment.dir 

def main(model_cls, dataset_cls, args):
    
    # Instantiate objects according to parameters
    dataset = dataset_cls(**vars(args))
    dataset.prepare_data()

    input_dim = dataset.input_dim
    output_dim = dataset.output_dim
    gpu = args.gpu
    print('Running with hyperparameters:')
    print(args)

    
    if args.ODE_mode:
        model_type = "ODE"
    else:
        model_type = "RNN"

    logger = get_logger(args.logger_type, model_type, args.dataset_name, entity = args.entity)
    logger.log_hyperparams(args)

    log_dir = get_logdir(logger, args.logger_type)
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=log_dir,
        monitor='val_loss',
        mode='min',
        verbose=True,
        save_last = True,
    )

    early_stop_callback =   EarlyStopping(monitor='val_loss',
            min_delta=0.00,
            patience=200,
            verbose=False,
        mode='min'
        )

    if args.exact_iptw:
        args.propensity_scores = False
        print("Exact IPTW, propensity score training has been set to False")
    assert not (args.propensity_scores and args.exact_iptw) # those options are mutually exclusive.

    if args.propensity_scores:
        model_propensity = models.PropensityScoreModule(input_dim = input_dim, **vars(args))
        logger_propensity = get_logger(args.logger_type,"propensity", args.dataset_name)
        log_dir_prop = get_logdir(logger_propensity, args.logger_type)

        prop_checkpoint_cb = ModelCheckpoint(
        dirpath=log_dir,
        monitor='val_loss_iptw',
        mode='min',
        verbose=True
    )

        prop_early_stop_callback =   EarlyStopping(monitor='val_loss_iptw',
            min_delta=0.00,
            patience=15,
            verbose=False,
        mode='min'
        )

        trainer_prop = pl.Trainer(
        gpus=gpu,
        logger=logger_propensity,
        log_every_n_steps=5,
        max_epochs=args.max_epochs,
        callbacks=[prop_checkpoint_cb, prop_early_stop_callback]
    )
        trainer_prop.fit(model_propensity,datamodule = dataset)

        checkpoint_propensity_model_path = prop_checkpoint_cb.best_model_path
    else:
        checkpoint_propensity_model_path = None

    model = model_cls( input_dim = input_dim, output_dim = output_dim, checkpoint_propensity = checkpoint_propensity_model_path,
        **vars(args),
        )
    trainer = pl.Trainer(
        gpus=gpu,
        logger=logger,
        log_every_n_steps=5,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb, early_stop_callback]
    )
   
    trainer.fit(model, datamodule=dataset)
    
    test_results = trainer.test(ckpt_path="best", test_dataloaders=dataset.test_dataloader())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--max_epochs', type=int, default=250)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--model', default = "RNN", type = str)
    parser.add_argument('--logger_type', default = "tensorboard", type = str, help = "logger to use (wandb,MLFlow or tensorboard)")
    parser.add_argument('--propensity_scores', default = False, type = str2bool,help="train a model to estimate the iptw")
    parser.add_argument('--exact_iptw', default = False, type = str2bool, help = "uses the exact probabilities to compute the iptw")
    parser.add_argument('--dataset_name', default = "pendulum", type = str, help = "dataset to train on")
    parser.add_argument('--variational', default = False, type = str2bool, help = "if to use the variational approach to train the ODE")
    parser.add_argument('--entity', default = "edebrouwer", type = str, help = "name of the wandb logger entity")

    partial_args, _ = parser.parse_known_args()
    
    if partial_args.variational:
        model_cls = models.VariationalSDE
    else:
        model_cls = models.RNNModule
    
    if partial_args.dataset_name=="pendulum":
        dataset_cls = data_utils.PendulumDataModule
    elif partial_args.dataset_name=="cv":
        dataset_cls = cv_data_utils.CVDataModule
    elif partial_args.dataset_name=="covid":
        dataset_cls = covid_data_utils.CovidDataModule
    else:
        raise("Invalid dataset name")

    parser = model_cls.add_model_specific_args(parser)
    parser = dataset_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    main(model_cls, dataset_cls, args)


