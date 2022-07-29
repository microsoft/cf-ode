# Causal ODE package

This is the code repository of the AISTATS 2022 paper : "Predicting the impact of treatments over time with uncertainty awareneural differential equations."

## Package and dependencies installation

We support both conda environments and poetry types of environments.

### Conda

`conda create --name causalode --file requirements.txt`

### Poetry

Poetry is a python dependencies manager. You can install it following the instructions [here](https://python-poetry.org/docs/).

You can then install all the dependencies at once by running:

`poetry install`

**Attention** When using poetry as package dependencies, you'll have to add `poetry run` to your commands. Example:

`poetry run python train_model.py`


## Run the code

To run the experiments from the paper, you can use the parametrizations below. All datasets are generated on the fly. The different folds were created by changing the *seed* parameter.

**Harmonic Oscillator data set**

`python train_model.py --N_ts=1000 --batch_size=64 --continuous_treatment=True --embedding_dim=64 --gamma=8 --kl_param=0.001 --logger_type=wandb --max_epochs=2000 --num_samples=5 --propensity_scores=False --seed=44 --sigma_sde=0.1 --static=False --std_dev=False --variational=True`

**CardioVascular data set**

`python train_model.py --N_ts=1000 --batch_size=64 --continuous_treatment=True --dataset_name=cv --embedding_dim=64 --gamma=10 --horizon=10 --kl_param=0.001 --logger_type=wandb --max_epochs=2000 --num_samples=5 --propensity_scores=False --seed=44 --sigma_sde=0.1 --std_dev=False --variational=True`

**Dexamethasone data set**

`train_model.py --N_ts=1000 --batch_size=64 --continuous_treatment=True --dataset_name=covid --embedding_dim=64 --gamma=10 --horizon=10 --kl_param=0.001 --logger_type=wandb --max_epochs=2000 --num_samples=5 --propensity_scores=False --seed=43 --sigma_sde=0.1 --std_dev=False --variational=True`

## Process the results and make the figures

We provide jupyter notebooks to process the results of the experiments and generate the figures displayed in the paper.

- #### Evaluation CFODE.ipynb
computes the results reported in Table 1 in the paper.

- #### Figure 3.ipynb
creates the Figure 3 in the paper.

- #### CostTreatment.ipynb 
contains the experiments of section 5.3 about improving the treatment strategies with the derived uncertainties

- #### OOD.ipynb
contains the scripts for the experiment on out-of-distribution sampels as reported in the Appendix.

## Logging

All experiments have been logged with Weights and Biases but we also support MLFlow and Tensorboard. You change the type of logger by setting the `--logger_type` option.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
