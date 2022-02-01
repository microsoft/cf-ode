import os.path
import torch
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

import numpy as np
import pandas as pd

import pickle
import sys
from torch import Tensor

def get_processed_data(raw_sim_data,
                       scaling_params):
    """
    Create formatted data to train both encoder and seq2seq atchitecture.
    """
    mean, std = scaling_params

    mean['chemo_application'] = 0
    mean['radio_application'] = 0
    std['chemo_application'] = 1
    std['radio_application'] = 1

    input_means = mean[
        ['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()
    input_stds = std[['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()

    # Continuous values
    cancer_volume = (raw_sim_data['cancer_volume'] - mean['cancer_volume']) / std['cancer_volume']
    cancer_volume_cf = (raw_sim_data["cancer_volume_cf"]-mean["cancer_volume"])/std["cancer_volume"]
    patient_types = (raw_sim_data['patient_types'] - mean['patient_types']) / std['patient_types']

    patient_types = np.stack([patient_types for t in range(cancer_volume.shape[1])], axis=1)

    # Binary application
    chemo_application = raw_sim_data['chemo_application']
    radio_application = raw_sim_data['radio_application']
    sequence_lengths = raw_sim_data['sequence_lengths']

    # Convert treatments to one-hot encoding

    treatments = chemo_application
    """
    treatments = np.concatenate(
        [chemo_application[:, :, np.newaxis], radio_application[:, :, np.newaxis]], axis=-1)

    one_hot_treatments = np.zeros(shape=(treatments.shape[0], treatments.shape[1], 3))
    for patient_id in range(treatments.shape[0]):
        for timestep in range(treatments.shape[1]):
            if (treatments[patient_id][timestep][0] == 0 and treatments[patient_id][timestep][1] == 0):
                one_hot_treatments[patient_id][timestep] = [1, 0, 0, 0]
            elif (treatments[patient_id][timestep][0] == 1 and treatments[patient_id][timestep][1] == 0):
                one_hot_treatments[patient_id][timestep] = [0, 1, 0, 0]
            elif (treatments[patient_id][timestep][0] == 0 and treatments[patient_id][timestep][1] == 1):
                one_hot_treatments[patient_id][timestep] = [0, 0, 1, 0]
            elif (treatments[patient_id][timestep][0] == 1 and treatments[patient_id][timestep][1] == 1):
                one_hot_treatments[patient_id][timestep] = [0, 0, 0, 1]

    one_hot_previous_treatments = one_hot_treatments[:, :-1, :]
    

    current_covariates = np.concatenate(
        [cancer_volume[:, :-offset, np.newaxis], patient_types[:, :-offset, np.newaxis]], axis=-1)
    """
    
    #outputs = cancer_volume[:, horizon:, np.newaxis]

    output_means = mean[['cancer_volume']].values.flatten()[0]  # because we only need scalars here
    output_stds = std[['cancer_volume']].values.flatten()[0]
    

    ## Add active entires
    #active_entries = np.zeros(outputs.shape)

    #for i in range(sequence_lengths.shape[0])

    raw_sim_data['treatments'] = treatments
    raw_sim_data['outputs'] = cancer_volume
    raw_sim_data["outputs_cf"] = cancer_volume_cf

    raw_sim_data['unscaled_outputs'] = (cancer_volume * std['cancer_volume'] + mean['cancer_volume'])
    raw_sim_data['unscaled_outputs_cf'] = (cancer_volume_cf * std['cancer_volume'] + mean['cancer_volume'])
    
    raw_sim_data['input_means'] = input_means
    raw_sim_data['inputs_stds'] = input_stds
    raw_sim_data['output_means'] = output_means
    raw_sim_data['output_stds'] = output_stds

    return raw_sim_data


def get_mse_at_follow_up_time(mean, output, active_entires):
        mses = np.sum(np.sum((mean - output) ** 2 * active_entires, axis=-1), axis=0) \
               / active_entires.sum(axis=0).sum(axis=-1)

        return pd.Series(mses, index=[idx for idx in range(len(mses))])


def train_BR_optimal_model(dataset_train, dataset_val, hyperparams_file, model_name, model_folder,
                           b_decoder_model=False):
    _, length, num_covariates = dataset_train['current_covariates'].shape
    num_treatments = dataset_train['current_treatments'].shape[-1]
    num_outputs = dataset_train['outputs'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 100}

    print("Loading best hyperparameters for model")
    with open(hyperparams_file, 'rb') as handle:
        best_hyperparams = pickle.load(handle)

    print("Best Hyperparameters")
    print(best_hyperparams)

    if (b_decoder_model):
        print(best_hyperparams)
        model = CRN_Model(params, best_hyperparams, b_train_decoder=True)
    else:
        model = CRN_Model(params, best_hyperparams)
    model.train(dataset_train, dataset_val, model_name=model_name, model_folder=model_folder)

def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * ((b>0).float()*2-1))
    return a / b


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0, start = 0):
            self._iters = max(1, iters)
            self._val = 0 # maxval / self._iters
            self._maxval = maxval
            self._start = start
            self.current_iter = 0
    
    def step(self):
        if self.current_iter>self._iters:
            self._val = min(self._maxval, self._val + self._maxval / self._iters)
        self.current_iter += 1

    @property
    def val(self):
        return self._val


def str2bool(value, raise_exc=False):

    _true_set = {'yes', 'true', 't', 'y', '1'}
    _false_set = {'no', 'false', 'f', 'n', '0'}
    
    if isinstance(value, str) or sys.version_info[0] < 3 and isinstance(value, basestring):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None


def gaussian_nll_loss(input, target, var, *, full=False, eps=1e-6, reduction='mean'):
    r"""Gaussian negative log likelihood loss.
    See :class:`~torch.nn.GaussianNLLLoss` for details.
    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full: ``True``/``False`` (bool), include the constant term in the loss
            calculation. Default: ``False``.
        eps: value added to var, for stability. Default: 1e-6.
        reduction: specifies the reduction to apply to the output:
            `'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    if not torch.jit.is_scripting():
        tens_ops = (input, target, var)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                gaussian_nll_loss, tens_ops, input, target, var, full=full, eps=eps, reduction=reduction)

    # Inputs and targets much have same shape
    #input = input.view(input.size(0), -1)
    #target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of var must match that of input or be equal to 1
    #var = var.view(input.size(0), -1)
    if var.size() != input.size():
        raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate loss (without constant)
    #loss = 0.5 * (torch.log(var) + (input - target)**2 / var).view(input.size(0), -1).sum(dim=1)
    loss = 0.5 * (torch.log(var) + (input - target)**2 / var)

    # Add constant to loss term if required
    if full:
        D = input.size(1)
        loss = loss + 0.5 * D * math.log(2 * math.pi)

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

class GaussianNLLLoss(torch.nn.modules.loss._Loss):
    r"""Gaussian negative log likelihood loss.

    The targets are treated as samples from Gaussian distributions with
    expectations and variances predicted by the neural network. For a
    ``target`` tensor modelled as having Gaussian distribution with a tensor
    of expectations ``input`` and a tensor of positive variances ``var`` the loss is:

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{input} - \text{target}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

    where :attr:`eps` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``var`` is not the same
    size as ``input`` (due to a homoscedastic assumption), it must either have a final dimension
    of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting.

    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``.
        eps (float, optional): value used to clamp ``var`` (see note below), for
            stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the
            utput:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input, or same shape as the input
          but with one dimension equal to 1 (to allow for broadcasting)
        - Var: :math:`(N, *)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting)
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Examples::
        >>> loss = nn.GaussianNLLLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 2, requires_grad=True) #heteroscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()

        >>> loss = nn.GaussianNLLLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 1, requires_grad=True) #homoscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()

    Note:
        The clamping of ``var`` is ignored with respect to autograd, and so the
        gradients are unaffected by it.

    Reference:
        Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
        target probability distribution", Proceedings of 1994 IEEE International
        Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
        vol.1, doi: 10.1109/ICNN.1994.374138.
    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        return gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)
