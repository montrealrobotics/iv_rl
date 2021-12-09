from .sac import VarSACTrainer
from .ensembleSAC import EnsembleSAC, VarEnsembleSAC

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from scipy.optimize import minimize

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


def get_iv_weights(variances):
    '''
    Returns Inverse Variance weights
    Params
    ======
        variances (numpy array): variance of the targets
    '''
    weights = 1/variances
    (weights)
    weights = weights/np.sum(weights)
    (weights)
    return weights

def compute_eff_bs(weights):
    # Compute original effective mini-batch size
    eff_bs = 1/np.sum(np.square(weights))
    #print(eff_bs)
    return eff_bs

def get_optimal_xi(variances, minimal_size, epsilon_start):
    minimal_size = min(variances.shape[0] - 1, minimal_size)
    if compute_eff_bs(get_iv_weights(variances)) >= minimal_size:
        return 0        
    fn = lambda x: np.abs(compute_eff_bs(get_iv_weights(variances+np.abs(x))) - minimal_size)
    epsilon = minimize(fn, 0, method='Nelder-Mead', options={'fatol': 1.0, 'maxiter':100})
    xi = np.abs(epsilon.x[0])
    xi = 0 if xi is None else xi
    return xi


class IV_EnsembleSAC(EnsembleSAC):
    def __init__(
            self,
            args,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            num_ensemble,
            feedback_type,
            temperature,
            temperature_act,
            expl_gamma,
            log_dir,
        
            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            xi=None,
            dynamic_xi=False,
            minimal_eff_bs=None,):

        super().__init__(args,env,policy,qf1,qf2,target_qf1,target_qf2,num_ensemble,feedback_type,temperature,\
            temperature_act,expl_gamma,log_dir,discount,reward_scale,policy_lr,qf_lr,\
            optimizer_class,soft_target_tau,target_update_period,plotter,render_eval_paths,\
            use_automatic_entropy_tuning,target_entropy,xi,dynamic_xi,minimal_eff_bs)

    def iv_weights(self, variance, xi):
        weights = (1. / (variance+xi))
        weights /= weights.sum(0)
        return weights

    def get_weights(self, std, xi, feedback_type):
        if feedback_type == 0 or feedback_type == 1:
            weight_target_Q = self.iv_weights(std**2, xi)
        else:
            weight_target_Q = self.iv_weights(std**2, xi)

        return weight_target_Q

class IV_VarEnsembleSAC(VarEnsembleSAC):
    def __init__(
            self,
            args,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            num_ensemble,
            feedback_type,
            temperature,
            temperature_act,
            expl_gamma,
            log_dir,
        
            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            xi=None,
            dynamic_xi=False,
            minimal_eff_bs=None,):

        super().__init__(args,env,policy,qf1,qf2,target_qf1,target_qf2,num_ensemble,feedback_type,temperature,\
            temperature_act,expl_gamma,log_dir,discount,reward_scale,policy_lr,qf_lr,\
            optimizer_class,soft_target_tau,target_update_period,plotter,render_eval_paths,\
            use_automatic_entropy_tuning,target_entropy,xi,dynamic_xi,minimal_eff_bs)

    def iv_weights(self, variance, xi):
        weights = (1. / (variance+xi))
        weights /= weights.sum(0)
        return weights

    def get_weights(self, std, xi, feedback_type):
        if feedback_type == 0 or feedback_type == 1:
            weight_target_Q = self.iv_weights(std**2, xi)
        else:
            weight_target_Q = self.iv_weights(std**2, xi)

        return weight_target_Q




class IV_VarSAC(VarSACTrainer):
    def __init__(
            self,
            args,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__(args,env,policy,qf1,qf2,target_qf1,target_qf2,\
            discount,reward_scale,policy_lr,qf_lr,optimizer_class,\
            soft_target_tau,target_update_period,plotter,render_eval_paths,\
            use_automatic_entropy_tuning,target_entropy)

    def iv_weights(self, variance, xi):
        weights = (1. / (variance+xi))
        weights /= weights.sum(0)
        return weights

    def get_weights(self, var, xi):
        weight_target_Q = self.iv_weights(var, xi)
        return weight_target_Q

