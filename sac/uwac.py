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



class UWACSAC(EnsembleSAC):
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

        self.beta = args.uwac_beta
        self.use_exp_weight = args.use_exp_weight
        self.clip_bottom = args.clip_bottom
        self.clip_top = args.clip_top
        self.factor = 1

    def get_weights(self, var, xi, feedback_type):
        if self.use_exp_weight:
            weight = torch.clamp(torch.exp(-self.beta * var/self.factor), self.clip_bottom, self.clip_top)
        else:
            weight = torch.clamp(self.beta*self.factor/var, self.clip_bottom, self.clip_top)
        return weight

 

class UWAC_VarEnsembleSAC(VarEnsembleSAC):
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

        self.beta = args.uwac_beta
        self.use_exp_weight = args.use_exp_weight
        self.clip_bottom = args.clip_bottom
        self.clip_top = args.clip_top
        self.factor = 1

    def get_weights(self, var, xi, feedback_type):
        if self.use_exp_weight:
            weight = torch.clamp(torch.exp(-self.beta * var/self.factor), self.clip_bottom, self.clip_top)
        else:
            weight = torch.clamp(self.beta*self.factor/var, self.clip_bottom, self.clip_top)
        return weight




 
