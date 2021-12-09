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

def get_optimal_eps(variances, minimal_size, epsilon_start):
    minimal_size = min(variances.shape[0] - 1, minimal_size)
    if compute_eff_bs(get_iv_weights(variances)) >= minimal_size:
        return 0
    fn = lambda x: np.abs(compute_eff_bs(get_iv_weights(variances+np.abs(x))) - minimal_size)
    epsilon = minimize(fn, 0, method='Nelder-Mead', options={'fatol': 1.0, 'maxiter':100})
    eps = np.abs(epsilon.x[0])
    eps = 0 if eps is None else eps
    return eps





class EnsembleSAC(TorchTrainer):
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
            minimal_eff_bs=None,
    ):
        super().__init__()
        self.args = args
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        
        self.target_update_period = target_update_period
        
        self.num_ensemble = num_ensemble
        self.feedback_type = feedback_type
        self.temperature = temperature
        self.temperature_act = temperature_act
        self.expl_gamma = expl_gamma
        self.model_dir = log_dir + '/model/'
        
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.alpha_optimizer, self.log_alpha = [], []
            for _ in range(self.num_ensemble):
                log_alpha = ptu.zeros(1, requires_grad=True)
                alpha_optimizer = optimizer_class(
                    [log_alpha],
                    lr=policy_lr,
                )
                self.alpha_optimizer.append(alpha_optimizer)
                self.log_alpha.append(log_alpha)
                

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss(reduce=False)
        self.vf_criterion = nn.MSELoss(reduce=False)
        
        self.policy_optimizer, self.qf1_optimizer, self.qf2_optimizer, = [], [], []
        
        for en_index in range(self.num_ensemble):
            policy_optimizer = optimizer_class(
                self.policy[en_index].parameters(),
                lr=policy_lr,
            )
            qf1_optimizer = optimizer_class(
                self.qf1[en_index].parameters(),
                lr=qf_lr,
            )
            qf2_optimizer = optimizer_class(
                self.qf2[en_index].parameters(),
                lr=qf_lr,
            )
            self.policy_optimizer.append(policy_optimizer)
            self.qf1_optimizer.append(qf1_optimizer)
            self.qf2_optimizer.append(qf2_optimizer)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.xi = xi 
        self.dynamic_xi = dynamic_xi
        self.minimal_eff_bs = minimal_eff_bs

    def get_variance(self, obs, update_type):
        std_Q_list1, std_Q_list2 = [], []
        for en_index1 in range(self.num_ensemble):
            with torch.no_grad():
                policy_action, _, _, _, *_ = self.policy[en_index1](
                    obs, reparameterize=True, return_log_prob=True,
                )
            L_target_Q1, L_target_Q2 = [], []
            for en_index2 in range(self.num_ensemble):
                if update_type == 0: # actor
                    target_Q1 = self.qf1[en_index2](obs, policy_action)
                    target_Q2 = self.qf2[en_index2](obs, policy_action)
                else: # critic
                    target_Q1 = self.target_qf1[en_index2](obs, policy_action)
                    target_Q2 = self.target_qf2[en_index2](obs, policy_action)
                L_target_Q1.append(target_Q1)
                L_target_Q2.append(target_Q2)

            std_Q_list1.append(torch.stack(L_target_Q1).std(axis=0).detach())
            std_Q_list2.append(torch.stack(L_target_Q2).std(axis=0).detach())

        std_Q_list = torch.stack([torch.stack(std_Q_list1).squeeze(),torch.stack(std_Q_list2).squeeze()], axis=-1)

        return std_Q_list


    def get_weights(self, variance, eps, feedback_type):
        return torch.ones(variance.size()).cuda() / variance.size()[0]


    def get_critic_loss(self, en_index, obs, next_obs, actions, rewards, terminals, mask, std_Q_critic_list, alpha):
        """
        QF Loss
        """
        q1_pred = self.qf1[en_index](obs, actions)
        q2_pred = self.qf2[en_index](obs, actions)
        
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy[en_index](
            next_obs, reparameterize=True, return_log_prob=True,
        )
        q_target_both = torch.stack([self.target_qf1[en_index](next_obs, new_next_actions),\
                                self.target_qf2[en_index](next_obs, new_next_actions)], axis=-1)
        target_q, target_q_argmin = q_target_both.min(-1)
        target_q_values = target_q - alpha * new_log_pi

        std_Q_critic = std_Q_critic_list[en_index].gather(1, target_q_argmin)[mask]

        xi_critic = get_optimal_eps((std_Q_critic**2).detach().cpu().numpy(),self.minimal_eff_bs, 0) if self.dynamic_xi else self.xi #* torch.median(std_Q_critic**2).item()
        weight_target_Q = self.get_weights(std_Q_critic, xi_critic, self.feedback_type) #torch.ones(std_Q_critic_list[en_index].size()).cuda() / std_Q_critic_list[en_index].size()[0] #torch.sigmoid(-std_Q_critic_list[en_index]*self.temperature) + 0.5
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        # print(q1_pred.size(), q_target.size(), weight_target_Q.size())
        qf1_loss = self.qf_criterion(q1_pred[mask], q_target[mask].detach())  * (weight_target_Q.detach())
        qf2_loss = self.qf_criterion(q2_pred[mask], q_target[mask].detach()) * (weight_target_Q.detach())
        qf1_loss = qf1_loss.sum()
        qf2_loss = qf2_loss.sum()

        return q1_pred, q2_pred, q_target, weight_target_Q, xi_critic, qf1_loss, qf2_loss
        
        
    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        masks = batch['masks']
        
        # variables for logging
        tot_qf1_loss, tot_qf2_loss, tot_q1_pred, tot_q2_pred, tot_q_target = 0, 0, 0, 0, 0
        tot_log_pi, tot_policy_mean, tot_policy_log_std, tot_policy_loss = 0, 0, 0, 0
        tot_alpha, tot_alpha_loss = 0, 0
        tot_variance_actor, tot_weights_actor = [], []
        tot_variance_critic, tot_weights_critic = [], []
        tot_xi_critic, tot_ebs_critic = 0, 0
        
        tot_xi_actor, tot_ebs_actor = 0, 0

        std_Q_actor_list = self.get_variance(obs=obs, update_type=0)#.unsqueeze(1)
        std_Q_critic_list = self.get_variance(obs=next_obs, update_type=1)#.unsqueeze(1)
        
        for en_index in range(self.num_ensemble):
            mask = masks[:,en_index].reshape(-1, 1).bool()

            """
            Policy and Alpha Loss
            """
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy[en_index](
                obs, reparameterize=True, return_log_prob=True,
            )
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha[en_index] * (log_pi + self.target_entropy).detach()) * mask
                alpha_loss = alpha_loss.sum() / (mask.sum() + 1)
                self.alpha_optimizer[en_index].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer[en_index].step()
                alpha = self.log_alpha[en_index].exp()
            else:
                alpha_loss = 0
                alpha = 1

            qf_next = torch.stack([self.qf1[en_index](obs, new_obs_actions),\
                self.qf2[en_index](obs, new_obs_actions)], axis=-1)
            q_new_actions, q_argmin = qf_next.min(-1)
            #print(q_new_actions.size(), q_argmin.size(), qf_next.size())
            #print(std_Q_actor_list[en_index].size(), q_argmin.size(), mask.size())
            std_Q = std_Q_actor_list[en_index].gather(1, q_argmin)[mask]      
            xi_actor = get_optimal_xi((std_Q**2).detach().cpu().numpy(),self.minimal_eff_bs, 0) if self.dynamic_xi else self.xi #* torch.median(std_Q**2).item()
            weight_actor_Q = self.get_weights(std_Q, xi_actor, self.feedback_type) #2*torch.sigmoid(-std_Q* self.temperature_act)
            policy_loss = (alpha*log_pi[mask] - q_new_actions[mask] - self.expl_gamma * std_Q) * weight_actor_Q.detach()
            policy_loss = policy_loss.sum()

            q1_pred, q2_pred, q_target, weight_target_Q, eps_critic, qf1_loss, qf2_loss = self.get_critic_loss(en_index, obs, next_obs, actions, rewards, terminals, mask, std_Q_critic_list, alpha)

            """
            Update networks
            """
            self.qf1_optimizer[en_index].zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer[en_index].step()

            self.qf2_optimizer[en_index].zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer[en_index].step()

            self.policy_optimizer[en_index].zero_grad()
            policy_loss.backward()
            self.policy_optimizer[en_index].step()

            """
            Soft Updates
            """
            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    self.qf1[en_index], self.target_qf1[en_index], self.soft_target_tau
                )
                ptu.soft_update_from_to(
                    self.qf2[en_index], self.target_qf2[en_index], self.soft_target_tau
                )
                
            """
            Statistics for log
            """
            tot_qf1_loss += qf1_loss * (1/self.num_ensemble)
            tot_qf2_loss += qf2_loss * (1/self.num_ensemble)
            tot_q1_pred += q1_pred * (1/self.num_ensemble)
            tot_q2_pred += q2_pred * (1/self.num_ensemble)
            tot_q_target += q_target * (1/self.num_ensemble)
            tot_log_pi += log_pi * (1/self.num_ensemble)
            tot_policy_mean += policy_mean * (1/self.num_ensemble)
            tot_policy_log_std += policy_log_std * (1/self.num_ensemble)
            tot_alpha += alpha.item() * (1/self.num_ensemble)
            tot_alpha_loss += alpha_loss.item()
            tot_policy_loss = (log_pi - q_new_actions).mean() * (1/self.num_ensemble)
            tot_variance_actor.extend(std_Q_actor_list)
            tot_variance_critic.extend(std_Q_critic_list)
            tot_weights_actor.extend(weight_actor_Q)
            tot_weights_critic.extend(weight_target_Q)
            tot_ebs_actor += compute_eff_bs(weight_actor_Q.detach().cpu().numpy()) * (1/self.num_ensemble)            
            tot_ebs_critic += compute_eff_bs(weight_target_Q.detach().cpu().numpy()) * (1/self.num_ensemble)
            # tot_xi_actor += xi_actor * (1/self.num_ensemble)
            # tot_xi_critic += xi_critic * (1/self.num_ensemble)
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(tot_qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(tot_qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                tot_policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(tot_q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(tot_q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(tot_q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(tot_log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(tot_policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(tot_policy_log_std),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Mean Variance (Actor)',
                np.mean((torch.stack(tot_variance_actor)**2).cpu().numpy()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Median Variance (Actor)',
                np.median(torch.stack(tot_variance_actor).cpu().numpy()),           
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Mean Variance (Critic)',
                np.mean((torch.stack(tot_variance_critic)**2).cpu().numpy()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Median Variance (Critic)',
                np.median((torch.stack(tot_variance_critic)**2).cpu().numpy()),   
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Mean Weights (Actor)',
                np.mean(torch.stack(tot_weights_actor).cpu().numpy()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Median Weights (Actor)',
                np.median(torch.stack(tot_weights_actor).cpu().numpy()),           
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Mean Weights (Critic)',
                np.mean(torch.stack(tot_weights_critic).cpu().numpy()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Median Weights (Critic)',
                np.median(torch.stack(tot_weights_critic).cpu().numpy()),   
            ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Avg EPS (Actor)',
            #     tot_xi_actor,
            # ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'Avg EPS (Critic)',
            #     tot_xi_critic,
            # ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Avg EBS (Actor)',
                tot_ebs_actor,
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Avg EBS (Critic)',
                tot_ebs_critic,
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = tot_alpha
                self.eval_statistics['Alpha Loss'] = tot_alpha_loss
                
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
    
    @property
    def networks(self):
        output = []
        for en_index in range(self.num_ensemble):
            output.append(self.policy[en_index])
            output.append(self.qf1[en_index])
            output.append(self.qf2[en_index])
            output.append(self.target_qf1[en_index])
            output.append(self.target_qf2[en_index])
        return output

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )
    
    def save_models(self, step):
        for en_index in range(self.num_ensemble):
            torch.save(
                self.policy[en_index].state_dict(), '%s/%d_th_actor_%s.pt' % (self.model_dir, en_index, step)
            )
            torch.save(
                self.qf1[en_index].state_dict(), '%s/%d_th_1st_critic_%s.pt' % (self.model_dir, en_index, step)
            )
            torch.save(
                self.qf2[en_index].state_dict(), '%s/%d_th_2nd_critic_%s.pt' % (self.model_dir, en_index, step)
            )
            torch.save(
                self.target_qf1[en_index].state_dict(), '%s/%d_th_1st_target_critic_%s.pt' % (self.model_dir, en_index, step)
            )
            torch.save(
                self.target_qf2[en_index].state_dict(), '%s/%d_th_2nd_target_critic_%s.pt' % (self.model_dir, en_index, step)
            )
            
    def load_models(self, step):
        for en_index in range(self.num_ensemble):
            self.policy[en_index].load_state_dict(
                torch.load('%s/%d_th_actor_%s.pt' % (self.model_dir, en_index, step))
            )
            self.qf1[en_index].load_state_dict(
                torch.load('%s/%d_th_1st_critic_%s.pt' % (self.model_dir, en_index, step))
            )
            self.qf2[en_index].load_state_dict(
                torch.load('%s/%d_th_2nd_critic_%s.pt' % (self.model_dir, en_index, step))
            )
            self.target_qf1[en_index].load_state_dict(
                torch.load('%s/%d_th_1st_target_critic_%s.pt' % (self.model_dir, en_index, step))
            )
            self.target_qf2[en_index].load_state_dict(
                torch.load('%s/%d_th_2nd_target_critic_%s.pt' % (self.model_dir, en_index, step))
            )
            
            
    def print_model(self):
        for name, param in self.policy[0].named_parameters():
            if param.requires_grad:
                print(name)
                print(param.data)
                break;
        
 


class VarEnsembleSAC(EnsembleSAC):
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

    def iv_weights(self, variance, eps):
        weights = (1. / (variance+eps))
        weights /= weights.sum(0)
        return weights

    def get_weights(self, std, eps, feedback_type):
        if feedback_type == 0 or feedback_type == 1:
            weight_target_Q = self.iv_weights(std**2, eps)
        else:
            weight_target_Q = self.iv_weights(std**2, eps)

        return weight_target_Q

    def get_variance(self, obs, update_type):
        std_Q_list1, std_Q_list2 = [], []

        for en_index1 in range(self.num_ensemble):
            with torch.no_grad():
                policy_action, _, _, _, *_ = self.policy[en_index1](
                    obs, reparameterize=True, return_log_prob=True,
                )
            L_target_Q1, L_target_Q2 = [], []
            L_target_var_Q1, L_target_var_Q2 = [], []
            for en_index2 in range(self.num_ensemble):
                if update_type == 0: # actor
                    target_Q1, target_logvar_Q1 = self.qf1[en_index2](obs, policy_action, return_logstd=True)
                    target_Q2, target_logvar_Q2 = self.qf2[en_index2](obs, policy_action, return_logstd=True)
                else: # critic
                    target_Q1, target_logvar_Q1 = self.target_qf1[en_index2](obs, policy_action, return_logstd=True)
                    target_Q2, target_logvar_Q2 = self.target_qf2[en_index2](obs, policy_action, return_logstd=True)
                L_target_Q1.append(target_Q1)
                L_target_Q2.append(target_Q2)
                L_target_var_Q1.append(torch.exp(target_logvar_Q1))
                L_target_var_Q2.append(torch.exp(target_logvar_Q2))

            L_target_Q1, L_target_Q2 = torch.stack(L_target_Q1), torch.stack(L_target_Q2)
            L_target_var_Q1, L_target_var_Q2 = torch.stack(L_target_var_Q1), torch.stack(L_target_var_Q2)
#            print(L_target_Q1.size(), L_target_var_Q1.size())
            var_mixture_Q1 = (L_target_var_Q1 + L_target_Q1**2 - L_target_Q1.mean(0).repeat(self.num_ensemble,1,1)**2).mean(0)
            var_mixture_Q2 = (L_target_var_Q2 + L_target_Q2**2 - L_target_Q2.mean(0).repeat(self.num_ensemble,1,1)**2).mean(0)

            std_Q_list1.append(torch.sqrt(var_mixture_Q1).detach())
            std_Q_list2.append(torch.sqrt(var_mixture_Q2).detach())

        std_Q_list = torch.stack([torch.stack(std_Q_list1).squeeze(),torch.stack(std_Q_list2).squeeze()], axis=-1)

        return std_Q_list

    def get_critic_loss(self, en_index, obs, next_obs, actions, rewards, terminals, mask, std_Q_critic_list, alpha):
        """
        QF Loss
        """
        q1_pred, q1_pred_logvar = self.qf1[en_index](obs, actions, return_logstd=True)
        q2_pred, q2_pred_logvar = self.qf2[en_index](obs, actions, return_logstd=True)
        
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy[en_index](
            next_obs, reparameterize=True, return_log_prob=True,
        )

        target_qf1, target_qf2 = self.target_qf1[en_index](next_obs, new_next_actions), self.target_qf2[en_index](next_obs, new_next_actions) 
        q_target_both = torch.stack([target_qf1, target_qf2], axis=-1)
        target_q, target_q_argmin = q_target_both.min(-1)
        target_q_values = target_q - alpha * new_log_pi

#        print(std_Q_critic_list[en_index].size(), target_q_argmin.size(), mask.size())
        std_Q_critic = std_Q_critic_list[en_index].gather(1, target_q_argmin)[mask]

        xi_critic = get_optimal_eps((std_Q_critic**2).detach().cpu().numpy(),self.minimal_eff_bs, 0) if self.dynamic_xi else self.xi #* torch.median(std_Q_critic**2).item()
        weight_target_Q = self.get_weights(std_Q_critic, xi_critic, self.feedback_type) #torch.ones(std_Q_critic_list[en_index].size()).cuda() / std_Q_critic_list[en_index].size()[0] #torch.sigmoid(-std_Q_critic_list[en_index]*self.temperature) + 0.5
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q1_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_qf1
        q2_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_qf2
        # print(q1_pred.size(), q_target.size(), weight_target_Q.size())
        qf1_loss = self.qf_criterion(q1_pred[mask], q_target[mask].detach())  * (weight_target_Q.detach())
        qf2_loss = self.qf_criterion(q2_pred[mask], q_target[mask].detach()) * (weight_target_Q.detach())

        lossatt_q1 = (torch.mean((q1_target.detach() - q1_pred)**2 / (2 * torch.exp(q1_pred_logvar)) + (1/2) * torch.log(torch.exp(q1_pred_logvar))))
        lossatt_q2 = (torch.mean((q2_target.detach() - q2_pred)**2 / (2 * torch.exp(q2_pred_logvar)) + (1/2) * torch.log(torch.exp(q2_pred_logvar))))
        qf1_loss = qf1_loss.sum() + self.args.loss_att_weight * lossatt_q1
        qf2_loss = qf2_loss.sum() + self.args.loss_att_weight * lossatt_q2

        return q1_pred, q2_pred, q_target, weight_target_Q, xi_critic, qf1_loss, qf2_loss
        
        
