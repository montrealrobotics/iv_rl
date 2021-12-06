from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from scipy.optimize import minimize

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

def compute_eff_bs(weights):
    # Compute original effective mini-batch size
    eff_bs = 1/np.sum([weight**2 for weight in weights])
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
            eps=None,
            dynamic_eps=False,
            minimal_eff_bs=None,
    ):
        super().__init__()
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
        self.eps = eps 
        self.dynamic_eps = dynamic_eps
        self.minimal_eff_bs = minimal_eff_bs

    def corrective_feedback(self, obs, update_type):
        std_Q_list = []
        
        if self.feedback_type == 0 or self.feedback_type == 2:
            for en_index in range(self.num_ensemble):
                with torch.no_grad():
                    policy_action, _, _, _, *_ = self.policy[en_index](
                        obs, reparameterize=True, return_log_prob=True,
                    )
                    if update_type == 0:
                        actor_Q1 = self.qf1[en_index](obs, policy_action)
                        actor_Q2 = self.qf2[en_index](obs, policy_action)
                    else:
                        actor_Q1 = self.target_qf1[en_index](obs, policy_action)
                        actor_Q2 = self.target_qf2[en_index](obs, policy_action)
                    mean_actor_Q = 0.5*(actor_Q1 + actor_Q2)
                    var_Q = 0.5*((actor_Q1 - mean_actor_Q)**2 + (actor_Q2 - mean_actor_Q)**2)
                std_Q_list.append(torch.sqrt(var_Q).detach())
                
        elif self.feedback_type == 1 or self.feedback_type == 3:
            mean_Q, var_Q = None, None
            L_target_Q = []
            for en_index in range(self.num_ensemble):
                with torch.no_grad():
                    policy_action, _, _, _, *_ = self.policy[en_index](
                        obs, reparameterize=True, return_log_prob=True,
                    )
                    
                    if update_type == 0: # actor
                        target_Q1 = self.qf1[en_index](obs, policy_action)
                        target_Q2 = self.qf2[en_index](obs, policy_action)
                    else: # critic
                        target_Q1 = self.target_qf1[en_index](obs, policy_action)
                        target_Q2 = self.target_qf2[en_index](obs, policy_action)
                    L_target_Q.append(target_Q1)
                    L_target_Q.append(target_Q2)
                    if en_index == 0:
                        mean_Q = 0.5*(target_Q1 + target_Q2) / self.num_ensemble
                    else:
                        mean_Q += 0.5*(target_Q1 + target_Q2) / self.num_ensemble

            temp_count = 0
            for target_Q in L_target_Q:
                if temp_count == 0:
                    var_Q = (target_Q.detach() - mean_Q)**2
                else:
                    var_Q += (target_Q.detach() - mean_Q)**2
                temp_count += 1
            var_Q = var_Q / temp_count
            std_Q_list.append(torch.sqrt(var_Q).detach())

        return std_Q_list
        
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
        tot_eps_critic, tot_ebs_critic = 0, 0
        tot_eps_actor, tot_ebs_actor = 0, 0

        std_Q_actor_list = self.corrective_feedback(obs=obs, update_type=0)
        std_Q_critic_list = self.corrective_feedback(obs=next_obs, update_type=1)
        
        for en_index in range(self.num_ensemble):
            mask = masks[:,en_index].reshape(-1, 1)

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

            q_new_actions = torch.min(
                self.qf1[en_index](obs, new_obs_actions),
                self.qf2[en_index](obs, new_obs_actions),
            )
            
            if self.feedback_type == 0 or self.feedback_type == 2:
                std_Q = std_Q_actor_list[en_index]
            else:
                std_Q = std_Q_actor_list[0]
                
            eps_actor = get_optimal_eps(std_Q.detach().cpu().numpy(),self.minimal_eff_bs, self.eps) if self.dynamic_eps else self.eps
            if self.feedback_type == 1 or self.feedback_type == 0:
                weight_actor_Q = torch.ones(std_Q.size()).cuda() / std_Q.size()[0] #torch.sigmoid(-std_Q*self.temperature_act) + 0.5
            else:
                weight_actor_Q = torch.ones(std_Q.size()).cuda() / std_Q.size()[0] #2*torch.sigmoid(-std_Q* self.temperature_act)
            policy_loss = (alpha*log_pi - q_new_actions - self.expl_gamma * std_Q) * mask * weight_actor_Q.detach()
            policy_loss = policy_loss.sum() / (mask.sum() + 1)

            """
            QF Loss
            """
            q1_pred = self.qf1[en_index](obs, actions)
            q2_pred = self.qf2[en_index](obs, actions)
            
            # Make sure policy accounts for squashing functions like tanh correctly!
            new_next_actions, _, _, new_log_pi, *_ = self.policy[en_index](
                next_obs, reparameterize=True, return_log_prob=True,
            )
            target_q_values = torch.min(
                self.target_qf1[en_index](next_obs, new_next_actions),
                self.target_qf2[en_index](next_obs, new_next_actions),
            ) - alpha * new_log_pi


            eps_critic = get_optimal_eps(std_Q_critic_list[en_index].detach().cpu().numpy(),self.minimal_eff_bs, self.eps) if self.dynamic_eps else self.eps

            if self.feedback_type == 0 or self.feedback_type == 2:
                if self.feedback_type == 0:
                    weight_target_Q = torch.ones(std_Q_critic_list[en_index].size()).cuda() / std_Q_critic_list[en_index].size()[0] #torch.sigmoid(-std_Q_critic_list[en_index]*self.temperature) + 0.5
                else:
                    weight_target_Q = torch.ones(std_Q_critic_list[en_index].size()).cuda() / std_Q_critic_list[en_index].size()[0] #2*torch.sigmoid(-std_Q_critic_list[en_index]*self.temperature)
            else:
                if self.feedback_type == 1:
                    weight_target_Q = torch.ones(std_Q_critic_list[0].size()).cuda() / std_Q_critic_list[0].size()[0] # torch.sigmoid(-std_Q_critic_list[0]*self.temperature) + 0.5
                else:
                    weight_target_Q = torch.ones(std_Q_critic_list[0].size()).cuda() / std_Q_critic_list[0].size()[0] #2*torch.sigmoid(-std_Q_critic_list[0]*self.temperature)
            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach()) * mask * (weight_target_Q.detach())
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach()) * mask * (weight_target_Q.detach())
            qf1_loss = qf1_loss.sum() / (mask.sum() + 1)
            qf2_loss = qf2_loss.sum() / (mask.sum() + 1)
            
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
            tot_eps_actor += eps_actor * (1/self.num_ensemble)
            tot_eps_critic += eps_critic * (1/self.num_ensemble)
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
                np.mean(torch.stack(tot_variance_actor).cpu().numpy()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Median Variance (Actor)',
                np.median(torch.stack(tot_variance_actor).cpu().numpy()),           
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Mean Variance (Critic)',
                np.mean(torch.stack(tot_variance_critic).cpu().numpy()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Median Variance (Critic)',
                np.median(torch.stack(tot_variance_critic).cpu().numpy()),   
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
            self.eval_statistics.update(create_stats_ordered_dict(
                'Avg EPS (Actor)',
                tot_eps_actor,
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Avg EPS (Critic)',
                tot_eps_critic,
            ))
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
