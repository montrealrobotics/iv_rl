import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

import os
import wandb
import random
import numpy as np
from collections import namedtuple, deque, Counter

from utils import * 
from .networks import *
import src.utils as utils
from src.models.risk_models import *
from src.datasets.risk_datasets import *


class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, env, opt, device="cuda"):
        """Initialize an Agent object.
        
        Params
        ======
            env (gym object): Initialized gym environment
            opt (dict): command line options for the model
            device (str): cpu or gpu
        """
        self.env = env
        self.opt = opt
        self.state_size = np.prod(env.observation_space.shape)
        self.action_size = env.action_space.n
        self.seed = random.seed(opt.env_seed)
        self.test_scores = []
        self.device = device
        self.mask = False

        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, opt.net_seed).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, opt.net_seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=opt.lr)

        # Replay memory
        self.memory = ReplayBuffer(opt, self.action_size, 42, self.device, self.mask)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.xi = 0
        self.loss = 0 

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if self.mask:
	        mask = self.random_state.binomial(1, self.opt.mask_prob, self.opt.num_nets)
	        self.memory.add(state, action, reward, next_state, done, mask)
        else:
            self.memory.add(state, action, reward, next_state, done)
	        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.opt.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.opt.batch_size:
                experiences = self.memory.sample()
                return self.learn(experiences, self.opt.gamma)
            else:
            	return None

    def act(self, state, eps=0., is_train=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        action_values = action_values.cpu().data.numpy()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values), np.mean(action_values)
        else:
            return random.choice(np.arange(self.action_size)), np.mean(action_values)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        weights = torch.ones(Q_expected.size()).to(self.device) / self.opt.batch_size
        loss = self.weighted_mse(Q_expected, Q_targets, weights)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # In order to log the loss value
        self.loss = loss.item()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def save(self, scores):
        torch.save(self.qnetwork_local.state_dict(), os.path.join(self.opt.log_dir, "%s_%s_seed_%d_net_seed_%d.pth"%(self.opt.env, self.opt.model, self.opt.env_seed, self.opt.net_seed)))
        np.save(os.path.join(self.opt.log_dir,"logs_%s_%s_%s_seed_%d_net_seed_%d.npy"%(self.opt.exp, self.opt.env, self.opt.model, self.opt.env_seed, self.opt.net_seed)), scores)
        wandb.save(os.path.join(self.opt.log_dir,"logs_%s_%s_%s_seed_%d_net_seed_%d.npy"%(self.opt.exp, self.opt.env, self.opt.model, self.opt.env_seed, self.opt.net_seed)), base_path=self.opt.log_dir)


    def weighted_mse(self, inputs, targets, weights, mask = None):
        loss = weights*((targets - inputs)**2)
        if mask is not None:
        	loss *= mask
        #print(loss.size())
        return loss.sum(0)


    def train(self, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.
        
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        flag = 1 # used for hyperparameter tuning
        scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            state = self.env.reset()[0].ravel()
            score, ep_var, ep_weights, eff_bs_list, xi_list, ep_Q, ep_loss = 0, [], [], [], [], [], []   # list containing scores from each episode
            for t in range(max_t):
                action, Q = self.act(state.ravel(), eps, is_train=True)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = next_state.ravel()
                logs = self.step(state, action, reward, next_state, done)
                state = next_state
                if done or truncated:
                    reward += self.opt.end_reward
                score += reward
                if logs is not None:
                    # try:
                    ep_var.extend(logs[0])
                    ep_weights.extend(logs[1])
                    eff_bs_list.append(logs[2])
                    xi_list.append(logs[3])
                    # except:
                    #     pass
                ep_Q.append(Q)
                ep_loss.append(self.loss)
                if done or truncated:
                    break 

            #wandb.log({"V(s) (VAR)": np.var(ep_Q), "V(s) (Mean)": np.mean(ep_Q),
            #    "V(s) (Min)": np.min(ep_Q), "V(s) (Max)": np.max(ep_Q), 
            #    "V(s) (Median)": np.median(ep_Q)}, commit=False)
            #wandb.log({"Loss (VAR)": np.var(ep_loss), "Loss (Mean)": np.mean(ep_loss),
            #    "Loss (Min)": np.min(ep_loss), "Loss (Max)": np.max(ep_loss), 
            #    "Loss (Median)": np.median(ep_loss)}, commit=False)
            #if len(ep_var) > 0: # if there are entries in the variance list
	    #        self.train_log(ep_var, ep_weights, eff_bs_list, eps_list)
            if i_episode % self.opt.test_every == 0:
                self.test(episode=i_episode)
 
            scores_window.append(score)        # save most recent score
            scores.append(score)               # save most recent score
            eps = max(eps_end, eps_decay*eps)  # decrease epsilon
            wandb.log({"Moving Average Return/100episode": np.mean(scores_window)})
            if np.mean(self.test_scores[-100:]) >= self.opt.goal_score and flag:
               flag = 0 
               wandb.log({"EpisodeSolved": i_episode}, commit=False)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #self.save(scores)

    def test(self, episode, num_trials=5, max_t=1000):
        score_list, variance_list = [], []
        #for i in range(num_trials):
        state = self.env.reset()[0]
        score = 0
        for t in range(max_t):
            action, _ = self.act(state.ravel(), -1)
            next_state, reward, done, truncated, info = self.env.step(action)
            state = next_state.ravel()
            score += reward
            if done or truncated:
                break
        self.test_scores.append(score)
        wandb.log({"Test Environment (Moving Average Return/100 episodes)": np.mean(self.test_scores[-100:]),
                  "Test Environment Return": score}, step=episode)
        return np.mean(score_list), np.var(score_list)



class RiskDQN(DQNAgent):
    def __init__(self, env, opt, device="cuda"):
        super().__init__(env, opt, device)


        risk_model_class = {"bayesian": {"continuous": BayesRiskEstCont, "binary": BayesRiskEst, "quantile": BayesRiskEst}, 
                            "mlp": {"continuous": RiskEst, "binary": RiskEst}} 

        risk_size_dict = {"continuous": 1, "binary": 2, "quantile": opt.quantile_num}
        self.risk_size = risk_size_dict[opt.risk_type]

        # Q-Network
        self.qnetwork_local = QRiskNetwork(self.state_size, self.action_size, self.risk_size, opt.net_seed).to(self.device)
        self.qnetwork_target = QRiskNetwork(self.state_size, self.action_size, self.risk_size, opt.net_seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=opt.lr)

        # Replay memory
        self.memory = ReplayBuffer(opt, self.action_size, 42, self.device)
        if opt.fine_tune_risk:
            if opt.risk_type == "quantile":
                weight_tensor = torch.Tensor([1]*opt.quantile_num).to(device)
                weight_tensor[0] = opt.weight
            elif opt.risk_type == "binary":
                weight_tensor = torch.Tensor([1., opt.weight]).to(device)
            if opt.model_type == "bayesian":
                self.risk_criterion = nn.NLLLoss(weight=weight_tensor)
            else:
                self.risk_criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

        self.risk_rb = utils.ReplayBuffer(buffer_size=10000)

        print("using risk")
        self.risk_model = risk_model_class[opt.model_type][opt.risk_type](obs_size=self.state_size,fc1_size=32, fc2_size=32,\
                                                                          fc3_size=32, fc4_size=32, batch_norm=True, out_size=self.risk_size) 
        if os.path.exists(opt.risk_model_path):
            self.risk_model.load_state_dict(torch.load(opt.risk_model_path, map_location=device))
            print("Pretrained risk model loaded successfully")
        self.risk_model.to(device)
        if opt.fine_tune_risk:
            # print("Fine Tuning risk")
            ## Freezing all except last layer of the risk model
            if opt.freeze_risk_layers:
                for param in self.risk_model.parameters():
                    param.requires_grad = False 
                self.risk_model.out.weight.requires_grad = True
                self.risk_model.out.bias.requires_grad = True 
            self.opt_risk = optim.Adam(filter(lambda p: p.requires_grad, self.risk_model.parameters()), lr=opt.risk_lr, eps=1e-10)
            self.risk_model.eval()
        else:
            print("No model in the path specified!!")

    def act(self, state, eps=0., is_train=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            risk = self.risk_model(state)
            action_values = self.qnetwork_local(state, risk)
        self.qnetwork_local.train()
        action_values = action_values.cpu().data.numpy()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values), np.mean(action_values)
        else:
            return random.choice(np.arange(self.action_size)), np.mean(action_values)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            risks = self.risk_model(states)
            next_risks = self.risk_model(next_states)
            Q_targets_next = self.qnetwork_target(next_states, next_risks).max(1)[0].unsqueeze(1)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states, risks).gather(1, actions)

        # Compute loss
        weights = torch.ones(Q_expected.size()).to(self.device) / self.opt.batch_size
        loss = self.weighted_mse(Q_expected, Q_targets, weights)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # In order to log the loss value
        self.loss = loss.item()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def save(self, scores):
        torch.save(self.qnetwork_local.state_dict(), os.path.join(self.opt.log_dir, "%s_%s_seed_%d_net_seed_%d.pth"%(self.opt.env, self.opt.model, self.opt.env_seed, self.opt.net_seed)))
        np.save(os.path.join(self.opt.log_dir,"logs_%s_%s_%s_seed_%d_net_seed_%d.npy"%(self.opt.exp, self.opt.env, self.opt.model, self.opt.env_seed, self.opt.net_seed)), scores)
        wandb.save(os.path.join(self.opt.log_dir,"logs_%s_%s_%s_seed_%d_net_seed_%d.npy"%(self.opt.exp, self.opt.env, self.opt.model, self.opt.env_seed, self.opt.net_seed)), base_path=self.opt.log_dir)


    def weighted_mse(self, inputs, targets, weights, mask = None):
        loss = weights*((targets - inputs)**2)
        if mask is not None:
        	loss *= mask
        #print(loss.size())
        return loss.sum(0)

    def train_risk(self, cfg, model, data, criterion, opt, device):
        model.train()
        dataset = RiskyDataset(data["next_obs"].to('cpu'), data["actions"].to('cpu'), data["risks"].to('cpu'), False, risk_type=cfg.risk_type,
                                fear_clip=None, fear_radius=cfg.fear_radius, one_hot=True, quantile_size=cfg.quantile_size, quantile_num=cfg.quantile_num)
        dataloader = DataLoader(dataset, batch_size=cfg.risk_batch_size, shuffle=True, num_workers=10, generator=torch.Generator(device='cpu'))
        net_loss = 0
        for batch in dataloader:
            pred = model(batch[0].to(device))
            if cfg.model_type == "mlp":
                loss = criterion(pred, batch[1].squeeze().to(device))
            else:
                loss = criterion(pred, torch.argmax(batch[1].squeeze(), axis=1).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

            net_loss += loss.item()
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "risk_model.pt"))
        wandb.save("risk_model.pt")
        model.eval()
        return net_loss


    def train(self, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.
        
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        flag = 1 # used for hyperparameter tuning
        scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        global_step = 0
        for i_episode in range(1, n_episodes+1):
            f_next_state, f_actions = None, None
            state = self.env.reset()[0].ravel()
            score, ep_var, ep_weights, eff_bs_list, xi_list, ep_Q, ep_loss = 0, [], [], [], [], [], []   # list containing scores from each episode
            for t in range(max_t):
                global_step += 1
                action, Q = self.act(state.ravel(), eps, is_train=True)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = next_state.ravel()
                f_next_state = torch.Tensor(next_state).unsqueeze(0) if f_next_state is None else torch.cat([f_next_state, torch.Tensor(next_state).unsqueeze(0)], axis=0)
                f_actions = torch.Tensor([action]).unsqueeze(0) if f_actions is None else torch.cat([f_actions, torch.Tensor([action]).unsqueeze(0)], axis=0)
                logs = self.step(state, action, reward, next_state, done)
                state = next_state
                if done or truncated:
                    reward += self.opt.end_reward
                score += reward
                if logs is not None:
                    # try:
                    ep_var.extend(logs[0])
                    ep_weights.extend(logs[1])
                    eff_bs_list.append(logs[2])
                    xi_list.append(logs[3])
                    # except:
                    #     pass
                ep_Q.append(Q)
                ep_loss.append(self.loss)
                if done or truncated:
                    f_risks = torch.Tensor(list(reversed(range(t+1)))) if done else torch.Tensor([max_t]*(t+1))
                    self.risk_rb.add(f_next_state, f_next_state, f_actions, f_actions, f_actions, f_actions, f_risks, f_actions)
                    break 
            if i_episode % self.opt.risk_update_period == 0 and global_step > self.opt.risk_batch_size*self.opt.num_update_risk:
                    data = self.risk_rb.sample(self.opt.risk_batch_size*self.opt.num_update_risk)
                    risk_loss = self.train_risk(self.opt, self.risk_model, data, self.risk_criterion, self.opt_risk, self.device)              

            #wandb.log({"V(s) (VAR)": np.var(ep_Q), "V(s) (Mean)": np.mean(ep_Q),
            #    "V(s) (Min)": np.min(ep_Q), "V(s) (Max)": np.max(ep_Q), 
            #    "V(s) (Median)": np.median(ep_Q)}, commit=False)
            #wandb.log({"Loss (VAR)": np.var(ep_loss), "Loss (Mean)": np.mean(ep_loss),
            #    "Loss (Min)": np.min(ep_loss), "Loss (Max)": np.max(ep_loss), 
            #    "Loss (Median)": np.median(ep_loss)}, commit=False)
            #if len(ep_var) > 0: # if there are entries in the variance list
	    #        self.train_log(ep_var, ep_weights, eff_bs_list, eps_list)
            if i_episode % self.opt.test_every == 0:
                self.test(episode=i_episode)
 
            scores_window.append(score)        # save most recent score
            scores.append(score)               # save most recent score
            eps = max(eps_end, eps_decay*eps)  # decrease epsilon
            wandb.log({"Moving Average Return/100episode": np.mean(scores_window)})
            if np.mean(self.test_scores[-100:]) >= self.opt.goal_score and flag:
               flag = 0 
               wandb.log({"EpisodeSolved": i_episode}, commit=False)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #self.save(scores)

    def test(self, episode, num_trials=5, max_t=1000):
        score_list, variance_list = [], []
        #for i in range(num_trials):
        state = self.env.reset()[0]
        score = 0
        for t in range(max_t):
            action, _ = self.act(state.ravel(), -1)
            next_state, reward, done, truncated, info = self.env.step(action)
            state = next_state.ravel()
            score += reward
            if done or truncated:
                break
        self.test_scores.append(score)
        wandb.log({"Test Environment (Moving Average Return/100 episodes)": np.mean(self.test_scores[-100:]),
                  "Test Environment Return": score}, step=episode)
        return np.mean(score_list), np.var(score_list)

class LossAttDQN(DQNAgent):

    def __init__(self, env, opt, device="cuda"):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_nets (int): number of Q-networks
            seed (int): random seed
        """
        super().__init__(env, opt, device)

        # Q-Network
        self.qnetwork_local = TwoHeadQNetwork(self.state_size, self.action_size, opt.net_seed).to(self.device)
        self.qnetwork_target = TwoHeadQNetwork(self.state_size, self.action_size, opt.net_seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=opt.lr)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next_all, Q_target_next_var_all = self.qnetwork_target(next_states, True)
        Q_targets_next, next_actions, Q_targets_next_var = Q_targets_next_all.max(1)[0].unsqueeze(1), Q_targets_next_all.max(1)[1].unsqueeze(1),\
        																					Q_target_next_var_all.detach()

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get variance for the next actions
        Q_targets_var = torch.exp(Q_targets_next_var.gather(1, next_actions))

        # Get expected Q values from local model
        Q_expected, Q_log_var  = [x.gather(1, actions) for x in self.qnetwork_local(states, True)] 

        # Compute loss
        self.xi = get_optimal_xi(Q_targets_var.detach().cpu().numpy(), self.opt.minimal_eff_bs, self.xi) if self.opt.dynamic_xi else self.opt.xi
        weights = self.get_mse_weights(Q_targets_var)
        loss = self.weighted_mse(Q_expected, Q_targets, weights)

        # Compute Loss Attenuation 
        y, mu, var = Q_targets, Q_expected, torch.exp(Q_log_var)
        std = torch.sqrt(var) 
        # print(y.size(), mu.size(), std.size())
        lossatt = torch.mean((y - mu)**2 / (2 * torch.square(std)) + (1/2) * torch.log(torch.square(std)))

        net_loss = loss + self.opt.loss_att_weight*lossatt

        # Minimize the loss
        self.optimizer.zero_grad()
        net_loss.backward()
        self.optimizer.step()

        # In order to log the loss value
        self.loss = loss.item()

        eff_batch_size = compute_eff_bs(weights.detach().cpu().numpy())

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.tau)                     

        return torch.exp(Q_log_var).detach().cpu().numpy(), weights.detach().cpu().numpy(), eff_batch_size, self.xi

    def get_mse_weights(self, variance):
    	weights = torch.ones(variance.size()).to(self.device) / self.opt.batch_size
    	return weights

    def train_log(self, var, weights, eff_batch_size, eps_list):
        wandb.log({"Variance(Q) (VAR)": np.var(var), "Variance(Q) (Mean)": np.mean(var),\
        "Variance(Q) (Min)": np.min(var), "Variance(Q) (Max)": np.max(var), "Variance(Q) (Median)": np.median(var)}, commit=False)
        wandb.log({"Variance(Q) (VAR)": np.var(var), "Variance(Q) (Mean)": np.mean(var),
            "Variance(Q) (Min)": np.min(var), "Variance(Q) (Max)": np.max(var), "Variance(Q) (Median)": np.median(var)}, commit=False)
        wandb.log(
            {"Avg Effective Batch Size / Episode": np.mean(eff_batch_size), "Avg Epsilon / Episode": np.mean(eps_list),
            "Max Epsilon / Episode": np.max(eps_list), "Median Epsilon / Episode": np.median(eps_list), 
            "Min Epsilon / Episode": np.min(eps_list)}, commit=False)


