from .dqn import *


class EnsembleDQN(DQNAgent):
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
        self.qnets, self.target_nets, self.optims = [], [], []
        for i in range(self.opt.num_nets):
            qnetwork = QNetwork(self.state_size, self.action_size,
                                seed=i+opt.net_seed).to(self.device)
            self.qnets.append(qnetwork)
            self.target_nets.append(QNetwork(
                self.state_size, self.action_size, seed=i+opt.net_seed).to(self.device))
            # self.params += list(qnetwork.paramsrameters())
            self.optims.append(optim.Adam(qnetwork.parameters(), lr=opt.lr))

        self.eps = 0

    def greedy(self, Q_ensemble):
        mean_Q = np.mean(Q_ensemble, 0)
        # ------------------- action selection ------------------- #
        if self.opt.select_action == "vote":
            actions = [np.argmax(Q) for Q in Q_ensemble]
            data = Counter(actions)
            action = data.most_common(1)[0][0]
        elif self.opt.select_action == "mean":
            action = np.argmax(mean_Q)

        return action


    def epsilon_greedy(self, Q_ensemble, eps=0.):

        action = self.greedy(Q_ensemble)
        # Epsilon-greedy action selection
        if random.random() > eps:
            return action   # Selecting the action with max mean Q value
        else:
            return random.choice(np.arange(self.action_size))
        return action

    def bootstrap(self, Q_ensemble, eps=0.):

        i = random.choice(range(self.opt.num_nets))
        action = np.argmax(Q_ensemble[i])
        return action

    def thomson_sampling(self, Q_ensemble, eps=0.):

        mean_Q = np.mean(Q_ensemble, 0)
        std_Q = np.std(Q_ensemble, 0)
        ts_Q = np.random.normal(mean_Q, std_Q)
        ts_action = np.argmax(ts_Q)
        return ts_action

    def ucb(self, Q_ensemble, eps=0.):
        mean_Q = np.mean(Q_ensemble, 0)
        std_Q = np.std(Q_ensemble, 0)
        ucb_Q = mean_Q + self.opt.ucb_lambda * std_Q
        ucb_action = np.argmax(ucb_Q)
        return ucb_action      

    # Overloading Action function
    def act(self, state, eps=0., is_train=False):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        for qnet in self.qnets:
            qnet.eval()
        with torch.no_grad():
            Q_ensemble = np.array([qnet(state).cpu().data.numpy()
                               for qnet in self.qnets])

        # Act according to your preferred exploration strategy 
        if is_train:
            if self.opt.exploration == "e-greedy":
                return self.epsilon_greedy(Q_ensemble, eps), np.mean(Q_ensemble)
            elif self.opt.exploration == "bootstrap":
                return self.bootstrap(Q_ensemble), np.mean(Q_ensemble)
            elif self.opt.exploration == "ts":
                return self.thomson_sampling(Q_ensemble), np.mean(Q_ensemble)
            elif self.opt.exploration == "ucb":
                return self.ucb(Q_ensemble), np.mean(Q_ensemble)
        else:
            return self.greedy(Q_ensemble), np.mean(Q_ensemble)


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Individual Network Target & Next Actions
        Q_targets_next = torch.stack([self.target_nets[i](next_states).detach()
                                                     for i in range(self.opt.num_nets)])
        Q_targets = torch.stack([(rewards + (gamma * Q_targets_next[i].max(1)[0].unsqueeze(1) * (1 - dones)))
                                                                                 for i in range(self.opt.num_nets)])
        next_actions_ind = torch.stack([Q_targets_next[i].max(1)[1].unsqueeze(1)  # Next actions for each individual networks
        															 for i in range(self.opt.num_nets)])

        # Mean Target
        Q_targets_next_mean = Q_targets_next.mean(0)
        Q_targets_next_var = (gamma ** 2) * Q_targets_next.var(0)
        next_actions = Q_targets_next_mean.max(1)[1].unsqueeze(1)
        Q_targets_mu = rewards + \
            (gamma * Q_targets_next.mean(0).max(1)
             [0].unsqueeze(1) * (1 - dones))

        # ------------------- update Q networks ------------------- #
        eff_batch_size_list, xi_list, loss_list = [], [], []
        for i in range(self.opt.num_nets):
            Q_expected = self.qnets[i](states).gather(1, actions)
            Q_targets_var = Q_targets_next_var.gather(1, next_actions_ind[i])
            self.eps = get_optimal_xi(Q_targets_var.detach().cpu().numpy(),\
                         self.opt.minimal_eff_bs, self.xi) if self.opt.dynamic_xi else self.opt.xi
            weights = self.get_mse_weights(Q_targets_var)
            loss = self.weighted_mse(Q_expected, Q_targets[i], weights)
            # SGD step
            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()

            eff_batch_size_list.append(
                compute_eff_bs(weights.detach().cpu().numpy()))
            xi_list.append(self.xi)
            loss_list.append(loss.item())

        # In order to log loss statistics
        self.loss = np.mean(loss_list)

        # ------------------- update target networks ------------------- #
        for i in range(self.opt.num_nets):
            self.soft_update(self.qnets[i], self.target_nets[i], self.opt.tau)

        return Q_targets_var.detach().cpu().numpy(), weights.detach().cpu().numpy(), np.mean(eff_batch_size_list), np.mean(xi_list)

    def get_mse_weights(self, variance):
    	weights = torch.ones(variance.size()).to(
    	    self.device) / variance.size(0)
    	return weights

    def train_log(self, var, weights, eff_batch_size, eps_list):
        wandb.log({"IV Weights(VAR)": np.var(weights), "IV Weights(Mean)": np.mean(weights),
            "IV Weights(Min)": np.min(weights), "IV Weights(Max)": np.max(weights), "IV Weights(Median)": np.median(weights)}, commit=False)
        wandb.log({"Variance(Q) (VAR)": np.var(var), "Variance(Q) (Mean)": np.mean(var),
            "Variance(Q) (Min)": np.min(var), "Variance(Q) (Max)": np.max(var), "Variance(Q) (Median)": np.median(var)}, commit=False)
        wandb.log(
            {"Avg Effective Batch Size / Episode": np.mean(eff_batch_size), "Avg Epsilon / Episode": np.mean(eps_list),
            "Max Epsilon / Episode": np.max(eps_list), "Median Epsilon / Episode": np.median(eps_list), 
            "Min Epsilon / Episode": np.min(eps_list)}, commit=False)


class MaskEnsembleDQN(EnsembleDQN):
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

        self.mask = True
        # Mask Replay Buffer
        self.memory = MaskReplayBuffer(
            self.opt, self.action_size, 42, self.device)
        self.random_state = np.random.RandomState(11)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        mask = self.random_state.binomial(1, self.opt.mask_prob, self.opt.num_nets)
        self.memory.add(state, action, reward, next_state, done, mask)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.opt.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.opt.batch_size:
                experiences = self.memory.sample()
                return self.learn(experiences, self.opt.gamma)
            else:
                # print("Buffer Not filled to threshold yet!!! Size: %d"%(len(self.memory)), end='\r')
                return None

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, masks = experiences
        masks = masks.unsqueeze(2)

        # Individual Network Target & Next Actions
        Q_targets_next = torch.stack([self.target_nets[i](next_states).detach()
                                                     for i in range(self.opt.num_nets)])
        Q_targets = torch.stack([(rewards + (gamma * Q_targets_next[i].max(1)[0].unsqueeze(1) * (1 - dones)))
                                                                                 for i in range(self.opt.num_nets)])
        next_actions_ind = torch.stack([Q_targets_next[i].max(1)[1].unsqueeze(1)  # Next actions for each individual networks
        															 for i in range(self.opt.num_nets)])

        # Mean Target
        Q_targets_next_mean = Q_targets_next.mean(0)
        Q_targets_next_var = (gamma ** 2) * Q_targets_next.var(0)
        next_actions = Q_targets_next_mean.max(1)[1].unsqueeze(1)
        Q_targets_mu = rewards + \
            (gamma * Q_targets_next.mean(0).max(1)
             [0].unsqueeze(1) * (1 - dones))

        # ------------------- update Q networks ------------------- #
        eff_batch_size_list, xi_list, loss_list = [], [], []
        for i in range(self.opt.num_nets):
            Q_expected = self.qnets[i](states).gather(1, actions)[masks[:,i]]
            Q_target = Q_targets[i][masks[:,i]]
            Q_targets_var = Q_targets_next_var.gather(1, next_actions_ind[i])[masks[:,i]]
            self.xi = get_optimal_xi(Q_targets_var.detach().cpu().numpy(), self.opt.minimal_eff_bs, self.xi) if self.opt.dynamic_xi else self.opt.xi
            weights = self.get_mse_weights(Q_targets_var)
            loss = self.weighted_mse(Q_expected, Q_target, weights)
            # SGD step
            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()

            eff_batch_size_list.append(compute_eff_bs(weights.detach().cpu().numpy()))
            xi_list.append(self.xi)
            loss_list.append(loss.item())

        # In order to log loss statistics
        self.loss = np.mean(loss_list)

        # ------------------- update target networks ------------------- #
        for i in range(self.opt.num_nets):
            self.soft_update(self.qnets[i], self.target_nets[i], self.opt.tau)   

        return Q_targets_var.detach().cpu().numpy(), weights.detach().cpu().numpy(), np.mean(eff_batch_size_list), np.mean(xi_list)


class RPFMaskEnsembleDQN(MaskEnsembleDQN):
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
        self.qnets, self.target_nets, self.optims = [], [], [] 
        for i in range(self.opt.num_nets): 
            qnetwork = QNet_with_prior(self.state_size, self.action_size, prior_scale=opt.prior_scale, seed=i+opt.net_seed).to(self.device)     
            self.qnets.append(qnetwork)
            self.target_nets.append(QNet_with_prior(self.state_size, self.action_size, prior_scale=opt.prior_scale, seed=i+opt.net_seed).to(self.device))
            # self.params += list(qnetwork.paramsrameters())
            self.optims.append(optim.Adam(qnetwork.net.parameters(), lr=opt.lr))


class BootstrapDQN(MaskEnsembleDQN):
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

    def act(self, state, eps=0., i=0, is_train=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            i (int): network for the current episode
            is_train (bool): training mode or not
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        for qnet in self.qnets:
            qnet.eval()
        with torch.no_grad():
            Q_ensemble = np.array([qnet(state).cpu().data.numpy() for qnet in self.qnets])

            # ------------------- action selection ------------------- #
            if is_train:
                action = np.argmax(Q_ensemble[i])
            else:
                actions = [np.argmax(Q) for Q in Q_ensemble]
                data = Counter(actions)
                action = data.most_common(1)[0][0]

        for qnet in self.qnets:
            qnet.train()
        # var_Q = np.var(Q_ensemble, 0)
        # Epsilon-greedy action selection
        if random.random() > eps:
            return action, np.mean(Q_ensemble)   # Selecting the action with max mean Q value
        else:
            return random.choice(np.arange(self.action_size)), np.mean(Q_ensemble)

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
        scores = []   # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            state = self.env.reset()
            score, ep_var, ep_weights, eff_bs_list, xi_list, ep_Q, ep_loss = 0, [], [], [], [], [], []   # list containing scores from each episode
            # Select Network to take actions in the environment for the current episode
            curr_net = random.choice(range(self.opt.num_nets)) 
            for t in range(max_t):
                action, Q = self.act(state, eps, i=curr_net, is_train=True)
                next_state, reward, done, _ = self.env.step(action)
                logs = self.step(state, action, reward, next_state, done)
                state = next_state
                if done:
                    reward += self.opt.end_reward
                score += reward
                if logs is not None:
                    try:
                        ep_var.extend(logs[0])
                        ep_weights.extend(logs[1])
                        eff_bs_list.append(logs[2])
                        xi_list.append(logs[3])
                    except:
                        pass
                ep_Q.append(Q)
                ep_loss.append(self.loss)
                if done:
                    break 

            #wandb.log({"V(s) (VAR)": np.var(ep_Q), "V(s) (Mean)": np.mean(ep_Q),
            #    "V(s) (Min)": np.min(ep_Q), "V(s) (Max)": np.max(ep_Q), 
            #    "V(s) (Median)": np.median(ep_Q)}, commit=False)
            #wandb.log({"Loss (VAR)": np.var(ep_loss), "Loss (Mean)": np.mean(ep_loss),
            #    "Loss (Min)": np.min(ep_loss), "Loss (Max)": np.max(ep_loss), 
            #    "Loss (Median)": np.median(ep_loss)}, commit=False)
            #if len(ep_var) > 0: # if there are entries in the variance list
            #    self.train_log(ep_var, ep_weights, eff_bs_list, eps_list)
            if i_episode % self.opt.test_every == 0:
                self.test(episode=i_episode)
 
            scores_window.append(score)        # save most recent score
            scores.append(score)               # save most recent score
            eps = max(eps_end, eps_decay*eps)  # decrease epsilon
            #wandb.log({"Moving Average Return/100episode": np.mean(scores_window)})
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #self.save(scores)

class RPFBootstrapDQN(BootstrapDQN):
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
        self.qnets, self.target_nets, self.optims = [], [], [] 
        for i in range(self.opt.num_nets): 
            qnetwork = QNet_with_prior(self.state_size, self.action_size, prior_scale=opt.prior_scale, seed=i+opt.net_seed).to(self.device)     
            self.qnets.append(qnetwork)
            self.target_nets.append(QNet_with_prior(self.state_size, self.action_size, prior_scale=opt.prior_scale, seed=i+opt.net_seed).to(self.device))
            # self.params += list(qnetwork.paramsrameters())
            self.optims.append(optim.Adam(qnetwork.net.parameters(), lr=opt.lr))



class Lakshminarayan(EnsembleDQN):
    def __init__(self, env, opt, device="cuda"):
        super().__init__(env, opt, device)
        self.qnets, self.target_nets, self.optims = [], [], [] 
        for i in range(self.opt.num_nets): 
            qnetwork = TwoHeadQNetwork(self.state_size, self.action_size, seed=i+opt.net_seed).to(self.device)     
            self.qnets.append(qnetwork)
            self.target_nets.append(TwoHeadQNetwork(self.state_size, self.action_size, seed=i+opt.net_seed).to(self.device))
            # self.params += list(qnetwork.paramsrameters())
            self.optims.append(optim.Adam(qnetwork.parameters(), lr=opt.lr))


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Individual Network Target & Next Actions
        Q_targets_next_all =  torch.stack([torch.stack(self.target_nets[i](next_states, is_train=True))
                                                     for i in range(self.opt.num_nets)])
        Q_targets_next, Q_targets_next_var = Q_targets_next_all[:,0,:,:].detach(), torch.exp(Q_targets_next_all[:,1,:,:].detach())

        Q_targets_all = torch.stack([(rewards.repeat(1,self.action_size) + (gamma * Q_targets_next[i] * (1 - dones.repeat(1,self.action_size))))
                                                                  for i in range(self.opt.num_nets)])  # mu_i
        # print(Q_targets_next.size(), Q_targets_next_var.size(), Q_targets_next_all.size()   )
        Q_targets = torch.stack([(rewards + (gamma * Q_targets_next[i].max(1)[0].unsqueeze(1) * (1 - dones)))
                                                                                 for i in range(self.opt.num_nets)])  # mu_i
        next_actions_ind = torch.stack([Q_targets_next[i].max(1)[1].unsqueeze(1)  # Next actions for each individual networks
                                                                     for i in range(self.opt.num_nets)])


        # Mean Target
        Q_targets_next_mean = Q_targets_next.mean(0) 
        Q_targets_next_var = (gamma ** 2) * Q_targets_next.var(0) 
        next_actions = Q_targets_next_mean.max(1)[1].unsqueeze(1)
        Q_targets_mu = rewards + \
            (gamma * Q_targets_next.mean(0).max(1)   # mu*
             [0].unsqueeze(1) * (1 - dones))

        # print(Q_targets_next_var.size(), Q_targets.size(), Q_targets_mu.size())
        # Calculate Variance for Gaussian Mixture
        Q_target_net_var = (gamma**2) * Q_targets_next_var # var_i's 

        # Variance of Gaussian Mixture = Sum( var_i + mean_i^2 - (mu*)^2 )
        Q_var_mixture = (Q_targets_next_var + Q_targets_all**2 - Q_targets_all.mean(0).repeat(self.opt.num_nets,1,1)**2).mean(0) 

        # ------------------- update Q networks ------------------- #
        eff_batch_size_list, xi_list, loss_list = [], [], []
        for i in range(self.opt.num_nets):
            # Q_expected = self.qnets[i](states).gather(1, actions)
            Q_expected, Q_log_var  = [x.gather(1, actions) for x in self.qnets[i](states, True)] 
            Q_targets_var = Q_var_mixture.gather(1, next_actions_ind[i])
            self.xi = get_optimal_xi(Q_targets_var.detach().cpu().numpy(),\
                         self.opt.minimal_eff_bs, self.xi) if self.opt.dynamic_xi else self.opt.xi
            weights = self.get_mse_weights(Q_targets_var)
            loss = self.weighted_mse(Q_expected, Q_targets[i], weights)
            # Compute Loss Attenuation 
            y, mu, var = Q_targets[i], Q_expected, torch.exp(Q_log_var)
            std = torch.sqrt(var) 
            # print(y.size(), mu.size(), std.size())
            lossatt = torch.mean((y - mu)**2 / (2 * torch.square(std)) + (1/2) * torch.log(torch.square(std)))
            loss += self.opt.loss_att_weight*lossatt
            # SGD step
            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()

            eff_batch_size_list.append(
                compute_eff_bs(weights.detach().cpu().numpy()))
            xi_list.append(self.xi)
            loss_list.append(loss.item())

        # In order to log loss statistics
        self.loss = np.mean(loss_list)

        # ------------------- update target networks ------------------- #
        for i in range(self.opt.num_nets):
            self.soft_update(self.qnets[i], self.target_nets[i], self.opt.tau)

        return Q_targets_var.detach().cpu().numpy(), weights.detach().cpu().numpy(), np.mean(eff_batch_size_list), np.mean(xi_list)

class LakshmiBootstrapDQN(BootstrapDQN):
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
        self.qnets, self.target_nets, self.optims = [], [], []
        for i in range(self.opt.num_nets):
            qnetwork = TwoHeadQNetwork(self.state_size, self.action_size, seed=i+opt.net_seed).to(self.device)
            self.qnets.append(qnetwork)
            self.target_nets.append(TwoHeadQNetwork(self.state_size, self.action_size, seed=i+opt.net_seed).to(self.device))
            # self.params += list(qnetwork.paramsrameters())
            self.optims.append(optim.Adam(qnetwork.parameters(), lr=opt.lr))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, masks = experiences

        # Individual Network Target & Next Actions
        Q_targets_next_all =  torch.stack([torch.stack(self.target_nets[i](next_states, is_train=True))
                                                     for i in range(self.opt.num_nets)])
        Q_targets_next, Q_targets_next_var = Q_targets_next_all[:,0,:,:].detach(), torch.exp(Q_targets_next_all[:,1,:,:].detach())

        Q_targets_all = torch.stack([(rewards.repeat(1,self.action_size) + (gamma * Q_targets_next[i] * (1 - dones.repeat(1,self.action_size))))
                                                                  for i in range(self.opt.num_nets)])  # mu_i
        # print(Q_targets_next.size(), Q_targets_next_var.size(), Q_targets_next_all.size()   )
        Q_targets = torch.stack([(rewards + (gamma * Q_targets_next[i].max(1)[0].unsqueeze(1) * (1 - dones)))
                                                                                 for i in range(self.opt.num_nets)])  # mu_i
        next_actions_ind = torch.stack([Q_targets_next[i].max(1)[1].unsqueeze(1)  # Next actions for each individual networks
                                                                     for i in range(self.opt.num_nets)])


        # Mean Target
        Q_targets_next_mean = Q_targets_next.mean(0)
        Q_targets_next_var = (gamma ** 2) * Q_targets_next.var(0)
        next_actions = Q_targets_next_mean.max(1)[1].unsqueeze(1)
        Q_targets_mu = rewards + \
            (gamma * Q_targets_next.mean(0).max(1)   # mu*
             [0].unsqueeze(1) * (1 - dones))

        # print(Q_targets_next_var.size(), Q_targets.size(), Q_targets_mu.size())
        # Calculate Variance for Gaussian Mixture
        Q_target_net_var = (gamma**2) * Q_targets_next_var # var_i's

        # Variance of Gaussian Mixture = Sum( var_i + mean_i^2 - (mu*)^2 )
        Q_var_mixture = (Q_targets_next_var + Q_targets_all**2 - Q_targets_all.mean(0).repeat(self.opt.num_nets,1,1)**2).mean(0)

        # ------------------- update Q networks ------------------- #
        eff_batch_size_list, xi_list, loss_list = [], [], []
        for i in range(self.opt.num_nets):
            # Q_expected = self.qnets[i](states).gather(1, actions)
            Q_expected, Q_log_var  = [x.gather(1, actions) for x in self.qnets[i](states, True)]
            Q_expected, Q_log_var = Q_expected[masks[:,i]], Q_log_var[masks[:,i]]
            Q_targets_var = Q_var_mixture.gather(1, next_actions_ind[i])[masks[:,i]]
            self.xi = get_optimal_xi(Q_targets_var.detach().cpu().numpy(),\
                         self.opt.minimal_eff_bs, self.xi) if self.opt.dynamic_xi else self.opt.xi
            weights = self.get_mse_weights(Q_targets_var)
            loss = self.weighted_mse(Q_expected, Q_targets[i][masks[:,i]], weights)
            # Compute Loss Attenuation
            y, mu, var = Q_targets[i][masks[:,i]], Q_expected, torch.exp(Q_log_var)
            std = torch.sqrt(var)
            # print(y.size(), mu.size(), std.size())
            lossatt = torch.mean((y - mu)**2 / (2 * torch.square(std)) + (1/2) * torch.log(torch.square(std)))
            loss += self.opt.loss_att_weight*lossatt
            # SGD step
            self.optims[i].zero_grad()
            loss.backward()
            self.optims[i].step()

            eff_batch_size_list.append(
                compute_eff_bs(weights.detach().cpu().numpy()))
            xi_list.append(self.xi)
            loss_list.append(loss.item())

        # In order to log loss statistics
        self.loss = np.mean(loss_list)

        # ------------------- update target networks ------------------- #
        for i in range(self.opt.num_nets):
            self.soft_update(self.qnets[i], self.target_nets[i], self.opt.tau)

        return Q_targets_var.detach().cpu().numpy(), weights.detach().cpu().numpy(), np.mean(eff_batch_size_list), np.mean(xi_list)

