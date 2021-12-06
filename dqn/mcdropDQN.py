from .dqn import * 


class MCDropDQN(DQNAgent):

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
        self.qnetwork_local = MCDropQNet(self.state_size, self.action_size, opt.net_seed, p=opt.mcd_prob).to(self.device)
        self.qnetwork_target = MCDropQNet(self.state_size, self.action_size, opt.net_seed, p=opt.mcd_prob).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=opt.lr)

    # Overloading Action function
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            Q_ensemble = [self.qnetwork_local(state).cpu().data.numpy() for i in range(self.opt.mcd_samples)]
            mean_Q = np.mean(np.array(Q_ensemble), 0)
            # var_Q = np.var(np.array(Q_ensemble), 0)
        	# ------------------- action selection ------------------- #
            if self.opt.select_action == "vote":
                actions = [np.argmax(Q) for Q in Q_ensemble]
                data = Counter(actions)
                action = data.most_common(1)[0][0]
            elif self.opt.select_action == "mean":
                action = np.argmax(mean_Q)

        self.qnetwork_local.train()
        #var_Q = np.var(Q_ensemble, 0)
        # Epsilon-greedy action selection
        if random.random() > eps:
            return action   # Selecting the action with max mean Q value
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Individual Network Target & Next Actions
        Q_targets_next = torch.stack([self.qnetwork_target(next_states).detach()\
                                                     for i in range(self.opt.mcd_samples)])
        Q_targets = torch.stack([(rewards + (gamma * Q_targets_next[i].max(1)[0].unsqueeze(1) * (1 - dones)))\
                                                                                 for i in range(self.opt.mcd_samples)])
        next_actions_ind = torch.stack([Q_targets_next[i].max(1)[1].unsqueeze(1)  # Next actions for each individual networks
        															 for i in range(self.opt.mcd_samples)])


        # Mean Target 
        Q_targets_next_mean = Q_targets_next.mean(0)
        Q_targets_next_var = (gamma ** 2) * Q_targets_next.var(0)
        next_actions = Q_targets_next_mean.max(1)[1].unsqueeze(1)
        Q_targets_mu = rewards + (gamma * Q_targets_next.mean(0).max(1)[0].unsqueeze(1) * (1 - dones))

        # ------------------- update Q networks ------------------- #
        eff_batch_size_list, eps = [], 0
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        if self.opt.mean_target:
            Q_targets_var = Q_targets_next_var.gather(1, next_actions)
            self.eps = get_optimal_eps(Q_targets_var.detach().cpu().numpy(), self.opt.minimal_eff_bs, self.eps) if self.opt.dynamic_eps else self.opt.eps
            weights = self.get_mse_weights(Q_targets_var)
            loss = self.weighted_mse(Q_expected, Q_targets_mu, weights)
        else:
            Q_targets_var = Q_targets_next_var.gather(1, next_actions_ind[0])
            self.eps = get_optimal_eps(Q_targets_var.detach().cpu().numpy(), self.opt.minimal_eff_bs, self.eps) if self.opt.dynamic_eps else self.opt.eps
            weights = self.get_mse_weights(Q_targets_var)
            loss = self.weighted_mse(Q_expected, Q_targets[0], weights)
	        # SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        eff_batch_size_list = compute_eff_bs(weights.detach().cpu().numpy())

        # ------------------- update target networks ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.opt.tau)                     

        return Q_targets_var.detach().cpu().numpy(), weights.detach().cpu().numpy(), eff_batch_size_list


    def get_mse_weights(self, variance, eps):
    	weights = torch.ones(variance.size()).to(self.device) / self.opt.batch_size
    	return weights


    def train_log(self, var, weights, eff_batch_size):
        wandb.log({"IV Weights(VAR)": np.var(weights), "IV Weights(Mean)": np.mean(weights),\
            "IV Weights(Min)": np.min(weights), "IV Weights(Max)": np.max(weights), "IV Weights(Median)": np.median(weights)}, commit=False)
        wandb.log({"Variance(Q) (VAR)": np.var(var), "Variance(Q) (Mean)": np.mean(var),\
            "Variance(Q) (Min)": np.min(var), "Variance(Q) (Max)": np.max(var), "Variance(Q) (Median)": np.median(var)}, commit=False)
        wandb.log({"Avg Effective Batch Size / Episode": np.mean(eff_batch_size)}, commit=False)
