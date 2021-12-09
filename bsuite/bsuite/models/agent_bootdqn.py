import dm_env
import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim
import typing
import wandb

from utils.memory import Experience, ReplayMemory, PrioritizedReplayMemory
from models.qnet_MCdrop import Dqn, DuelDQN, TwoHeadDqn

from scipy.optimize import minimize
from collections import namedtuple, deque, Counter

# from qnet import Dqn, DuelDQN

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


class BootstrapDQN:
    def __init__(self,
                 opt,
                 action_spec: dm_env.specs.DiscreteArray,
                 observation_spec: dm_env.specs.Array,
                 num_ensemble: int,
                 net_seed: int,
                 device: torch.device,
                 settings: dict) -> None:
        """
        Initializes the agent,  constructs the qnet and the q_target, initializes the optimizer and ReplayMemory.
        Args:
            action_spec(dm_env.specs.DiscreteArray): description of the action space of the environment
            observation_spec(dm_env.specs.Array): description of observations form the environment
            device(str): "gpu" or "cpu"
            settings(dict): dictionary with settings
        """
        self.device = device
        self.opt = opt
        self.num_ensemble = num_ensemble
        action_size = action_spec.num_values
        state_size = np.prod(observation_spec.shape)
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = settings['batch_size']
        self.noisy_nets = settings['qnet_settings']['noisy_nets']

        self.qnets, self.tnets, self.optims = [], [], []
        for i in range(num_ensemble):
            if settings["duelling_dqn"]:
                qnet = DuelDQN(state_size, action_size, settings['qnet_settings']).to(device)
                q_target = DuelDQN(state_size, action_size, settings['qnet_settings']).to(device)
            else:
                qnet = Dqn(state_size, action_size, settings['qnet_settings'], seed=opt.net_seed+i).to(device)
                q_target = Dqn(state_size, action_size, settings['qnet_settings'], seed=opt.net_seed+i).to(device)
                self.drop_porb = 0

            self.qnets.append(qnet)
            q_target.load_state_dict(qnet.state_dict())
            self.tnets.append(q_target)
            self.optims.append(optim.Adam(qnet.parameters(), lr=settings['lr']))

        self.epsilon = settings["epsilon_start"]
        self.decay = settings["epsilon_decay"]
        self.epsilon_min = settings["epsilon_min"]
        self.gamma = settings['gamma']

        self.start_optimization = settings["start_optimization"]
        self.update_qnet_every = settings["update_qnet_every"]
        self.update_target_every = settings["update_target_every"]
        self.number_steps = 0
        self.ddqn = settings["ddqn"]

        self.xi = settings["xi"]
        self.dynamic_xi = settings["dynamic_xi"]
        self.minimal_eff_bs_ratio = settings["minimal_eff_bs_ratio"]
        self.minimal_eff_bs = int(self.batch_size * self.minimal_eff_bs_ratio)
        self.mask_prob = settings["mask_prob"]

        self._rng = np.random.RandomState(net_seed)
        self._active_head = self._rng.randint(self.num_ensemble)
        # Initialize replay memory
        self.prioritized_replay = settings["prioritized_buffer"]
        if self.prioritized_replay:
            self.memory = PrioritizedReplayMemory(device, settings["buffer_size"], self.gamma, settings["n_steps"],
                                                  settings["alpha"], settings["beta0"], settings["beta_increment"])
        else:
            self.memory = ReplayMemory(device, settings["buffer_size"], self.gamma, settings["n_steps"])
        return

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """
        Returns an action following an epsilon-greedy policy.
        Args:
            timestep(dm_env.TimeStep): An observation from the environment

        Returns:
            int: The chosen action.
        """
        observation = np.array(timestep.observation).flatten()
        observation = torch.from_numpy(observation).float().to(self.device)
        self.number_steps += 1

        if not self.noisy_nets:
            self.update_epsilon()

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return int(self.qnets[self._active_head].get_max_action(observation))


    def greedy(self, Q_ensemble):
        mean_Q = np.mean(Q_ensemble, 0)
        # ------------------- action selection ------------------- #
        # if self.opt.select_action == "vote":
        actions = [np.argmax(Q) for Q in Q_ensemble]
        data = Counter(actions)
        action = data.most_common(1)[0][0]
        # elif self.opt.select_action == "mean":
        # action = np.argmax(mean_Q)

        return action

    def select_action_test(self, timestep: dm_env.TimeStep) -> int:
        """
        Returns an action following an epsilon-greedy policy.
        Args:
            timestep(dm_env.TimeStep): An observation from the environment

        Returns:
            int: The chosen action.
        """
        observation = np.array(timestep.observation).flatten()
        observation = torch.from_numpy(observation).float().to(self.device)
        # self.number_steps += 1

        with torch.no_grad():
            Q_ensemble = np.array([qnet(observation).cpu().data.numpy()
                               for qnet in self.qnets])

        return int(self.greedy(Q_ensemble))

    def update_epsilon(self) -> None:
        """
        Decays epsilon until self.epsilon_min
        Returns:
            None
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay

    @staticmethod
    def calc_loss(q_observed: torch.Tensor,
                  q_target: torch.Tensor,
                  weights: torch.Tensor) -> typing.Tuple[torch.Tensor, np.float64]:
        """
        Returns the mean weighted MSE loss and the loss for each sample
        Args:
            q_observed(torch.Tensor): calculated q_value
            q_target(torch.Tensor):   target q-value
            weights: weights of the batch samples

        Returns:
            tuple(torch.Tensor, np.float64): mean squared error loss, loss for each indivdual sample
        """
        losses = functional.mse_loss(q_observed, q_target, reduction='none')
        loss = (weights * losses).sum()
        return loss, losses.cpu().detach().numpy() + 1e-8

    def update(self,
               step: dm_env.TimeStep,
               action: int,
               next_step: dm_env.TimeStep) -> None:
        """
        Adds experience to the replay memory, performs an optimization_step and updates the q_target neural network.
        Args:
            step(dm_env.TimeStep): Current observation from the environment
            action(int): The action that was performed by the agent.
            next_step(dm_env.TimeStep): Next observation from the environment
        Returns:
            None
        """

        logs = []
        observation = np.array(step.observation).flatten()
        next_observation = np.array(next_step.observation).flatten()
        done = next_step.last()

        if next_step.last():
          self._active_head = self._rng.randint(self.num_ensemble)

        exp = Experience(observation,
                         action,
                         next_step.reward,
                         next_step.discount,
                         next_observation,
                         0,
                         done,
                         self._rng.binomial(1, self.mask_prob, self.num_ensemble).astype(np.float32)
                         )
        self.memory.add(exp)

        if self.memory.number_samples() < self.start_optimization:
            return logs

        if self.number_steps % self.update_qnet_every == 0:
            s0, a0, n_step_reward, discount, s1, _, dones, indices, weights, masks = self.memory.sample_batch(self.batch_size)
            logs = self.optimization_step(s0, a0, n_step_reward, discount, s1, indices, weights, masks)

        if self.number_steps % self.update_target_every == 0:
            for i in range(self.num_ensemble):
                self.tnets[i].load_state_dict(self.qnets[i].state_dict())
        return logs

    def optimization_step(self,
                          s0: torch.Tensor,
                          a0: torch.Tensor,
                          n_step_reward: torch.Tensor,
                          discount: torch.Tensor,
                          s1: torch.Tensor,
                          indices: typing.Optional[torch.Tensor],
                          weights: typing.Optional[torch.Tensor],
                          masks: torch.Tensor) -> None:
        """
        Calculates the Bellmann update and updates the qnet.
        Args:
            s0(torch.Tensor): current state
            a0(torch.Tensor): current action
            n_step_reward(torch.Tensor): n-step reward
            discount(torch.Tensor): discount factor
            s1(torch.Tensor): next state
            indices(torch.Tensor): batch indices, needed for prioritized replay. Not used yet.
            weights(torch.Tensor): weights needed for prioritized replay

        Returns:
            None
        """

        with torch.no_grad():
            if self.noisy_nets:
                self.q_target.reset_noise()
                self.qnet.reset_noise()

            # Calculating the target values
            next_q_vals = torch.stack([self.tnets[i](s1) for i in range(self.num_ensemble)])
            next_actions = torch.stack([next_q_vals[i].max(1)[1] for i in range(self.num_ensemble)])
            # if self.ddqn:
            #     a1 = torch.argmax(self.qnet(s1), dim=1).unsqueeze(-1)
            #     next_q_val = next_q_vals.gather(1, a1).squeeze()
            # else:
            #     next_q_val = torch.max(next_q_vals, dim=2).values
            q_targets = torch.stack([n_step_reward.squeeze() + self.gamma * discount.squeeze() * torch.max(next_q_vals[i], dim=1).values\
                                        for i in range(self.num_ensemble)])
            # print(discount.size(), next_q_vals.size(), next_actions.size())
            q_target_var_all = (self.gamma**2) * (discount.repeat(1, self.action_size)**2) * next_q_vals.var(0)

        eff_batch_size_list, xi_list, loss_list = [], [], []        
        for i in range(self.num_ensemble):
            # print(next_actions[i].size(), masks.size(), q_targets.size())
            q_target_var = q_target_var_all.gather(1, next_actions[i].unsqueeze(-1).long())[masks[:, i, 0]]
            # print(q_target_var.size())
            self.xi = get_optimal_xi(q_target_var.detach().cpu().numpy(
                ), self.minimal_eff_bs, self.xi) if self.dynamic_xi else self.xi
            weights = self.get_mse_weights(q_target_var)
            q_observed = self.qnets[i](s0).gather(1, a0.long()).squeeze()[masks[:, i, 0]]
            critic_loss, batch_loss = self.calc_loss(q_observed, q_targets[i][masks[:, i, 0]], weights.to(self.device))

            # Backpropagation of the gradients
            self.optims[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qnets[i].parameters(), 5)
            self.optims[i].step()

            eff_batch_size_list.append(
                compute_eff_bs(weights.detach().cpu().numpy()))
            xi_list.append(self.xi)
            # loss_list.append(loss.item())

        # Update replay memory
        self.memory.update_priorities(indices, batch_loss)
        return q_target_var.detach().cpu().numpy(), weights.squeeze().detach().cpu().numpy(), np.mean(eff_batch_size_list), np.mean(xi_list)

    def train_log(self, var, weights, eff_batch_size, eps_list):
        wandb.log({"IV Weights(VAR)": np.var(weights), "IV Weights(Mean)": np.mean(weights),
            "IV Weights(Min)": np.min(weights), "IV Weights(Max)": np.max(weights), "IV Weights(Median)": np.median(weights)}, commit=False)
        wandb.log({"Variance(Q) (VAR)": np.var(var), "Variance(Q) (Mean)": np.mean(var),
            "Variance(Q) (Min)": np.min(var), "Variance(Q) (Max)": np.max(var), "Variance(Q) (Median)": np.median(var)}, commit=False)
        wandb.log(
            {"Avg Effective Batch Size / Episode": np.mean(eff_batch_size), "Avg Epsilon / Episode": np.mean(eps_list),
            "Max Epsilon / Episode": np.max(eps_list), "Median Epsilon / Episode": np.median(eps_list), 
            "Min Epsilon / Episode": np.min(eps_list)}, commit=False)

    def get_mse_weights(self, variance):
        return torch.ones(variance.size()) / variance.size()[0]


class EnsembleDQN(BootstrapDQN):
    def __init__(self,
                 opt,
                 action_spec: dm_env.specs.DiscreteArray,
                 observation_spec: dm_env.specs.Array,
                 num_ensemble: int,
                 net_seed: int,
                 device: torch.device,
                 settings: dict) -> None:

        super().__init__(opt, action_spec, observation_spec, num_ensemble, net_seed, device, settings)

    def greedy(self, Q_ensemble):
        mean_Q = np.mean(Q_ensemble, 0)
        # ------------------- action selection ------------------- #
        # if self.opt.select_action == "vote":
        # actions = [np.argmax(Q) for Q in Q_ensemble]
        # data = Counter(actions)
        # action = data.most_common(1)[0][0]
        # elif self.opt.select_action == "mean":
        action = np.argmax(mean_Q)

        return action

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """
        Returns an action following an epsilon-greedy policy.
        Args:
            timestep(dm_env.TimeStep): An observation from the environment

        Returns:
            int: The chosen action.
        """
        observation = np.array(timestep.observation).flatten()
        observation = torch.from_numpy(observation).float().to(self.device)
        self.number_steps += 1

        for qnet in self.qnets:
            qnet.eval()

        with torch.no_grad():
            Q_ensemble = np.array([qnet(observation).cpu().data.numpy()
                               for qnet in self.qnets])

        if not self.noisy_nets:
            self.update_epsilon()

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return int(self.greedy(Q_ensemble))

    def select_action_test(self, timestep: dm_env.TimeStep) -> int:
        """
        Returns an action following an epsilon-greedy policy.
        Args:
            timestep(dm_env.TimeStep): An observation from the environment

        Returns:
            int: The chosen action.
        """
        observation = np.array(timestep.observation).flatten()
        observation = torch.from_numpy(observation).float().to(self.device)
        self.number_steps += 1

        for qnet in self.qnets:
            qnet.eval()

        with torch.no_grad():
            Q_ensemble = np.array([qnet(observation).cpu().data.numpy()
                               for qnet in self.qnets])

        return int(self.greedy(Q_ensemble))



class LakshmiBootDQN(BootstrapDQN):
    def __init__(self,
                 opt,
                 action_spec: dm_env.specs.DiscreteArray,
                 observation_spec: dm_env.specs.Array,
                 num_ensemble: int,
                 net_seed: int,
                 device: torch.device,
                 settings: dict) -> None:

        super().__init__(opt, action_spec, observation_spec, num_ensemble, net_seed, device, settings)


        self.qnets, self.tnets, self.optims = [], [], []
        for i in range(num_ensemble):
            if settings["duelling_dqn"]:
                qnet = DuelDQN(self.state_size, self.action_size, settings['qnet_settings']).to(device)
                q_target = DuelDQN(self.state_size, self.action_size, settings['qnet_settings']).to(device)
            else:
                qnet = TwoHeadDqn(self.state_size, self.action_size, settings['qnet_settings'], seed=opt.net_seed+i).to(device)
                q_target = TwoHeadDqn(self.state_size, self.action_size, settings['qnet_settings'], seed=opt.net_seed+i).to(device)
                self.drop_porb = 0

            self.qnets.append(qnet)
            q_target.load_state_dict(qnet.state_dict())
            self.tnets.append(q_target)
            self.optims.append(optim.Adam(qnet.parameters(), lr=settings['lr']))

    def optimization_step(self,
                          s0: torch.Tensor,
                          a0: torch.Tensor,
                          n_step_reward: torch.Tensor,
                          discount: torch.Tensor,
                          s1: torch.Tensor,
                          indices: typing.Optional[torch.Tensor],
                          weights: typing.Optional[torch.Tensor],
                          masks: torch.Tensor) -> None:
        """
        Calculates the Bellmann update and updates the qnet.
        Args:
            s0(torch.Tensor): current state
            a0(torch.Tensor): current action
            n_step_reward(torch.Tensor): n-step reward
            discount(torch.Tensor): discount factor
            s1(torch.Tensor): next state
            indices(torch.Tensor): batch indices, needed for prioritized replay. Not used yet.
            weights(torch.Tensor): weights needed for prioritized replay

        Returns:
            None
        """

        with torch.no_grad():
            if self.noisy_nets:
                self.q_target.reset_noise()
                self.qnet.reset_noise()

            # Calculating the target values
            next_q_vals_all = torch.stack([torch.stack(self.tnets[i](s1, is_training=True))
                                                         for i in range(self.num_ensemble)])
            next_q_vals, next_q_vals_std = next_q_vals_all[:,0], next_q_vals_all[:,1]
            next_actions = torch.stack([next_q_vals[i].max(1)[1] for i in range(self.num_ensemble)])
            # q_targets_all = torch.stack([n_step_reward.squeeze() + self.gamma * discount.squeeze() * next_q_vals[i]\
                                                                                    # for i in range(self.num_ensemble)])
            # print(discount.size(), n_step_reward.size(), next_q_vals.size())
            q_targets_all = torch.stack([n_step_reward.repeat(1, self.action_size) + self.gamma * discount.repeat(1, self.action_size) * next_q_vals[i]\
                                                                                    for i in range(self.num_ensemble)])
            q_targets = torch.stack([n_step_reward.squeeze() + self.gamma * discount.squeeze() * torch.max(next_q_vals[i], dim=1).values\
                                                                                    for i in range(self.num_ensemble)])
            # print(discount.size(), next_q_vals.size(), next_actions.size())
            # q_target_var_all = (self.gamma**2) * (discount.repeat(1, self.action_size)**2) * next_q_vals.var(0)
            next_q_vals_std = (self.gamma**2) * torch.stack([next_q_vals_std[i].gather(1, next_actions[i].unsqueeze(-1).long()) for i in range(self.num_ensemble)])
            # print(next_q_vals.size(), next_q_vals_std.size(), q_targets.size())
            # print((next_q_vals_std**2 + q_targets**2 - q_targets.mean(0).unsqueeze(-1).repeat(self.num_ensemble,1,1)**2).mean(0).size())
            q_var_mixture = (discount.repeat(1, self.action_size)**2) * (next_q_vals_std**2 + q_targets_all**2 - q_targets_all.mean(0).unsqueeze(0).repeat(self.num_ensemble,1,1)**2).mean(0)

        eff_batch_size_list, xi_list, loss_list = [], [], []        
        for i in range(self.num_ensemble):
            # print(next_actions[i].size(), masks.size(), q_targets.size())
            q_target_var = q_var_mixture.gather(1, next_actions[i].unsqueeze(-1).long())[masks[:, i, 0]]
            # print(q_target_var.size())
            self.xi = get_optimal_xi(q_target_var.detach().cpu().numpy(
                ), self.minimal_eff_bs, self.xi) if self.dynamic_xi else self.xi
            weights = self.get_mse_weights(q_target_var)
            q_observed, q_observed_std = self.qnets[i](s0, is_training=True)
            q_observed = q_observed.gather(1, a0.long()).squeeze()#[masks[:, i, 0]]
            q_observed_std = q_observed_std.gather(1, a0.long()).squeeze()#[masks[:,i,0]]

            y, mu, std = q_targets, q_observed, q_observed_std
            lossatt = torch.mean((y - mu)**2 / (2 * (std**2)) + (1/2) * torch.log((std**2)))

            critic_loss, batch_loss = self.calc_loss(q_observed[masks[:, i, 0]], q_targets[i][masks[:, i, 0]], weights.to(self.device))

            # Backpropagation of the gradients
            self.optims[i].zero_grad()
            critic_loss += self.opt.lossatt_weight * lossatt
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qnets[i].parameters(), 5)
            self.optims[i].step()

            eff_batch_size_list.append(
                compute_eff_bs(weights.detach().cpu().numpy()))
            xi_list.append(self.xi)
            # loss_list.append(loss.item())

        # Update replay memory
        self.memory.update_priorities(indices, batch_loss)
        return q_target_var.detach().cpu().numpy(), weights.squeeze().detach().cpu().numpy(), np.mean(eff_batch_size_list), np.mean(xi_list)


class IV_BootstrapDQN(BootstrapDQN):
    def __init__(self,
                 opt,
                 action_spec: dm_env.specs.DiscreteArray,
                 observation_spec: dm_env.specs.Array,
                 num_ensemble: int,
                 net_seed: int,
                 device: torch.device,
                 settings: dict) -> None:

        super().__init__(opt, action_spec, observation_spec, num_ensemble, net_seed, device, settings)

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
        return self.iv_weights(variance)


class IV_DQN(EnsembleDQN):
    def __init__(self,
                 opt,
                 action_spec: dm_env.specs.DiscreteArray,
                 observation_spec: dm_env.specs.Array,
                 num_ensemble: int,
                 net_seed: int,
                 device: torch.device,
                 settings: dict) -> None:

        super().__init__(opt, action_spec, observation_spec, num_ensemble, net_seed, device, settings)

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
        return self.iv_weights(variance)

class IV_LakshmiBootDQN(LakshmiBootDQN):
    def __init__(self,
                 opt,
                 action_spec: dm_env.specs.DiscreteArray,
                 observation_spec: dm_env.specs.Array,
                 num_ensemble: int,
                 net_seed: int,
                 device: torch.device,
                 settings: dict) -> None:

        super().__init__(opt, action_spec, observation_spec, num_ensemble, net_seed, device, settings)

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
        return self.iv_weights(variance)


class SunriseDQN(BootstrapDQN):
    def __init__(self,
                 opt,
                 action_spec: dm_env.specs.DiscreteArray,
                 observation_spec: dm_env.specs.Array,
                 num_ensemble: int,
                 net_seed: int,
                 device: torch.device,
                 settings: dict) -> None:

        super().__init__(opt, action_spec, observation_spec, num_ensemble, net_seed, device, settings)
        self.opt = opt

    def sunrise_weights(self, variance):
        temp = self.opt.sunrise_temp
        weights = torch.sigmoid(-torch.sqrt(variance)*temp) + 0.5
        return weights

    def get_mse_weights(self, variance):
    	return self.sunrise_weights(variance)

