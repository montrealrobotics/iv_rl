import dm_env
import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim
import typing

from utils.memory import Experience, ReplayMemory, PrioritizedReplayMemory
from models.qnet import Dqn, DuelDQN
from models.qnet_EP import Epn

from uncertaintylearning.features.density_estimator import MAFMOGDensityEstimator, FixedKernelDensityEstimator, CVKernelDensityEstimator


class Agent:
    def __init__(self,
                 action_spec: dm_env.specs.DiscreteArray,
                 observation_spec: dm_env.specs.Array,
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
        action_size = action_spec.num_values
        state_size = np.prod(observation_spec.shape)
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = settings['batch_size']
        self.noisy_nets = settings['qnet_settings']['noisy_nets']

        self.qnet = Dqn(state_size, action_size, settings['qnet_settings']).to(device)
        self.q_target = Dqn(state_size, action_size, settings['qnet_settings']).to(device)

        self.q_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=settings['lr'])

        self.epsilon = settings["epsilon_start"]
        self.decay = settings["epsilon_decay"]
        self.epsilon_min = settings["epsilon_min"]
        self.gamma = settings['gamma']

        self.start_optimization = settings["start_optimization"]
        self.update_qnet_every = settings["update_qnet_every"]
        self.update_target_every = settings["update_target_every"]
        self.number_steps = 0
        self.ddqn = settings["ddqn"]

        # Initialize replay memory
        self.prioritized_replay = settings["prioritized_buffer"]
        if self.prioritized_replay:
            self.memory = PrioritizedReplayMemory(device, settings["buffer_size"], self.gamma, settings["n_steps"],
                                                  settings["alpha"], settings["beta0"], settings["beta_increment"])
        else:
            self.memory = ReplayMemory(device, settings["buffer_size"], self.gamma, settings["n_steps"])

        # Density Estimator
        self.features = 'd'
        self.DE_type = 'KDE'

        if self.DE_type == 'flow':
            self.density_estimator = MAFMOGDensityEstimator(batch_size=50, n_components=3, n_blocks=5, lr=1e-4,
                                                            use_log_density=True,
                                                            use_density_scaling=True)
        elif self.DE_type == 'KDE':
            # self.density_estimator = FixedKernelDensityEstimator('gaussian', 0.1, use_log_density = True)
            self.density_estimator = CVKernelDensityEstimator(use_log_density=True)

        # Epistemic predictor
        self.enet = Epn((state_size + len(self.features)) - 1 if "x" in self.features else len(self.features),
                        action_size, settings['qnet_settings']).to(device)
        self.e_optimizer = optim.Adam(self.enet.parameters(), lr=settings['lr'])

        self.burn_in_density = 10000
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
            if self.number_steps <= self.burn_in_density:
                qvals = self.qnet.forward(observation)
            else:
                qvals = self.qnet.forward(observation) + 0.1 * self._epistemic_uncertainty(observation.unsqueeze(0))
            return int(torch.argmax(qvals, dim=-1).cpu().detach().numpy())

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
        # print('q_observed is cuda', q_observed.is_cuda)
        # print('q_target is cuda', q_target.is_cuda)

        losses = functional.mse_loss(q_observed, q_target, reduction='none')
        loss = (weights * losses).sum() / weights.sum()
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

        observation = np.array(step.observation).flatten()
        next_observation = np.array(next_step.observation).flatten()
        done = next_step.last()
        exp = Experience(observation,
                         action,
                         next_step.reward,
                         next_step.discount,
                         next_observation,
                         0,
                         done
                         )
        self.memory.add(exp)

        if self.memory.number_samples() < self.start_optimization:
            return

        if self.number_steps % self.update_qnet_every == 0:
            s0, a0, n_step_reward, discount, s1, _, dones, indices, weights = self.memory.sample_batch(self.batch_size)
            self.optimization_step(s0, a0, n_step_reward, discount, s1, indices, weights)

        if self.number_steps % self.update_target_every == 0:
            self.q_target.load_state_dict(self.qnet.state_dict())
        return

    def optimization_step(self,
                          s0: torch.Tensor,
                          a0: torch.Tensor,
                          n_step_reward: torch.Tensor,
                          discount: torch.Tensor,
                          s1: torch.Tensor,
                          indices: typing.Optional[torch.Tensor],
                          weights: typing.Optional[torch.Tensor]) -> None:
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
            next_q_vals = self.q_target(s1)
            if self.ddqn:
                a1 = torch.argmax(self.qnet(s1), dim=1).unsqueeze(-1)
                next_q_val = next_q_vals.gather(1, a1).squeeze()
            else:
                next_q_val = torch.max(next_q_vals, dim=1).values
            q_target = n_step_reward.squeeze() + self.gamma * discount.squeeze() * next_q_val

        # Getting the observed q-values
        if self.noisy_nets:
            self.qnet.reset_noise()
        q_observed = self.qnet(s0).gather(1, a0.long()).squeeze()

        # Calculating the losses
        if not self.prioritized_replay:
            weights = torch.ones(self.batch_size)
        critic_loss, batch_loss = self.calc_loss(q_observed, q_target, weights.to(self.device))

        # Backpropagation of the gradients
        self.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 5)
        self.optimizer.step()

        # Update density estimator
        if self.number_steps % self.burn_in_density == 0:
            s0_for_d, a0_for_d, _, _, _, _, _, _, _ = self.memory.sample_batch(self.burn_in_density)
            self.density_estimator.fit(s0_for_d.cpu())
            if hasattr(self.density_estimator, 'kde'):
                print('steps: {}, DE fitted: {}, bandwifth: {}'.format(self.number_steps, self.density_estimator.kde,
                                                                       self.density_estimator.kde.bandwidth))

        # Update Enet
        if self.memory.number_samples() > self.burn_in_density:
            e_observed = self._epistemic_uncertainty(s0).gather(1, a0.long()).squeeze()
            e_loss, e_batch_loss = self.calc_loss(e_observed, torch.tensor(batch_loss).to(self.device),
                                                  weights.to(self.device))
            if self.number_steps % self.burn_in_density == 0:
                # print("steps, Top k samples from Qnet: batch_loss:", self.number_steps, torch.topk(torch.tensor(batch_loss), 10))
                # print("Top k samples from Enet, e_observed:", torch.topk(e_observed, 10))
                print('steps, e_loss', self.number_steps, e_loss)
                # print('density', self.density_estimator.score_samples(s0.cpu()).to(self.device))

            self.e_optimizer.zero_grad()
            e_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.enet.parameters(), 5)
            self.e_optimizer.step()

        # Update replay memory
        self.memory.update_priorities(indices, batch_loss)
        return

    def _epistemic_uncertainty(self, x):
        """
        Computes uncertainty for input sample and
        returns epistemic uncertainty estimate.
        """
        u_in = []
        if 'x' in self.features:
            u_in.append(x)
        if 'd' in self.features:
            density_feature = self.density_estimator.score_samples(x.cpu()).to(self.device)
            u_in.append(density_feature)
        u_in = torch.cat(u_in, dim=1)
        return self.enet.forward(u_in)

    def pretrain_density_estimator(self, x):
        """
        Trains density estimator on input samples
        """

        self.density_estimator.fit(x.cpu())
