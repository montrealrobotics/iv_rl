import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from .noisy_linear import NoisyLinear


class Qnet(nn.Module):
    def __init__(self) -> None:
        super(Qnet, self).__init__()
        return

    def get_max_action(self, observation: torch.Tensor) -> int:
        """
        Get the action with the maximum q-value for an observation.
        Args:
            observation(torch.Tensor): an observation
        Returns:
            int: action with the maximum q-value for the current state
        """
        qvals = self.forward(observation)
        return int(torch.argmax(qvals, dim=-1).cpu().detach().numpy())


class Dqn(Qnet):
    def __init__(self, states_size: np.ndarray, action_size: np.ndarray, settings: dict, seed:int) -> None:
        """
        Initializes the neural network.
        Args:
            states_size: Size of the input space.
            action_size:Size of the action space.
            settings: dictionary with settings
        """
        super(Dqn, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.batch_size = settings["batch_size"]
        self.noisy_net = settings['noisy_nets']
        self.drop_porb = settings['drop_prob']
        layers_size = settings["layers_sizes"][0]
        if not self.noisy_net:
            self.FC1 = nn.Linear(int(states_size), layers_size)
            self.FC2 = nn.Linear(layers_size, layers_size)
            self.FC3 = nn.Linear(layers_size, int(action_size))
        else:
            self.FC1 = NoisyLinear(int(states_size), layers_size )
            self.FC2 = NoisyLinear(layers_size, layers_size)
            self.FC3 = NoisyLinear(layers_size, int(action_size))
        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward step of the neural network
        Args:
            x(torch.Tensor): observation or a batch of observations

        Returns:
            torch.Tensor: q-values for all observations and actions, size: batch_size x actions_size
        """
        x = functional.relu(self.FC1(x))
        x = functional.dropout(x, training=True, p=self.drop_porb)
        x = functional.relu(self.FC2(x))
        x = functional.dropout(x, training=True, p=self.drop_porb)
        return self.FC3(x)

    def reset(self) -> None:
        """
        Resets the weights of the neural network layers.
        Returns:
            None
        """
        torch.nn.init.xavier_uniform_(self.FC1.weight.data)
        torch.nn.init.xavier_uniform_(self.FC2.weight.data)
        torch.nn.init.xavier_uniform_(self.FC3.weight.data)
        if self.noisy_net:
            self.reset_noise()

    def reset_noise(self) -> None:
        """
        Resets the noise of the noisy layers.
        """
        self.FC1.reset_noise()
        self.FC2.reset_noise()
        self.FC3.reset_noise()


class TwoHeadDqn(Qnet):
    def __init__(self, states_size: np.ndarray, action_size: np.ndarray, settings: dict, seed:int) -> None:
        """
        Initializes the neural network.
        Args:
            states_size: Size of the input space.
            action_size:Size of the action space.
            settings: dictionary with settings
        """
        super(TwoHeadDqn, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.batch_size = settings["batch_size"]
        self.noisy_net = settings['noisy_nets']
        self.drop_porb = settings['drop_prob']
        layers_size = settings["layers_sizes"][0]
        if not self.noisy_net:
            self.FC1 = nn.Linear(int(states_size), layers_size)
            self.FC2 = nn.Linear(layers_size, layers_size)
            self.FC3_mu = nn.Linear(layers_size, int(action_size))
            self.FC3_logstd = nn.Linear(layers_size, int(action_size))
        else:
            self.FC1 = NoisyLinear(int(states_size), layers_size )
            self.FC2 = NoisyLinear(layers_size, layers_size)
            self.FC3_mu = nn.Linear(layers_size, int(action_size))
            self.FC3_logstd = nn.Linear(layers_size, int(action_size))
        self.reset()

    def forward(self, x: torch.Tensor, is_training=False) -> torch.Tensor:
        """
        Forward step of the neural network
        Args:
            x(torch.Tensor): observation or a batch of observations

        Returns:
            torch.Tensor: q-values for all observations and actions, size: batch_size x actions_size
        """
        x = functional.relu(self.FC1(x))
        x = functional.dropout(x, training=True, p=self.drop_porb)
        x = functional.relu(self.FC2(x))
        x = functional.dropout(x, training=True, p=self.drop_porb)
        logstd = torch.clamp(self.FC3_logstd(x), -10, 10)
        if is_training:
            return self.FC3_mu(x), torch.exp(logstd)
        else:
            return self.FC3_mu(x)#, torch.exp(self.FC3_logstd(x))

    def reset(self) -> None:
        """
        Resets the weights of the neural network layers.
        Returns:
            None
        """
        torch.nn.init.xavier_uniform_(self.FC1.weight.data)
        torch.nn.init.xavier_uniform_(self.FC2.weight.data)
        torch.nn.init.xavier_uniform_(self.FC3_mu.weight.data)
        torch.nn.init.xavier_uniform_(self.FC3_logstd.weight.data)
        if self.noisy_net:
            self.reset_noise()

    def reset_noise(self) -> None:
        """
        Resets the noise of the noisy layers.
        """
        self.FC1.reset_noise()
        self.FC2.reset_noise()
        self.FC3_mu.reset_noise()
        self.FC3_logstd.reset_noise()



class DuelDQN(Qnet):
    def __init__(self, states_size: np.ndarray, action_size: np.ndarray, settings: dict) -> None:
        """
        Initializes the neural network.
        Args:
            states_size: Size of the input space.
            action_size:Size of the action space.
            settings: dictionary with settings, currently not used.
        """
        super(DuelDQN, self).__init__()
        self.batch_size = settings["batch_size"]
        layers_size = settings["layers_sizes"][0]
        self.noisy_net = settings['noisy_nets']
        if not self.noisy_net:
            self.FC1 = nn.Linear(int(states_size), layers_size)
            self.FC2 = nn.Linear(layers_size, layers_size)
            self.FC3v = nn.Linear(layers_size, 1)
            self.FC3a = nn.Linear(layers_size, int(action_size))
        else:
            self.FC1 = NoisyLinear(int(states_size), layers_size)
            self.FC2 = NoisyLinear(layers_size, layers_size)
            self.FC3v = NoisyLinear(layers_size, 1)
            self.FC3a = NoisyLinear(layers_size, int(action_size))
        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward step of the duelling q-network
        Args:
            x(torch.Tensor): observation or a batch of observations

        Returns:
            torch.Tensor: q-values for all  observations and actions
        """
        x = functional.relu(self.FC1(x))
        x = functional.relu(self.FC2(x))
        v = self.FC3v(x)
        a = self.FC3a(x)
        if x.ndimension() == 1:
            qvals = v + (a - torch.mean(a))
        else:
            qvals = v + (a - torch.mean(a, 1, True))
        return qvals

    def reset(self) -> None:
        """
        Resets the weights of the neural network layers.
        Returns:
            None
        """
        torch.nn.init.xavier_uniform_(self.FC1.weight.data)
        torch.nn.init.xavier_uniform_(self.FC2.weight.data)
        torch.nn.init.xavier_uniform_(self.FC3a.weight.data)
        torch.nn.init.xavier_uniform_(self.FC3v.weight.data)
        if self.noisy_net:
            self.reset_noise()

    def reset_noise(self) -> None:
        """
        Resets the noise of the noisy layers.
        """
        self.FC1.reset_noise()
        self.FC2.reset_noise()
        self.FC3a.reset_noise()
        self.FC3v.reset_noise()
