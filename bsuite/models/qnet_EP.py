import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from .noisy_linear import NoisyLinear


class Enet(nn.Module):
    def __init__(self) -> None:
        super(Enet, self).__init__()
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


class Epn(Enet):
    def __init__(self, states_size: np.ndarray, action_size: np.ndarray, settings: dict) -> None:
        """
        Initializes the neural network.
        Args:
            states_size: Size of the input space.
            action_size:Size of the action space.
            settings: dictionary with settings
        """
        super(Epn, self).__init__()
        self.batch_size = settings["batch_size"]
        self.noisy_net = settings['noisy_nets']
        layers_size = settings["layers_sizes"][0] * 2
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
        x = functional.relu(self.FC2(x))
        return functional.softplus(self.FC3(x))

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
