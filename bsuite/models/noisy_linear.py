# Implementation of the noisy linear class.
# The use of PyTorch methods to build the noisy linear layer is loosely based on the example from
# Deep Reinforcement Learning Hands-on by Maxim Lapan, 2018.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class NoisyLinear(nn.Linear):
    def __init__(self, in_size: int, out_size: int) -> None:
        """
        Args:
            in_size(int): input size of the neural network
            out_size(int): output size of the neural network
        """
        super(NoisyLinear, self).__init__(in_size, out_size)
        self.in_size = in_size
        self.out_size = out_size
        self.mu = 1.0 / np.sqrt(in_size)
        self.sigma = 0.5 / np.sqrt(in_size)

        torch.nn.init.uniform_(self.weight, a=-self.mu, b=self.mu)
        torch.nn.init.uniform_(self.bias, a=-self.mu, b=self.mu)

        self.sigma_w = nn.Parameter(torch.full((out_size, in_size), self.sigma), True)
        self.sigma_b = nn.Parameter(torch.full((out_size,), self.sigma), True)
        self.register_buffer("noise_in", torch.zeros(in_size))
        self.register_buffer("noise_out", torch.zeros(out_size))
        self.register_buffer("epsilon_w", torch.zeros(out_size, in_size))
        self.register_buffer("epsilon_b", torch.zeros(out_size))
        self.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the neural network.
        Args:
            x(torch.Tensor): the current observation

        Returns:
            torch.Tensor with the result of the neural network
        """
        # forward pass
        w = self.weight + self.sigma_w * self.epsilon_w
        b = self.bias + self.sigma_b * self.epsilon_b
        return functional.linear(x, w, b)

    def reset_noise(self):
        """
        Samples input noise and output noise and uses these two vectors to calculate factorized noise.
        Returns:
            Nothing
        """
        self.noise_in.normal_()
        self.noise_out.normal_()
        # calculate noise
        noise_w = torch.ger(self.noise_out, self.noise_in)
        noise_b = self.noise_out
        self.epsilon_w = torch.sign(noise_w) * torch.sqrt(torch.abs(noise_w))
        self.epsilon_b = torch.sign(noise_b) * torch.sqrt(torch.abs(noise_b))
