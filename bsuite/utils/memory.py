from collections import deque, namedtuple

import numpy as np
import torch
import typing

from .segment_tree import SumTree, MinTree

Experience = namedtuple('Experience', 'state action reward discount next_state next_action done mask')


class ReplayMemory:
    """
    ReplayMemory stores experience in form of tuples  (last_state last_action reward discount state action done)
    in a deque of maximum length buffer_size.

    Methods:
        add(self, sample): add a sample to the buffer
        sample_batch(self, batch_size): return an experience batch of size batch_size
        update_priorities(self, indices, weights): not implemented, needed for prioritizied replay buffer
        number_samples(self): returns the number of samples currently stored.
    """

    def __init__(self,
                 device: torch.device,
                 memory_size: int,
                 gamma: float,
                 number_steps: typing.Optional[torch.Tensor]) -> None:
        """
        Initializes the memory buffer

        Args:
            device(str): "gpu" or "cpu"
            memory_size(int): maximum number of elements in the ReplayMemory
            gamma(float): decay factor
            number_steps(torch.Tensor): not used yet
        """
        self.gamma = gamma
        self.number_steps = number_steps
        self.device = device

        self.data = deque(maxlen=memory_size)
        self.buffer = deque(maxlen=number_steps)
        return

    def add(self, sample: Experience) -> None:
        """
        Adds experience to a buffer. Once the buffer is at full capacity or when the episode
        is over, elements are added to the ReplayMemory.
        Args:
            sample(Experience): tuple of 'last_state last_action reward discount state action done'
        Returns:
            None
        """
        self.buffer.appendleft(sample)

        if sample.done:
            while self.buffer:
                self.add_to_memory()
                self.buffer.pop()

        if len(self.buffer) == self.number_steps:
            self.add_to_memory()
        return

    def buffer_to_experience(self):
        """
        Converts the current buffer to one experience with n-step returns.
        Returns:
            tuple with experience
        """
        buffer = self.buffer
        if len(buffer) == 0:
            return

        reward = 0.0
        for element in buffer:
            reward = element.reward + self.gamma * reward

        first_element = self.buffer[-1]
        last_element = self.buffer[0]

        exp = Experience(first_element.state,
                         first_element.action,
                         reward,
                         last_element.discount,
                         last_element.state,
                         0,
                         last_element.done, 
                         first_element.mask
                         )
        return exp

    def add_to_memory(self) -> None:
        """
            Adds experience to the memory after calculating the n-step returns.
        Args:

        Returns:

        """
        exp = self.buffer_to_experience()
        self.data.append(exp)
        return

    def sample_batch(self, batch_size: int) -> tuple:
        """
        Samples a batch of size batch_size and returns a tuple of PyTorch tensors.
        Args:
            batch_size(int):  number of elements for the batch

        Returns:
            tuple of tensors
        """
        number_elements = len(self.data)
        indices = np.random.randint(0, number_elements, batch_size)
        states, actions, rewards, discounts, next_states, next_actions, dones, masks = self.get_batch(indices)
        return tuple((states, actions, rewards, discounts, next_states, next_actions, dones, None, None, masks))

    def get_batch(self, indices: np.array) -> tuple:
        """

        Args:
            indices: indices of the data that should be returned

        Returns:

        """
        states, actions, masks = [], [], []
        next_states, next_actions = [], []
        rewards, discounts, dones = [], [], []
        for index in indices:
            experience = self.data[index]
            states.append(experience.state)
            actions.append(experience.action)
            rewards.append(experience.reward)
            discounts.append(experience.discount)
            next_states.append(experience.next_state)
            next_actions.append(experience.next_action)
            dones.append(experience.done)
            masks.append(experience.mask)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).float().unsqueeze(-1).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(-1).to(self.device)
        discounts = torch.from_numpy(np.array(discounts).astype(np.float32)).float().unsqueeze(-1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        next_actions = torch.from_numpy(np.array(next_actions)).float().unsqueeze(-1).to(self.device)
        dones = torch.from_numpy(np.array(dones)).bool().unsqueeze(-1).to(self.device)
        masks = torch.from_numpy(np.array(masks)).bool().unsqueeze(-1).to(self.device)
        return tuple((states, actions, rewards, discounts, next_states, next_actions, dones, masks))

    def update_priorities(self, indices: typing.Optional[np.array], priorities: typing.Optional[np.array]):
        """
        This method later needs to be implemented for prioritized experience replay.
        Args:
            indices(list(int)): list of integers with the indices of the experience tuples in the batch
            priorities(list(float)): priorities of the samples in the batch

        Returns:
            None
        """
        return

    def number_samples(self):
        """
        Returns:
              Number of elements in the Replay Memory
        """
        return len(self.data)


class PrioritizedReplayMemory(ReplayMemory):
    """
    Implemented the prioritized replay buffer according to "Schaul, Tom, et al. "Prioritized experience replay."
    arXiv preprint arXiv:1511.05952 (2015)." and inspired by https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py.

    Methods:
        add(self, sample): add a sample to the buffer
        sample_batch(self, batch_size): return an experience batch of size batch_size
        update_priorities(self, indices, weights): not implemented, needed for prioritizied replay buffer
        number_samples(self): returns the number of samples currently stored.
    """

    def __init__(self,
                 device: torch.device,
                 memory_size: int,
                 gamma: float,
                 number_steps: typing.Optional[torch.Tensor],
                 alpha: float,
                 beta: float,
                 beta_increment: float) -> None:
        """
        Initializes the memory buffer

        Args:
            device(str): "gpu" or "cpu"
            memory_size(int): maximum number of elements in the ReplayMemory
            gamma(float): decay factor
            number_steps(torch.Tensor): not used yet
        """
        super(PrioritizedReplayMemory, self).__init__(device, memory_size, gamma, number_steps)
        self.memory_size = memory_size

        self.index = 0
        self.data = []
        self.buffer = deque(maxlen=number_steps)

        self.max_prio = 1
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.sum_tree = SumTree(memory_size)
        self.min_tree = MinTree(memory_size)
        return

    def add_to_memory(self) -> None:
        """
            Adds experience to the memory after calculating the n-step return.
        Args:

        Returns:

        """
        exp = self.buffer_to_experience()
        if self.number_samples() < self.memory_size:
            self.data.append(exp)
        else:
            self.data[self.index] = exp

        weight = self.max_prio ** self.alpha
        self.sum_tree.add(self.index, weight)
        self.min_tree.add(self.index, weight)
        self.index = (self.index + 1) % self.memory_size
        return

    def sample_batch(self, batch_size: int) -> tuple:
        """
        Samples a batch of size batch_size according to their priority.
        Args:
            batch_size(int):  number of elements for the batch

        Returns:
            tuple of tensors with experiences
        """

        interval_prio = self.sum_tree.root / batch_size
        indices = []
        for i in range(batch_size):
            low = i * interval_prio
            high = (i + 1) * interval_prio
            prio = low + (high - low) * np.random.rand()
            indices.append(self.sum_tree.get_index(prio))

        min_prio = self.min_tree.root / self.sum_tree.root
        max_weight = (self.number_samples() * min_prio) ** -self.beta
        weights = (self.number_samples() * self.sum_tree.get_elements(indices)) ** -self.beta / max_weight
        weights = torch.tensor(weights).to(self.device)

        if self.beta < 1.0:
            self.beta = self.beta + self.beta_increment

        states, actions, rewards, discounts, next_states, next_actions, dones = self.get_batch(np.array(indices))
        return tuple((states, actions, rewards, discounts, next_states, next_actions, dones, indices, weights))

    def update_priorities(self, indices: typing.Optional[np.array], priorities: typing.Optional[np.array]):
        """
        Updates the priorities of the replay buffer.
        Args:
            indices(list(int)): list of integers with the indices of the experience tuples in the batch
            priorities(list(float)): priorities of the samples in the batch

        Returns:
            None
        """
        for index, prio in zip(indices, priorities):
            if self.max_prio < prio:
                self.max_prio = prio
            self.sum_tree.update(index, prio ** self.alpha)
            self.min_tree.update(index, prio ** self.alpha)
        return
