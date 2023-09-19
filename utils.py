import torch
import numpy as np
import random
from collections import namedtuple, deque, Counter

from scipy.optimize import minimize

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    eff_bs = 1/np.sum([weight**2 for weight in weights])
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



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, opt, action_size, seed, device, mask=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.opt = opt
        self.action_size = action_size
        self.memory = deque(maxlen=opt.buffer_size)  
        self.batch_size = opt.batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
        self.mask = mask
    
    def add(self, state, action, reward, next_state, done, mask=None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, sample_size=None):
        """Randomly sample a batch of experiences from memory."""
        if sample_size is None:
            sample_size = self.batch_size

        if sample_size > len(self.memory):
            sample_size = len(self.memory)

        experiences = random.sample(self.memory, k=sample_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        if self.mask:
            if self.opt.mask == "sampling":
                effective_batch_size = self.opt.batch_size*self.opt.mask_prob
                masks = np.zeros((self.opt.batch_size, self.opt.num_nets))
                for i in range(self.opt.num_nets):
                    masks[:effective_batch_size, i] = 1 
                    random.shuffle(masks[:, i]) 
                masks = torch.from_numpy(masks).to(self.device).bool()
            else:
                masks = torch.from_numpy(np.vstack([e.mask for e in experiences if e is not None])).to(self.device).bool()
            return (states, actions, rewards, next_states, dones, masks)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class MaskReplayBuffer(ReplayBuffer):
    def __init__(self, opt, action_size, seed, device):
        super().__init__(opt, action_size, seed, device)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "mask"])

    def add(self, state, action, reward, next_state, done, mask=None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, mask)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        masks = torch.from_numpy(np.vstack([e.mask for e in experiences if e is not None])).bool().to(self.device)
        return (states, actions, rewards, next_states, dones, masks)



class RiskReplayBuffer(ReplayBuffer):
    def __init__(self, opt, action_size, seed, device):
        super().__init__(opt, action_size, seed, device)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "risk"])

    def add(self, state, action, reward, next_state, done, risk=None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, risk)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        masks = torch.from_numpy(np.vstack([e.mask for e in experiences if e is not None])).bool().to(self.device)
        return (states, actions, rewards, next_states, dones, masks)


class ReplayBuffer2:
        def __init__(self, buffer_size=100000):
                self.obs = None 
                self.next_obs = None
                self.actions = None 
                self.rewards = None 
                self.dones = None
                self.risks = None 
                self.dist_to_fails = None 
                self.costs = None

        def add(self, obs, next_obs, action, reward, done, cost, risk):
                self.obs = obs if self.obs is None else torch.concat([self.obs, obs], axis=0)
                self.next_obs = next_obs if self.next_obs is None else torch.concat([self.next_obs, next_obs], axis=0)
                self.actions = action if self.actions is None else torch.concat([self.actions, action], axis=0)
                self.rewards = reward if self.rewards is None else torch.concat([self.rewards, reward], axis=0)
                self.dones = done if self.dones is None else torch.concat([self.dones, done], axis=0)
                self.risks = risk if self.risks is None else torch.concat([self.risks, risk], axis=0)
                self.costs = cost if self.costs is None else torch.concat([self.costs, cost], axis=0)
                self.dist_to_fails = dist_to_fail if self.dist_to_fails is None else torch.concat([self.dist_to_fails, dist_to_fail], axis=0)

        def __len__(self):
                return self.obs.size()[0]
        
        def sample(self, sample_size):
                idx = range(self.obs.size()[0])
                sample_idx = np.random.choice(idx, sample_size)
                return {"obs": self.obs[sample_idx],
                        "next_obs": self.next_obs[sample_idx],
                        "actions": self.actions[sample_idx],
                        "rewards": self.rewards[sample_idx],
                        "dones": self.dones[sample_idx],
                        "risks": self.risks[sample_idx], 
                        "costs": self.costs[sample_idx],
                        "dist_to_fail": self.dist_to_fails[sample_idx]}