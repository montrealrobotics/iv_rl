from .ensembleDQN import * 
from .mcdropDQN import *

class IV_DQN(EnsembleDQN):
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

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi ))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
    	return self.iv_weights(variance)



class IV_MaskEnsembleDQN(MaskEnsembleDQN):
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

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi ))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
        return self.iv_weights(variance)


class IV_RPFMaskEnsembleDQN(RPFMaskEnsembleDQN):
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

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi ))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
        return self.iv_weights(variance)


class IV_LossAttDQN(LossAttDQN):
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

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi ))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
        return self.iv_weights(variance)



class IV_BootstrapDQN(BootstrapDQN):
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

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi ))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
        return self.iv_weights(variance)

class IV_RPFBootstrapDQN(RPFBootstrapDQN):
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

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi ))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
        return self.iv_weights(variance)



class IV_MCDropDQN(MCDropDQN):
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

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi ))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
        return self.iv_weights(variance)

class IV_Lakshminarayan(Lakshminarayan):
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

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi ))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
        return self.iv_weights(variance)

class IV_LakshmiBootstrapDQN(LakshmiBootstrapDQN):
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

    def iv_weights(self, variance):
        weights = (1. / (variance+self.xi ))
        weights /= weights.sum(0)
        return weights

    def get_mse_weights(self, variance):
        return self.iv_weights(variance)



