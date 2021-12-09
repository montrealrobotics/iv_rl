from .ensembleDQN import * 
from .mcdropDQN import *


class UWAC_DQN(EnsembleDQN):
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
        self.beta = opt.uwac_beta
        self.use_exp_weight = opt.use_exp_weight
        self.clip_bottom = opt.clip_bottom
        self.clip_top = opt.clip_top
        self.factor = 1

    def uwac_weights(self, variance):
        weight = torch.clamp(self.beta*self.factor/variance, self.clip_bottom, self.clip_top)
        return weight


    def get_mse_weights(self, variance):
    	return self.uwac_weights(variance)




class UWAC_LakshmiBootstrapDQN(LakshmiBootstrapDQN):
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
        self.beta = opt.uwac_beta
        self.use_exp_weight = opt.use_exp_weight
        self.clip_bottom = opt.clip_bottom
        self.clip_top = opt.clip_top
        self.factor = 1

    def uwac_weights(self, variance):
        weight = torch.clamp(self.beta*self.factor/variance, self.clip_bottom, self.clip_top)
        return weight

    def get_mse_weights(self, variance):
        return self.uwac_weights(variance)

