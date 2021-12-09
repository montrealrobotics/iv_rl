import bsuite
from bsuite import sweep
from bsuite.baselines import experiment
from bsuite.baselines.utils import pool

import os
import wandb
import torch
import warnings
import numpy as np
import random
from argparse import ArgumentParser

from config import config

warnings.filterwarnings('ignore')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = ArgumentParser()

# Main arguments for optimization
parser.add_argument("--model", default='deup_dqn',
                    help='name of the agent')
parser.add_argument("--env", default='cartpole/0',
                    help='name of the env')
parser.add_argument("--save_path", default="runs" ,
                    help='Path were the results are saved.')
parser.add_argument("--batch_size", type=int, default=64,
                    help="batch size")
parser.add_argument("--eff_batch_size", type=int, default=64,
                    help="effective batch size")
parser.add_argument("--num_workers", default=3,
                    help="Number of parallel threads to run")
parser.add_argument("--net_seed", default=42, type=int,
                    help="set seed for reproducibility")
parser.add_argument("--dynamic_xi", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="whether to use calculated xi using minimum effective batch size")
parser.add_argument("--minimal_eff_bs_ratio", type=float, default=1.0, 
                    help="Minimal Effective Batch Ratio for calculating Policy Improvement Noise")
parser.add_argument("--xi", type=float, default=5.,
                    help="Minimum Variance value")
parser.add_argument("--mask_prob", default=1.0, type=float,
                    help="how samples are masked to generate diversity across the ensemble")
parser.add_argument("--num_episodes", default=300, type=int,
                    help="Max number of episodes")
parser.add_argument("--lossatt_weight", default=1, type=float,
                    help="Loss Attenuation weight")
parser.add_argument("--sunrise_temp", type=float, default=20.0,
                    help="sunrise temperature for weighted Bellman backup")

args = parser.parse_args()


# Setting seed
np.random.seed(args.net_seed)
# torch.manual_seed(args.seed)
random.seed(args.net_seed)


if args.mask_prob < 1.0:
    args.batch_size = int(args.batch_size / args.mask_prob)


print(config.keys(), args.env.split("/")[0])
try:
    config_env = config[args.env.split("/")[0]][args.model]
    for key in config_env.keys():
        setattr(args, key, config_env[key])
except:
    pass
    

save_path = os.path.join(args.save_path, args.env.split("/")[0], "%s_bs_%d_mask_%.1f_mebs%2f_lossatt_%.1f"%(args.model, args.batch_size, args.mask_prob, args.minimal_eff_bs_ratio, args.lossatt_weight))

drop_prob = 0
if args.model  == 'DQN':
    from bsuite.models.agent import DQN as Agent
elif args.model == 'VarDQN':
    from bsuite.models.agent import LossAttDQN as Agent
elif args.model == 'IV_VarDQN':
    from bsuite.models.agent import IV_LossAttDQN as Agent
elif args.model == 'BootstrapDQN':
    from bsuite.models.agent_bootdqn import BootstrapDQN as Agent
elif args.model == 'EnsembleDQN':
    from bsuite.models.agent_bootdqn import EnsembleDQN as Agent
elif args.model == "VarEnsembleDQN":
    from bsuite.models.agent_bootdqn import LakshmiBootDQN as Agent
elif args.model == 'IV_EnsembleDQN':
    from bsuite.models.agent_bootdqn import IV_DQN as Agent
elif args.model == "IV_BootstrapDQN":
    from bsuite.models.agent_bootdqn import IV_BootstrapDQN as Agent
elif args.model == "SunriseDQN":
    from bsuite.models.agent_bootdqn import SunriseDQN as Agent
elif args.model in ["IV_VarEnsembleDQN", "IV_DQN"]:
    from bsuite.models.agent_bootdqn import IV_LakshmiBootDQN as Agent
else:
    print('This agent is not implemented!!')


def run(bsuite_id: str) -> str:
    """
    Runs a bsuite experiment and saves the results as csv files

    Args:
        bsuite_id: string, the id of the bsuite experiment to run

    Returns: none

    """
    print(bsuite_id)
    train_env = bsuite.load_and_record(
        bsuite_id=bsuite_id,
        save_path=save_path,
        logging_mode='terminal',
    )

    test_env = bsuite.load_and_record(
        bsuite_id=bsuite_id,
        save_path=save_path,
        logging_mode='csv',
        overwrite=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Settings for the neural network
    qnet_settings = {'layers_sizes': [50], 'batch_size': args.batch_size, 'noisy_nets': False, 'distributional': False, 'vmin': 0,
                     'vmax': 1000, 'number_atoms': 51, 'drop_prob': drop_prob}

    # Settings for the specific agent
    settings = {'batch_size': qnet_settings["batch_size"], 'epsilon_start': 0.05, 'epsilon_decay': 0.5,
                'epsilon_min': 0.05, 'gamma': 0.99, 'buffer_size': 2 ** 16, 'lr': 1e-3, 'qnet_settings': qnet_settings,
                'start_optimization': 64, 'update_qnet_every': 2, 'update_target_every': 50, 'ddqn': False, 'n_steps': 4,
                'duelling_dqn': False, 'prioritized_buffer': False, 'alpha': 0.6, 'beta0': 0.4, 'beta_increment': 1e-6,
                'dynamic_xi': args.dynamic_xi, 'minimal_eff_bs_ratio': args.minimal_eff_bs_ratio, 'xi': args.xi,
                'mask_prob': args.mask_prob}
 
    # if args.agent == 'boot_dqn':
    #     agent = Agent(obs_spec=env.observation_spec(),
    #                   action_spec=env.action_spec(), num_ensemble=5
    #                   )
    # else:
    agent = Agent(args, action_spec=train_env.action_spec(),
                      observation_spec=train_env.observation_spec(),
                      device=device,
                      num_ensemble=5, 
                      net_seed=42,
                      settings=settings
                      )
    # print("Bsuite Num Epsiodes: ", env.bsuite_num_episodes)

    experiment.run(
        agent=agent,
        train_environment=train_env,
        test_environment=test_env,
        num_episodes=args.num_episodes,
        verbose=True)
    return bsuite_id

if __name__ == '__main__':
    run(args.env)
    #bsuite_sweep = getattr(sweep, 'CARTPOLE')
    #pool.map_mpi(run, bsuite_sweep, args.num_workers)

